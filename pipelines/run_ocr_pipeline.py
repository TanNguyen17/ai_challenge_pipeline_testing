import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["OPENCV_FFMPEG_DEBUG"] = "0"
import sys
import time
import json
import argparse
import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from collections import defaultdict
import re

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure correct Python path so it can find the 'extract' module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Any
from extract.workers.keyframe_loader import KeyframeLoader

class OCRPipelineRunner:
    """
    SOTA Video OCR Pipeline using Qwen3-VL-8B-Instruct:
    - Supports vLLM (Extremely fast, batched) or Transformers backend.
    - Uses Frame Deduplication (SSIM) to skip visually identical frames.
    - Runs 2 frames per shot for maximum coverage without performance loss (thanks to deduplication).
    """
    def __init__(self, video_dir: str, output_dir: str, keyframes_dir: str, backend: str = "vllm", batch_size: int = None):
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.keyframes_dir = keyframes_dir
        self.backend = backend
        self.batch_size = batch_size if batch_size else (16 if backend == "vllm" else 8)
        os.makedirs(output_dir, exist_ok=True)
        self.keyframe_loader = KeyframeLoader(keyframes_dir)
        self._init_model()

    def _init_model(self):
        self.qwen_id = "Qwen/Qwen3-VL-8B-Instruct"
        print(f"Loading {self.qwen_id} with backend: {self.backend.upper()}...")
        
        if self.backend == "vllm":
            try:
                from vllm import LLM, SamplingParams
                from transformers import AutoProcessor
                # Initialize vLLM with PagedAttention and Multimodal support
                self.llm = LLM(
                    model=self.qwen_id,
                    trust_remote_code=True,
                    max_model_len=4096,
                    limit_mm_per_prompt={"image": 1},
                    gpu_memory_utilization=0.9,
                    enforce_eager=True # Recommended for newer multimodal models
                )
                self.sampling_params = SamplingParams(max_tokens=256, temperature=0.0)
                self.processor = AutoProcessor.from_pretrained(self.qwen_id, trust_remote_code=True)
                print("✅ vLLM Engine loaded successfully.")
            except Exception as e:
                print(f"❌ vLLM failed to load. The exact error is:\n{e}\n\nPlease use '--backend transformers' instead.")
                sys.exit(1)
        else:
            # Fallback to standard transformers
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
            self.processor = AutoProcessor.from_pretrained(self.qwen_id, trust_remote_code=True)
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.qwen_id, 
                dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="sdpa"
            )
            self.model.eval()
            print("✅ Transformers Engine loaded successfully.")

    @staticmethod
    def _remove_accents(input_str: str) -> str:
        s = input_str.lower()
        accents = {
            'a': 'àáảạãăằắẳặẵâầấẩậẫ', 'd': 'đ', 'e': 'èéẻẹẽêềếểệễ',
            'i': 'ìíỉịĩ', 'o': 'òóỏọõôồốổộỗơờớởợỡ',
            'u': 'ùúủụũưừứửựữ', 'y': 'ỳýỷỵỹ'
        }
        for char, accented_chars in accents.items():
            for a in accented_chars:
                s = s.replace(a, char)
        return s
        
    def _resize_image_max(self, pil_img, max_pixels=1280 * 720):
        w, h = pil_img.size
        if w * h > max_pixels:
            scale = (max_pixels / (w * h)) ** 0.5
            new_w, new_h = int(w * scale), int(h * scale)
            return pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return pil_img

    def _check_duplicate(self, frame1, frame2, threshold=0.92):
        # Fast Structural Similarity Check via cv2.matchTemplate
        g1 = cv2.cvtColor(cv2.resize(frame1, (320, 180)), cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(cv2.resize(frame2, (320, 180)), cv2.COLOR_BGR2GRAY)
        sim = cv2.matchTemplate(g1, g2, cv2.TM_CCOEFF_NORMED)[0][0]
        return sim >= threshold

    def _run_vllm_batch(self, frames: List[np.ndarray]):
        if not frames: return []
        
        pil_images = [self._resize_image_max(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))) for f in frames]
        
        llm_inputs = []
        for pil_img in pil_images:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": "Hãy trích xuất tất cả văn bản tiếng Việt xuất hiện trong bức ảnh này. Phân loại chúng thành hai nhóm: 'overlay_text' (chữ được chèn lên video như tiêu đề, tin chạy) và 'scene_text' (chữ tự nhiên trong cảnh như biển hiệu, áo). Bỏ qua các logo nhỏ. Chỉ trả về JSON format: {\"overlay_text\": \"...\", \"scene_text\": \"...\"}. Không giải thích."}
                ]
            }]
            prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            llm_inputs.append({
                "prompt": prompt,
                "multi_modal_data": {"image": pil_img}
            })
            
        outputs = self.llm.generate(llm_inputs, self.sampling_params, use_tqdm=False)
        return [out.outputs[0].text for out in outputs]

    def _run_transformers_batch(self, frames: List[np.ndarray]):
        if not frames: return []
        from qwen_vl_utils import process_vision_info
        
        pil_images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
        texts = []
        image_inputs_list = []
        
        for pil_img in pil_images:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img, "max_pixels": 1280 * 720},
                    {"type": "text", "text": "Hãy trích xuất tất cả văn bản tiếng Việt xuất hiện trong bức ảnh này. Phân loại chúng thành hai nhóm: 'overlay_text' (chữ được chèn lên video như tiêu đề, tin chạy) và 'scene_text' (chữ tự nhiên trong cảnh như biển hiệu, áo). Bỏ qua các logo nhỏ. Chỉ trả về JSON format: {\"overlay_text\": \"...\", \"scene_text\": \"...\"}. Không giải thích."}
                ]
            }]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            texts.append(text)
            image_inputs, _ = process_vision_info(messages)
            if image_inputs is not None:
                image_inputs_list.extend(image_inputs)

        inputs = self.processor(
            text=texts,
            images=image_inputs_list if image_inputs_list else None,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=256)
            
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def process_video(self, video_path: str) -> Dict[str, Any]:
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        start_time = time.time()

        shots = self.keyframe_loader.load(video_id)
        if not shots:
            return {"video_id": video_id, "elapsed_sec": 0.0, "num_shots": 0, "num_documents": 0, "documents": []}
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"video_id": video_id, "elapsed_sec": 0.0, "num_shots": 0, "num_documents": 0, "documents": []}

        # ==========================================
        # STAGE 1: EXTRACT & DEDUPLICATE FRAMES
        # ==========================================
        unique_frames = []
        shot_frame_mapping = [] # stores (f_idx, shot_data, unique_idx)
        last_unique_frame = None
        last_unique_idx = -1

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        
        for shot in tqdm(shots, desc=f"Extract & Dedup {video_id}", leave=False):
            start_f = int(shot.get("start_sec", 0.0) * fps)
            end_f = int(shot.get("end_sec", shot.get("start_sec", 0.0) + 3.0) * fps)
            
            # REVERTED TO 2 FRAMES PER SHOT FOR MAXIMUM COVERAGE! 
            # (Duplicate frames will be skipped below)
            if end_f <= start_f:
                frame_indices = [shot["keyframe_id"]]
            else:
                frame_indices = [
                    start_f + int((end_f - start_f) * 0.33),
                    start_f + int((end_f - start_f) * 0.67)
                ]
            
            for f_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, frame = cap.read()
                if not ret: continue
                
                is_dup = False
                if last_unique_frame is not None:
                    if self._check_duplicate(last_unique_frame, frame, threshold=0.92):
                        is_dup = True
                
                if not is_dup:
                    last_unique_frame = frame
                    unique_frames.append((f_idx, frame))
                    last_unique_idx = len(unique_frames) - 1
                
                shot_frame_mapping.append((f_idx, shot, last_unique_idx))
                
        cap.release()

        # ==========================================
        # STAGE 2: BATCH PROCESS UNIQUE FRAMES
        # ==========================================
        unique_ocr_results = {}
        batch_size = self.batch_size
        
        print(f"[{video_id}] VLM Processing {len(unique_frames)} unique frames (Filtered from {len(shot_frame_mapping)} total)...")
        
        for i in tqdm(range(0, len(unique_frames), batch_size), desc="VLM Inference", leave=False):
            batch = unique_frames[i:i+batch_size]
            f_indices = [b[0] for b in batch]
            frames = [b[1] for b in batch]
            
            try:
                if self.backend == "vllm":
                    output_texts = self._run_vllm_batch(frames)
                else:
                    output_texts = self._run_transformers_batch(frames)
                    
                for j, out_text in enumerate(output_texts):
                    ocr_text = out_text.strip()
                    cleaned_text = re.sub(r'```json\s*', '', ocr_text)
                    cleaned_text = re.sub(r'```\s*', '', cleaned_text)
                    try:
                        ocr_data = json.loads(cleaned_text)
                        if not isinstance(ocr_data, dict):
                            ocr_data = {"overlay_text": "", "scene_text": str(ocr_data)}
                    except json.JSONDecodeError:
                        ocr_data = {"overlay_text": "", "scene_text": cleaned_text}
                        
                    unique_ocr_results[f_indices[j]] = ocr_data
            except Exception as e:
                print(f"Error processing batch: {e}")

        # ==========================================
        # STAGE 3: ROLLUP TO DOCUMENTS
        # ==========================================
        final_documents = []
        shots_data = defaultdict(lambda: {"all": [], "time_range": None})
        
        for f_idx, shot, unique_idx in shot_frame_mapping:
            u_f_idx = unique_frames[unique_idx][0]
            ocr_data = unique_ocr_results.get(u_f_idx, {"overlay_text": "", "scene_text": ""})
            
            shot_id = shot["shot_id"]
            if not shots_data[shot_id]["time_range"]:
                shots_data[shot_id]["time_range"] = {"start_sec": shot.get("start_sec", 0.0), "end_sec": shot.get("end_sec", 0.0)}
            
            shots_data[shot_id]["all"].append(ocr_data)
            
            doc = {
                "doc_type": "span",
                "video_id": video_id,
                "shot_id": shot_id,
                "tracklet_id": f"TRK_{f_idx:05d}",
                "frame_idx": f_idx,
                "time_range": {"start_sec": shot.get("start_sec", 0.0), "end_sec": shot.get("end_sec", 0.0)},
                "ocr_data": {
                    "overlay_text": str(ocr_data.get("overlay_text", "")),
                    "scene_text": str(ocr_data.get("scene_text", ""))
                },
                "confidence": 0.95
            }
            final_documents.append(doc)

        for shot_id, data in shots_data.items():
            overlay_texts = []
            scene_texts = []
            
            for ocr_data in data["all"]:
                ot_raw = ocr_data.get("overlay_text", "")
                st_raw = ocr_data.get("scene_text", "")
                
                ot = " | ".join([str(x) for x in ot_raw]) if isinstance(ot_raw, list) else str(ot_raw).strip()
                st = " | ".join([str(x) for x in st_raw]) if isinstance(st_raw, list) else str(st_raw).strip()
                if ot and ot not in overlay_texts:
                    overlay_texts.append(ot)
                if st and st not in scene_texts:
                    scene_texts.append(st)
                    
            overlay_combined = " | ".join(overlay_texts)
            scene_combined = " | ".join(scene_texts)
            
            if not overlay_combined and not scene_combined:
                continue
                
            full_text = f"{overlay_combined} {scene_combined}".strip()
                
            shot_doc = {
                "doc_type": "shot",
                "video_id": video_id,
                "shot_id": shot_id,
                "time_range": data["time_range"],
                "ocr_data_combined": {
                    "overlay_text": overlay_combined,
                    "scene_text": scene_combined,
                    "ocr_no_accent_combined": self._remove_accents(full_text)
                }
            }
            final_documents.append(shot_doc)

        elapsed_sec = round(time.time() - start_time, 2)
        return {
            "video_id": video_id,
            "elapsed_sec": elapsed_sec,
            "num_shots": len(shots),
            "num_documents": len(final_documents),
            "documents": final_documents
        }

    def run_benchmark(self, limit_videos: int = 50):
        print(f"\n🚀 Running VLM OCR Pipeline on up to {limit_videos} videos...")
        video_files = []
        for root, _, files in os.walk(self.video_dir):
            for f in files:
                if f.lower().endswith(('.mp4', '.mkv', '.avi', '.webm')):
                    video_files.append(os.path.join(root, f))
        video_files = video_files[:limit_videos]

        if not video_files:
            print("⚠️ No video files found in directory.")
            return

        out_jsonl = os.path.join(self.output_dir, "ocr_extracted_documents.jsonl")
        processed_video_ids = set()
        if os.path.exists(out_jsonl):
            with open(out_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        doc = json.loads(line)
                        processed_video_ids.add(doc.get("video_id"))
        
        pending_videos = []
        for v in video_files:
            v_id = os.path.splitext(os.path.basename(v))[0]
            if v_id not in processed_video_ids:
                pending_videos.append(v)
            else:
                print(f"⏭️ Skipping {v_id} - already processed.")
                
        video_files = pending_videos

        if not video_files:
            print("⚠️ No pending video files found to process (all done).")
            return

        total_time = 0.0
        total_shots = 0
        total_docs = 0

        with open(out_jsonl, "a", encoding="utf-8") as f:
            for idx, v_path in enumerate(video_files):
                print(f"[{idx+1}/{len(video_files)}] Processing OCR for {os.path.basename(v_path)}...")
                res = self.process_video(v_path)
                
                for doc in res["documents"]:
                    f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                f.flush()
                
                total_time += res["elapsed_sec"]
                total_shots += res["num_shots"]
                total_docs += res["num_documents"]

        benchmark_report = {
            "pipeline": f"VLM OCR (Qwen3-VL-8B, Backend: {self.backend})",
            "videos_processed": len(video_files),
            "total_elapsed_sec": round(total_time, 2),
            "avg_time_per_video_sec": round(total_time / len(video_files), 3) if video_files else 0,
            "total_shots_extracted": total_shots,
            "total_clean_ocr_documents": total_docs,
            "output_jsonl_path": out_jsonl
        }

        report_path = os.path.join(self.output_dir, "ocr_benchmark_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(benchmark_report, f, indent=2, ensure_ascii=False)

        print("\n📊 --- OCR BENCHMARK REPORT ---")
        print(json.dumps(benchmark_report, indent=2, ensure_ascii=False))
        print(f"✅ Saved OCR extracted documents to: {out_jsonl}")

def main():
    parser = argparse.ArgumentParser(description="Run VLM OCR Pipeline")
    parser.add_argument("--video-dir", type=str, default="./data/extracted", help="Directory containing raw videos")
    parser.add_argument("--output-dir", type=str, default="./data/processed/ocr", help="Output directory for OCR records")
    parser.add_argument("--keyframes-dir", type=str, default="./data/extracted/video batch 1/map-keyframes-aic25-b1/map-keyframes", help="Directory containing BTC keyframe CSVs")
    parser.add_argument("--limit", type=int, default=50, help="Maximum number of videos to process")
    parser.add_argument("--backend", type=str, choices=["vllm", "transformers"], default="vllm", help="Inference backend engine (vLLM is ~3x faster)")
    parser.add_argument("--batch-size", type=int, default=None, help="Force a specific batch size (e.g., 32 or 64 for A100)")
    args = parser.parse_args()

    runner = OCRPipelineRunner(args.video_dir, args.output_dir, args.keyframes_dir, backend=args.backend, batch_size=args.batch_size)
    runner.run_benchmark(args.limit)

if __name__ == "__main__":
    main()
