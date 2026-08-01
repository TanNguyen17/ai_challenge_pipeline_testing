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
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForCausalLM, AutoProcessor, Qwen3VLForConditionalGeneration

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from typing import List, Dict, Any
from extract.workers.keyframe_loader import KeyframeLoader

class OCRPipelineRunner:
    """
    SOTA Video OCR Pipeline using Qwen3-VL-7B-Instruct:
    Stage 1: Keyframe Sampling from BTC CSVs
    Stage 2: Batched VLM Text Extraction (Qwen3-VL)
    Stage 3: Elasticsearch Document Export (Span & Shot Rollup)
    """
    def __init__(self, video_dir: str, output_dir: str, keyframes_dir: str, device: str = "cuda"):
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.keyframes_dir = keyframes_dir
        self.device = device
        os.makedirs(output_dir, exist_ok=True)
        self.keyframe_loader = KeyframeLoader(keyframes_dir)
        self._init_qwen_vl()

    def _init_qwen_vl(self):
        self.model = None
        self.processor = None
        try:
            print("Loading Qwen3-VL-8B-Instruct...")
            qwen_id = "Qwen/Qwen3-VL-8B-Instruct"
            self.processor = AutoProcessor.from_pretrained(qwen_id, trust_remote_code=True)
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                qwen_id, 
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                device_map="cuda" if self.device == "cuda" else "cpu",
                trust_remote_code=True
            )
            self.model.eval()
            print("✅ Qwen3-VL-8B-Instruct loaded on GPU for OCR.")
        except Exception as e:
            print(f"❌ Failed to load Qwen3-VL: {e}")
            print("Please ensure you have run: uv pip install transformers>=4.45.0 qwen-vl-utils")

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

    def process_video(self, video_path: str) -> Dict[str, Any]:
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        start_time = time.time()

        shots = self.keyframe_loader.load(video_id)
        if not shots:
            return {"video_id": video_id, "elapsed_sec": 0.0, "num_shots": 0, "num_documents": 0, "documents": []}
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"video_id": video_id, "elapsed_sec": 0.0, "num_shots": 0, "num_documents": 0, "documents": []}

        batch_size = 4  # Safe for 24GB VRAM
        frame_batch = []
        shot_batch = []
        
        final_documents = []
        shots_data = defaultdict(lambda: {"all": [], "time_range": None})

        def process_batch(f_batch, s_batch):
            if not self.model or not f_batch: return
            
            try:
                # Convert OpenCV BGR to PIL RGB
                pil_images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in f_batch]
                
                texts = []
                image_inputs_list = []
                video_inputs_list = []
                
                for pil_img in pil_images:
                    messages = [{
                        "role": "user",
                        "content": [
                            {"type": "image", "image": pil_img},
                            {"type": "text", "text": "Hãy trích xuất tất cả văn bản tiếng Việt xuất hiện trong bức ảnh này. Phân loại chúng thành hai nhóm: 'overlay_text' (chữ được chèn lên video như tiêu đề, tin chạy) và 'scene_text' (chữ tự nhiên trong cảnh như biển hiệu, áo). Bỏ qua các logo nhỏ. Chỉ trả về JSON format: {\"overlay_text\": \"...\", \"scene_text\": \"...\"}. Không giải thích."}
                        ]
                    }]
                    text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    texts.append(text)
                    
                    image_inputs, video_inputs = process_vision_info(messages)
                    if image_inputs is not None:
                        image_inputs_list.extend(image_inputs)
                    if video_inputs is not None:
                        video_inputs_list.extend(video_inputs)

                inputs = self.processor(
                    text=texts,
                    images=image_inputs_list if image_inputs_list else None,
                    videos=video_inputs_list if video_inputs_list else None,
                    padding=True,
                    return_tensors="pt"
                ).to(self.model.device)
                
                with torch.no_grad():
                    generated_ids = self.model.generate(**inputs, max_new_tokens=256)
                    
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_texts = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                
                for i, out_text in enumerate(output_texts):
                    ocr_text = out_text.strip()
                    
                    # Robust JSON parsing from VLM output
                    cleaned_text = re.sub(r'```json\s*', '', ocr_text)
                    cleaned_text = re.sub(r'```\s*', '', cleaned_text)
                    try:
                        ocr_data = json.loads(cleaned_text)
                        if not isinstance(ocr_data, dict):
                            ocr_data = {"overlay_text": "", "scene_text": str(ocr_data)}
                    except json.JSONDecodeError:
                        ocr_data = {"overlay_text": "", "scene_text": cleaned_text}
                        
                    f_idx, shot = s_batch[i]
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
            except Exception as e:
                print(f"Error processing Qwen3-VL batch for {video_id}: {e}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        
        for shot in tqdm(shots, desc=f"OCR {video_id}", leave=False):
            start_f = int(shot.get("start_sec", 0.0) * fps)
            end_f = int(shot.get("end_sec", shot.get("start_sec", 0.0) + 3.0) * fps)
            
            # Sparse Multi-Frame Sampling (2 frames per shot)
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
                
                frame_batch.append(frame)
                shot_batch.append((f_idx, shot))
            
            if len(frame_batch) >= batch_size:
                process_batch(frame_batch, shot_batch)
                frame_batch = []
                shot_batch = []
                
        if len(frame_batch) > 0:
            process_batch(frame_batch, shot_batch)
            
        cap.release()

        # Generate Shot Rollup Documents
        for shot_id, data in shots_data.items():
            overlay_texts = []
            scene_texts = []
            
            for ocr_data in data["all"]:
                ot = ocr_data.get("overlay_text", "").strip()
                st = ocr_data.get("scene_text", "").strip()
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
        print(f"\n🚀 Running VLM OCR Pipeline (Qwen3-VL-7B-Instruct) on up to {limit_videos} videos...")
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

        # Open in append mode for true fail-safe streaming output
        with open(out_jsonl, "a", encoding="utf-8") as f:
            for idx, v_path in enumerate(video_files):
                print(f"[{idx+1}/{len(video_files)}] Processing OCR for {os.path.basename(v_path)}...")
                res = self.process_video(v_path)
                
                # Write immediately
                for doc in res["documents"]:
                    f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                f.flush()
                
                total_time += res["elapsed_sec"]
                total_shots += res["num_shots"]
                total_docs += res["num_documents"]

        benchmark_report = {
            "pipeline": "VLM OCR (Qwen3-VL-7B)",
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
    parser.add_argument("--limit", type=int, default=50, help="Maximum number of videos to process in benchmark")
    args = parser.parse_args()

    runner = OCRPipelineRunner(args.video_dir, args.output_dir, args.keyframes_dir)
    runner.run_benchmark(args.limit)

if __name__ == "__main__":
    main()
