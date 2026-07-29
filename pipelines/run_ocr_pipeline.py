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

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from typing import List, Dict, Any
from extract.workers.keyframe_loader import KeyframeLoader

class OCRPipelineRunner:
    """
    SOTA Video OCR Pipeline using Qwen2-VL-2B-Instruct:
    Stage 1: Keyframe Sampling from BTC CSVs
    Stage 2: Batched VLM Text Extraction (Qwen2-VL)
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
            print("Loading Qwen2-VL-2B-Instruct...")
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            qwen_id = "Qwen/Qwen2-VL-2B-Instruct"
            self.processor = AutoProcessor.from_pretrained(qwen_id)
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                qwen_id, 
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                device_map="cuda" if self.device == "cuda" else "cpu"
            )
            self.model.eval()
            print("✅ Qwen2-VL-2B-Instruct loaded on GPU for OCR.")
        except Exception as e:
            print(f"❌ Failed to load Qwen2-VL: {e}")
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
                
                messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Trích xuất toàn bộ văn bản tiếng Việt xuất hiện trong ảnh này. Chỉ xuất ra văn bản tiếng Việt, bỏ qua các chi tiết không phải là chữ. Không thêm bất kỳ lời giải thích nào."}]}]
                prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                inputs = self.processor(
                    text=[prompt] * len(pil_images),
                    images=pil_images,
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
                    f_idx, shot = s_batch[i]
                    shot_id = shot["shot_id"]
                    
                    if not shots_data[shot_id]["time_range"]:
                        shots_data[shot_id]["time_range"] = {"start_sec": shot.get("start_sec", 0.0), "end_sec": shot.get("end_sec", 0.0)}
                    
                    if ocr_text:
                        shots_data[shot_id]["all"].append(ocr_text)
                        
                        doc = {
                            "doc_type": "span",
                            "video_id": video_id,
                            "shot_id": shot_id,
                            "tracklet_id": f"TRK_{f_idx:05d}",
                            "frame_idx": f_idx,
                            "keyframe_n": shot.get("keyframe_n", 0),
                            "time_range": {"start_sec": shot.get("start_sec", 0.0), "end_sec": shot.get("end_sec", 0.0)},
                            "ocr_raw_full": ocr_text,
                            "ocr_no_accent": self._remove_accents(ocr_text),
                            "ocr_system": "qwen2-vl-2b",
                            "confidence": 1.0
                        }
                        final_documents.append(doc)
            except Exception as e:
                print(f"Error processing Qwen2-VL batch for {video_id}: {e}")

        for shot in tqdm(shots, desc=f"OCR {video_id}", leave=False):
            f_idx = shot["keyframe_id"]
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
            # Filter out empty texts
            valid_texts = [t for t in data["all"] if t.strip()]
            if not valid_texts:
                continue
                
            shot_doc = {
                "doc_type": "shot",
                "video_id": video_id,
                "shot_id": shot_id,
                "time_range": data["time_range"],
                "ocr_full_combined": " | ".join(valid_texts)
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
        print(f"\n🚀 Running VLM OCR Pipeline (Qwen2-VL-2B-Instruct) on up to {limit_videos} videos...")
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
            "pipeline": "VLM OCR (Qwen2-VL-2B)",
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
