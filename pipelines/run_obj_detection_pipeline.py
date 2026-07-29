import os
import sys
import time
import json
import argparse
import cv2
import numpy as np
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
from typing import List, Dict, Any
from extract.workers.keyframe_loader import KeyframeLoader

class ObjectDetectionPipelineRunner:
    """
    SOTA 5-Stage Video Object Detection Pipeline:
    Stage 1: Spatial UI Exclusion Masking (Channel Logo & Ticker Banner Masking)
    Stage 2: YOLO-World v2 Open-Vocab Detection (Ultralytics)
    Stage 3: Crop ROI & Spatial Location Categorization
    Stage 4: Shot-Level Object Summarization & Count Aggregation
    Stage 5: Database Document Schema & Elasticsearch Export
    """
    def __init__(self, video_dir: str, output_dir: str, keyframes_dir: str, device: str = "cuda"):
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.keyframes_dir = keyframes_dir
        self.device = device
        os.makedirs(output_dir, exist_ok=True)
        self.keyframe_loader = KeyframeLoader(keyframes_dir)
        self._init_yolo_model()

    def _init_yolo_model(self):
        """Initializes Ultralytics YOLO-World v2 / YOLOv8 SOTA Model."""
        self.yolo_model = None
        try:
            from ultralytics import YOLO
            # We use YOLO-World v2 and MUST set open-vocab classes
            model_weights = "yolov8x-worldv2.pt"
            try:
                print(f"Loading YOLO Object Detection model '{model_weights}'...")
                self.yolo_model = YOLO(model_weights)
                if self.device == "cuda":
                    self.yolo_model.to("cuda")
                
                # Set open-vocabulary prompt classes from BTC query catalogue
                self.prompt_classes = [
                    "person", "car", "motorcycle", "bicycle", "bus", "truck",
                    "ambulance", "police car", "fire truck",
                    "áo dài", "đàn bầu", "guitar", "microphone", "stage",
                    "flag", "banner", "crowd", "boat", "ship", "airplane",
                    "dog", "cat", "horse", "elephant", "bird",
                    "food", "fruit", "flower", "tree", "bridge", "building",
                    "traffic light", "sign", "helmet", "umbrella",
                    "television", "laptop", "phone", "camera",
                ]
                self.yolo_model.set_classes(self.prompt_classes)
                print(f"✅ YOLOE loaded and set_classes() with {len(self.prompt_classes)} prompts successfully.")
            except Exception as ex:
                raise RuntimeError(f"❌ FATAL: Could not load YOLOE {model_weights} or set_classes: {ex}")
        except Exception as e:
            raise RuntimeError(f"❌ FATAL: Ultralytics YOLO loading error: {e}")

    def stage1_spatial_ui_masking(self, frame: np.ndarray) -> np.ndarray:
        """Stage 1: Mask out Top-Right Channel Logo and Bottom Ticker Banner to avoid TV graphic false positives."""
        h, w, _ = frame.shape
        masked_frame = frame.copy()
        
        # Mask Top-Right logo zone
        top_right_x1, top_right_y1 = int(0.70 * w), 0
        top_right_x2, top_right_y2 = w, int(0.20 * h)
        cv2.rectangle(masked_frame, (top_right_x1, top_right_y1), (top_right_x2, top_right_y2), (0, 0, 0), -1)

        # Mask Bottom ticker banner zone
        bottom_x1, bottom_y1 = 0, int(0.85 * h)
        bottom_x2, bottom_y2 = w, h
        cv2.rectangle(masked_frame, (bottom_x1, bottom_y1), (bottom_x2, bottom_y2), (0, 0, 0), -1)

        return masked_frame

    def _determine_spatial_position(self, bbox: List[float], frame_w: int, frame_h: int) -> str:
        """Categorizes bounding box into spatial quad zones."""
        x_center = (bbox[0] + bbox[2]) / 2.0 / frame_w
        y_center = (bbox[1] + bbox[3]) / 2.0 / frame_h

        horiz = "left" if x_center < 0.33 else ("right" if x_center > 0.66 else "center")
        vert = "top" if y_center < 0.33 else ("bottom" if y_center > 0.66 else "center")
        return f"{vert}_{horiz}" if horiz != "center" or vert != "center" else "center"

    def process_video_objects(self, video_path: str) -> Dict[str, Any]:
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        start_time = time.time()

        shots = self.keyframe_loader.load(video_id)
        if not shots:
            return {"video_id": video_id, "elapsed_sec": 0.0, "documents": []}

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"video_id": video_id, "elapsed_sec": 0.0, "documents": []}

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)

        per_frame_documents = []

        for shot in shots:
            f_idx = shot["keyframe_id"]
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, raw_frame = cap.read()
            if not ret:
                continue

            masked_frame = self.stage1_spatial_ui_masking(raw_frame)
            frame_detections = []

            if self.yolo_model is not None:
                try:
                    results = self.yolo_model(masked_frame, verbose=False, device="cuda" if self.device == "cuda" else "cpu")
                    if results and len(results) > 0:
                        boxes = results[0].boxes
                        for box in boxes:
                            conf = float(box.conf[0])
                            if conf >= 0.35:
                                cls_id = int(box.cls[0])
                                class_name = self.yolo_model.names[cls_id]
                                xyxy = box.xyxy[0].tolist()
                                bbox_norm = [
                                    round(xyxy[0] / frame_width, 3),
                                    round(xyxy[1] / frame_height, 3),
                                    round(xyxy[2] / frame_width, 3),
                                    round(xyxy[3] / frame_height, 3)
                                ]
                                spatial_pos = self._determine_spatial_position(xyxy, frame_width, frame_height)

                                frame_detections.append({
                                    "label": class_name,
                                    "confidence": round(conf, 3),
                                    "bbox": bbox_norm,
                                    "spatial_position": spatial_pos
                                })
                except Exception as ex:
                    print(f"Error running YOLO on frame {f_idx} for {video_id}: {ex}")

            # Emit per-frame document
            doc = {
                "video_id": video_id,
                "frame_idx": f_idx,
                "keyframe_n": shot.get("keyframe_n", 0),
                "time_range": {"start_sec": shot.get("start_sec", 0.0), "end_sec": shot.get("end_sec", 0.0)},
                "detections": frame_detections
            }
            per_frame_documents.append(doc)

        cap.release()
        
        # Phase 7: Generate Per-Shot Rollup Documents
        shot_documents = []
        from collections import defaultdict
        
        # Group frames by shot ID (keyframe_n)
        shot_groups = defaultdict(list)
        for doc in per_frame_documents:
            shot_groups[doc["keyframe_n"]].append(doc)
            
        for shot_id, frames in shot_groups.items():
            shot_time_range = {
                "start_sec": min(f["time_range"]["start_sec"] for f in frames),
                "end_sec": max(f["time_range"]["end_sec"] for f in frames)
            }
            
            max_counts = defaultdict(int)
            spatial_tokens = set()
            detected_classes = set()
            
            for f in frames:
                frame_counts = defaultdict(int)
                for det in f.get("detections", []):
                    label = det["label"]
                    frame_counts[label] += 1
                    detected_classes.add(label)
                    spatial_tokens.add(f"{label} {det['spatial_position']}")
                
                # Update max counts for the shot
                for label, count in frame_counts.items():
                    if count > max_counts[label]:
                        max_counts[label] = count
                        
            shot_doc = {
                "doc_type": "shot",
                "video_id": video_id,
                "shot_id": shot_id,
                "time_range": shot_time_range,
                "max_counts": dict(max_counts),
                "detected_classes": list(detected_classes),
                "spatial_tokens": list(spatial_tokens)
            }
            shot_documents.append(shot_doc)
            
        # Add doc_type to frame documents as well
        for doc in per_frame_documents:
            doc["doc_type"] = "frame"

        elapsed_sec = round(time.time() - start_time, 2)
        return {
            "video_id": video_id,
            "elapsed_sec": elapsed_sec,
            "documents": per_frame_documents + shot_documents
        }

    def run_benchmark(self, limit_videos: int = 50):
        print(f"\n🚀 Running SOTA Object Detection Pipeline (YOLO-World / YOLOv8) on up to {limit_videos} videos...")
        video_files = []
        for root, _, files in os.walk(self.video_dir):
            for f in files:
                if f.lower().endswith(('.mp4', '.mkv', '.avi', '.webm')):
                    video_files.append(os.path.join(root, f))
        video_files = video_files[:limit_videos]

        if not video_files:
            print("⚠️ No video files found in directory.")
            return

        out_jsonl = os.path.join(self.output_dir, "obj_extracted_documents.jsonl")
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
                print(f"⏭️ Skipping {v_id} - already processed (resume).")
                
        video_files = pending_videos

        if not video_files:
            print("⚠️ No pending video files found to process (all done).")
            return

        total_time = 0.0
        total_docs = 0
        total_objects = 0

        # Open in append mode for true fail-safe streaming output
        with open(out_jsonl, "a", encoding="utf-8") as f:
            for idx, v_path in enumerate(video_files):
                print(f"[{idx+1}/{len(video_files)}] Processing Object Detection for {os.path.basename(v_path)}...")
                res = self.process_video_objects(v_path)
                
                # Write immediately
                for doc in res["documents"]:
                    f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                f.flush()
                
                total_time += res["elapsed_sec"]
                total_docs += len(res["documents"])
                total_objects += sum(len(d.get("detections", [])) for d in res["documents"])

        benchmark_report = {
            "pipeline": "SOTA 5-Stage Object Detection (YOLO-World v2 / YOLOv8)",
            "videos_processed": len(video_files),
            "total_elapsed_sec": round(total_time, 2),
            "avg_time_per_video_sec": round(total_time / len(video_files), 3) if video_files else 0,
            "total_object_documents": total_docs,
            "total_objects_detected": total_objects,
            "output_jsonl_path": out_jsonl
        }

        report_path = os.path.join(self.output_dir, "obj_benchmark_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(benchmark_report, f, indent=2, ensure_ascii=False)

        print("\n📊 --- OBJECT DETECTION BENCHMARK REPORT ---")
        print(json.dumps(benchmark_report, indent=2, ensure_ascii=False))
        print(f"✅ Saved Object Detection extracted documents to: {out_jsonl}")

def main():
    parser = argparse.ArgumentParser(description="Run 5-Stage SOTA Object Detection Pipeline")
    parser.add_argument("--video-dir", type=str, default="./data/extracted", help="Directory containing raw videos")
    parser.add_argument("--output-dir", type=str, default="./data/processed/objects", help="Output directory for Object records")
    parser.add_argument("--keyframes-dir", type=str, default="./data/extracted/video batch 1/map-keyframes-aic25-b1/map-keyframes", help="Directory containing BTC keyframe CSVs")
    parser.add_argument("--limit", type=int, default=50, help="Maximum number of videos to process in benchmark")
    args = parser.parse_args()

    runner = ObjectDetectionPipelineRunner(args.video_dir, args.output_dir, args.keyframes_dir)
    runner.run_benchmark(args.limit)

if __name__ == "__main__":
    main()
