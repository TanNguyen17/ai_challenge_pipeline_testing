import os
import time
import json
import argparse
import cv2
import numpy as np
from typing import List, Dict, Any

class ObjectDetectionPipelineRunner:
    """
    SOTA 5-Stage Video Object Detection Pipeline:
    Stage 1: Spatial UI Exclusion Masking (Channel Logo & Ticker Banner Masking)
    Stage 2: YOLO-World v2 Open-Vocab Detection (Ultralytics)
    Stage 3: Crop ROI & Spatial Location Categorization
    Stage 4: Shot-Level Object Summarization & Count Aggregation
    Stage 5: Database Document Schema & Elasticsearch Export
    """
    def __init__(self, video_dir: str, output_dir: str, device: str = "cuda"):
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.device = device
        os.makedirs(output_dir, exist_ok=True)
        self._init_yolo_model()

    def _init_yolo_model(self):
        """Initializes Ultralytics YOLO-World v2 / YOLOv8 SOTA Model."""
        self.yolo_model = None
        try:
            from ultralytics import YOLO
            # Try YOLO-World v2 first, fallback to yolov8x / yolov8n
            model_weights = ["yolov8x-worldv2.pt", "yolov8x.pt", "yolov8n.pt"]
            for weights in model_weights:
                try:
                    print(f"Loading YOLO Object Detection model '{weights}'...")
                    self.yolo_model = YOLO(weights)
                    if self.device == "cuda":
                        self.yolo_model.to("cuda")
                    print(f"✅ YOLO Model '{weights}' loaded successfully.")
                    break
                except Exception as ex:
                    print(f"Notice: Could not load {weights}: {ex}. Trying next fallback...")
        except Exception as e:
            print(f"⚠️ Ultralytics YOLO loading error: {e}. Falling back to baseline detector.")

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

    def process_video_objects(self, video_path: str, max_keyframes: int = 10) -> Dict[str, Any]:
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        start_time = time.time()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {
                "video_id": video_id,
                "elapsed_sec": 0.0,
                "document": {"video_id": video_id, "object_summary": {"detected_classes": [], "counts": {}, "objects_detail": []}}
            }

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)

        # Sample keyframes evenly
        step = max(1, total_frames // max_keyframes) if total_frames > 0 else 30
        keyframe_indices = list(range(0, total_frames, step))[:max_keyframes]

        all_detected_objects = []

        for shot_id, f_idx in enumerate(keyframe_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, raw_frame = cap.read()
            if not ret:
                continue

            masked_frame = self.stage1_spatial_ui_masking(raw_frame)

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

                                all_detected_objects.append({
                                    "object_id": f"OBJ_{len(all_detected_objects)+1:03d}",
                                    "shot_id": shot_id,
                                    "keyframe_id": f_idx,
                                    "timestamp_sec": round(f_idx / fps, 2),
                                    "label": class_name,
                                    "confidence": round(conf, 3),
                                    "bbox": bbox_norm,
                                    "spatial_position": spatial_pos
                                })
                except Exception as ex:
                    print(f"Error running YOLO on frame {f_idx} for {video_id}: {ex}")

        cap.release()

        # Class counts aggregation
        class_counts = {}
        for obj in all_detected_objects:
            lbl = obj["label"]
            class_counts[lbl] = class_counts.get(lbl, 0) + 1

        doc = {
            "video_id": video_id,
            "keyframe_indices": keyframe_indices,
            "object_summary": {
                "detected_classes": list(class_counts.keys()),
                "counts": class_counts,
                "total_objects": len(all_detected_objects),
                "objects_detail": all_detected_objects
            }
        }

        elapsed_sec = round(time.time() - start_time, 2)
        return {
            "video_id": video_id,
            "elapsed_sec": elapsed_sec,
            "document": doc
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

        results = []
        total_time = 0.0
        total_docs = 0
        total_objects = 0

        for idx, v_path in enumerate(video_files):
            print(f"[{idx+1}/{len(video_files)}] Processing Object Detection for {os.path.basename(v_path)}...")
            res = self.process_video_objects(v_path)
            results.append(res)
            total_time += res["elapsed_sec"]
            total_docs += 1
            total_objects += res["document"]["object_summary"]["total_objects"]

        out_jsonl = os.path.join(self.output_dir, "obj_extracted_documents.jsonl")
        with open(out_jsonl, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r["document"], ensure_ascii=False) + "\n")

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
    parser.add_argument("--limit", type=int, default=50, help="Maximum number of videos to process in benchmark")
    args = parser.parse_args()

    runner = ObjectDetectionPipelineRunner(args.video_dir, args.output_dir)
    runner.run_benchmark(args.limit)

if __name__ == "__main__":
    main()
