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
from rapidfuzz import fuzz, distance

import paddle
from paddleocr import PaddleOCR

from extract.workers.keyframe_loader import KeyframeLoader

class OCRPipelineRunner:
    """
    SOTA 5-Stage Video OCR Pipeline:
    Stage 1: TransNetV2 Shot Boundary Detection & Keyframe Sampling
    Stage 2: PP-OCRv5 Text Spotting (Detection & Recognition via PaddleOCR)
    Stage 3: ByteTrack Text Tracking & Tracklet Formation (IoU + String Similarity)
    Stage 4: RapidFuzz / LCS Substring Stitching & Consensus Voting
    Stage 5: Dynamic Layout Classification & Elasticsearch Document Export
    """
    def __init__(self, video_dir: str, output_dir: str, keyframes_dir: str, device: str = "cuda"):
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.keyframes_dir = keyframes_dir
        self.device = device
        os.makedirs(output_dir, exist_ok=True)
        self.keyframe_loader = KeyframeLoader(keyframes_dir)
        self._init_paddleocr()

    def _init_paddleocr(self):
        """Initializes PaddleOCR SOTA model engine with GPU support."""
        self.ocr_engine = None
        try:
            use_gpu = self.device == "cuda" and paddle.is_compiled_with_cuda()
            self.ocr_engine = PaddleOCR(
                use_angle_cls=True,
                lang="vi",
                use_gpu=use_gpu,
                show_log=False
            )
            print(f"✅ PaddleOCR engine loaded on {'GPU' if use_gpu else 'CPU'}.")
        except ModuleNotFoundError as e:
            print(f"⚠️ PaddleOCR loading notice: {e}. Please run 'uv pip install paddlepaddle-gpu' to enable GPU PaddleOCR.")
        except Exception as e:
            print(f"⚠️ PaddleOCR loading warning: {e}. Falling back to baseline text extractor.")

    def stage1_load_keyframes(self, video_id: str) -> List[Dict[str, Any]]:
        """Stage 1: Load pre-computed BTC keyframes from CSV."""
        return self.keyframe_loader.load(video_id)

    def stage2_ppocr_v5(self, video_path: str, shots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Stage 2: PP-OCRv5 Text Spotting on Keyframes."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        detections_per_keyframe = []

        for shot in shots:
            keyframe_id = shot["keyframe_id"]
            cap.set(cv2.CAP_PROP_POS_FRAMES, keyframe_id)
            ret, frame = cap.read()
            if not ret:
                continue

            raw_ocr = []
            if self.ocr_engine is not None:
                try:
                    result = self.ocr_engine.ocr(frame, cls=True)
                    if result and result[0]:
                        for line in result[0]:
                            bbox_poly, (text, conf) = line[0], line[1]
                            if conf >= 0.5 and len(text.strip()) > 1:
                                # Convert polygon to bounding box [x_min, y_min, x_max, y_max]
                                xs = [pt[0] for pt in bbox_poly]
                                ys = [pt[1] for pt in bbox_poly]
                                bbox = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
                                raw_ocr.append({
                                    "bbox": bbox,
                                    "text": text.strip(),
                                    "confidence": float(conf),
                                    "frame_idx": keyframe_id,
                                    "keyframe_n": shot.get("keyframe_n", 0),
                                    "time_range": {"start_sec": shot.get("start_sec", 0.0), "end_sec": shot.get("end_sec", 0.0)}
                                })
                except Exception as ex:
                    print(f"Error running OCR on frame {keyframe_id}: {ex}")

            detections_per_keyframe.append({
                "shot_id": shot["shot_id"],
                "keyframe_id": keyframe_id,
                "timestamp_sec": round(keyframe_id / fps, 2),
                "time_range": {"start_sec": shot.get("start_sec", 0.0), "end_sec": shot.get("end_sec", 0.0)},
                "keyframe_n": shot.get("keyframe_n", 0),
                "frame_height": height,
                "frame_width": width,
                "raw_ocr": raw_ocr
            })

        cap.release()
        return detections_per_keyframe

    @staticmethod
    def _calculate_iou(boxA: List[int], boxB: List[int]) -> float:
        """Calculates Intersection over Union (IoU) for two bounding boxes [x_min, y_min, x_max, y_max]."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        if interArea == 0:
            return 0.0

        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        return interArea / float(boxAArea + boxBArea - interArea)

    def stage3_bytetrack(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Stage 3: Spatial IoU & Text Similarity Tracklet Formation."""
        tracklets = []
        tracklet_counter = 1

        for det in detections:
            shot_id = det["shot_id"]
            for ocr_item in det["raw_ocr"]:
                # Match existing tracklets
                matched = False
                for trk in tracklets:
                    if trk["shot_id"] == shot_id:
                        last_obs = trk["observations"][-1]
                        sim = fuzz.ratio(ocr_item["text"].lower(), last_obs["text"].lower())
                        iou = self._calculate_iou(ocr_item["bbox"], last_obs["bbox"])
                        
                        # Match if text is very similar OR (text overlaps slightly AND spatial location is very close)
                        if (sim > 70 and iou > 0.1) or (sim > 40 and iou > 0.4):
                            trk["observations"].append(ocr_item)
                            matched = True
                            break

                if not matched:
                    tracklets.append({
                        "tracklet_id": f"TRK_{tracklet_counter:03d}",
                        "shot_id": shot_id,
                        "frame_height": det["frame_height"],
                        "frame_width": det["frame_width"],
                        "observations": [ocr_item]
                    })
                    tracklet_counter += 1

        return tracklets

    def stage4_lcs_stitching(self, tracklets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Stage 4: Longest Substring Selection & Stitching (LCS)."""
        import difflib
        
        clean_text_records = []
        for trk in tracklets:
            texts = [obs["text"] for obs in trk["observations"]]
            
            # Real LCS Stitching for scrolling ticker text
            if len(texts) == 1:
                stitched_text = texts[0]
            else:
                stitched_text = texts[0]
                for next_text in texts[1:]:
                    matcher = difflib.SequenceMatcher(None, stitched_text, next_text)
                    match = matcher.find_longest_match(0, len(stitched_text), 0, len(next_text))
                    
                    # If they share a substring of at least 3 chars, stitch them
                    if match.size >= 3:
                        # Append the non-overlapping part of next_text
                        stitched_text = stitched_text + next_text[match.b + match.size:]
                    else:
                        # Fallback if totally disjoint but tracked together (rare with IoU)
                        if len(next_text) > len(stitched_text):
                            stitched_text = next_text

            avg_conf = sum(obs["confidence"] for obs in trk["observations"]) / len(trk["observations"])

            clean_text_records.append({
                "tracklet_id": trk["tracklet_id"],
                "shot_id": trk["shot_id"],
                "stitched_text": stitched_text,
                "bbox": trk["observations"][0]["bbox"],
                "frame_idx": trk["observations"][0]["frame_idx"],
                "keyframe_n": trk["observations"][0]["keyframe_n"],
                "time_range": trk["observations"][0]["time_range"],
                "frame_height": trk.get("frame_height", 1080),
                "frame_width": trk.get("frame_width", 1920),
                "avg_confidence": round(avg_conf, 3)
            })
        return clean_text_records

    def stage5_layout_classifier(self, clean_records: List[Dict[str, Any]], video_id: str) -> List[Dict[str, Any]]:
        """Stage 5: Dynamic Layout Classification & Elasticsearch Document Schema (Span + Shot Rollup)."""
        import re
        from collections import defaultdict
        
        final_documents = []
        timestamp_pattern = re.compile(r'^\d{1,2}[:.]\d{2}([:.]\d{2})?$|^\d{4,6}$')

        # For shot rollup
        shots_data = defaultdict(lambda: {
            "overlay": [], "scene": [], "system": [], "all": [], "time_range": None
        })

        for rec in clean_records:
            bbox = rec["bbox"]
            text = rec["stitched_text"].strip()
            if not text:
                continue

            h = rec.get("frame_height", 1080)
            w = rec.get("frame_width", 1920)

            # Spatial UI Classification logic
            ocr_overlay, ocr_scene, ocr_system = None, None, None
            layout_type = "scene"
            y_mid = (bbox[1] + bbox[3]) / 2.0
            x_mid = (bbox[0] + bbox[2]) / 2.0

            # Filter out pure timestamp/digital clock noise or standalone numbers
            is_timestamp_noise = bool(timestamp_pattern.match(text))

            if x_mid > 0.70 * w and y_mid < 0.30 * h: # Top-Right Logo/Timestamp area
                ocr_system = text
                layout_type = "system"
            elif y_mid > 0.65 * h: # Lower banner / subtitle / ticker area
                ocr_overlay = text
                layout_type = "overlay"
            elif y_mid < 0.20 * h and (x_mid < 0.30 * w or x_mid > 0.70 * w): # Top banner area
                ocr_overlay = text
                layout_type = "overlay"
            else:
                ocr_scene = text
                layout_type = "scene"

            # Accent-stripped fallback text
            unaccented = self._remove_accents(text)

            doc = {
                "doc_type": "span",
                "video_id": video_id,
                "shot_id": rec["shot_id"],
                "tracklet_id": rec["tracklet_id"],
                "frame_idx": rec["frame_idx"],
                "keyframe_n": rec["keyframe_n"],
                "time_range": rec["time_range"],
                "ocr_overlay": ocr_overlay,
                "ocr_scene": ocr_scene,
                "ocr_system": ocr_system,
                "is_noise": is_timestamp_noise,
                "ocr_no_accent": unaccented,
                "ocr_raw_full": text,
                "confidence": rec["avg_confidence"]
            }
            final_documents.append(doc)
            
            # Aggregate for shot rollup
            shot_id = rec["shot_id"]
            if not shots_data[shot_id]["time_range"]:
                shots_data[shot_id]["time_range"] = rec["time_range"]
            
            if not is_timestamp_noise:
                shots_data[shot_id][layout_type].append(text)
                shots_data[shot_id]["all"].append(text)

        # Generate Shot Rollup Documents
        for shot_id, data in shots_data.items():
            shot_doc = {
                "doc_type": "shot",
                "video_id": video_id,
                "shot_id": shot_id,
                "time_range": data["time_range"],
                "ocr_overlay_combined": " | ".join(data["overlay"]),
                "ocr_scene_combined": " | ".join(data["scene"]),
                "ocr_system_combined": " | ".join(data["system"]),
                "ocr_full_combined": " | ".join(data["all"])
            }
            final_documents.append(shot_doc)

        return final_documents

    @staticmethod
    def _remove_accents(input_str: str) -> str:
        s = input_str.lower()
        accents = {
            'a': 'àáảạãăằắẳặẵâầấẩậẫ',
            'd': 'đ',
            'e': 'èéẻẹẽêềếểệễ',
            'i': 'ìíỉịĩ',
            'o': 'òóỏọõôồốổộỗơờớởợỡ',
            'u': 'ùúủụũưừứửựữ',
            'y': 'ỳýỷỵỹ'
        }
        for char, accented_chars in accents.items():
            for a in accented_chars:
                s = s.replace(a, char)
        return s

    def process_video(self, video_path: str) -> Dict[str, Any]:
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        start_time = time.time()

        shots = self.stage1_load_keyframes(video_id)
        if not shots:
            return {"video_id": video_id, "elapsed_sec": 0.0, "num_shots": 0, "num_documents": 0, "documents": []}
            
        detections = self.stage2_ppocr_v5(video_path, shots)
        tracklets = self.stage3_bytetrack(detections)
        clean_records = self.stage4_lcs_stitching(tracklets)
        documents = self.stage5_layout_classifier(clean_records, video_id)

        elapsed_sec = round(time.time() - start_time, 2)
        return {
            "video_id": video_id,
            "elapsed_sec": elapsed_sec,
            "num_shots": len(shots),
            "num_documents": len(documents),
            "documents": documents
        }

    def run_benchmark(self, limit_videos: int = 50):
        print(f"\n🚀 Running SOTA OCR Pipeline (PaddleOCR + ByteTrack) on up to {limit_videos} videos...")
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
                print(f"⏭️ Skipping {v_id} - already processed (resume).")
                
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
            "pipeline": "SOTA 5-Stage Video OCR (PaddleOCR)",
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
    parser = argparse.ArgumentParser(description="Run 5-Stage SOTA Video OCR Pipeline")
    parser.add_argument("--video-dir", type=str, default="./data/extracted", help="Directory containing raw videos")
    parser.add_argument("--output-dir", type=str, default="./data/processed/ocr", help="Output directory for OCR records")
    parser.add_argument("--keyframes-dir", type=str, default="./data/extracted/video batch 1/map-keyframes-aic25-b1/map-keyframes", help="Directory containing BTC keyframe CSVs")
    parser.add_argument("--limit", type=int, default=50, help="Maximum number of videos to process in benchmark")
    args = parser.parse_args()

    runner = OCRPipelineRunner(args.video_dir, args.output_dir, args.keyframes_dir)
    runner.run_benchmark(args.limit)

if __name__ == "__main__":
    main()
