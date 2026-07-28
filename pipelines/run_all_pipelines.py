import os
import sys
import argparse
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from run_ocr_pipeline import OCRPipelineRunner
from run_asr_pipeline import ASRPipelineRunner
from run_obj_detection_pipeline import ObjectDetectionPipelineRunner

def main():
    parser = argparse.ArgumentParser(description="Master Execution Runner for Multimodal Video Retrieval Pipelines")
    parser.add_argument("--video-dir", type=str, default="./data/extracted", help="Directory containing raw extracted videos")
    parser.add_argument("--output-base-dir", type=str, default="./data/processed", help="Base output directory")
    parser.add_argument("--limit", type=int, default=50, help="Number of videos to run benchmark on")
    args = parser.parse_args()

    print("==========================================================================")
    print("🚀 STARTING MULTIMODAL EXTRACTION PIPELINES (OCR, ASR, OBJECT DETECTION)")
    print("==========================================================================")

    # 1. Run OCR Pipeline
    print("\n--- 1/3 Running OCR Pipeline ---")
    ocr_out = os.path.join(args.output_base_dir, "ocr")
    ocr_runner = OCRPipelineRunner(args.video_dir, ocr_out)
    ocr_runner.run_benchmark(args.limit)

    # 2. Run ASR Pipeline
    print("\n--- 2/3 Running ASR Pipeline ---")
    asr_out = os.path.join(args.output_base_dir, "asr")
    asr_runner = ASRPipelineRunner(args.video_dir, asr_out)
    asr_runner.run_benchmark(args.limit)

    # 3. Run Object Detection Pipeline
    print("\n--- 3/3 Running Object Detection Pipeline ---")
    obj_out = os.path.join(args.output_base_dir, "objects")
    obj_runner = ObjectDetectionPipelineRunner(args.video_dir, obj_out)
    obj_runner.run_benchmark(args.limit)

    print("\n==========================================================================")
    print("🎉 ALL 3 PIPELINES COMPLETED SUCCESSFULLY!")
    print(f"📁 Extracted database records saved under: {args.output_base_dir}")
    print("==========================================================================")

if __name__ == "__main__":
    main()
