import os
import sys
import argparse
import gc

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def clear_vram_cache():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

from run_ocr_pipeline import OCRPipelineRunner
from run_asr_pipeline import ASRPipelineRunner
from run_obj_detection_pipeline import ObjectDetectionPipelineRunner

def main():
    parser = argparse.ArgumentParser(description="Master Execution Runner for Multimodal Video Retrieval Pipelines")
    parser.add_argument("--video-dir", type=str, default="./data/extracted", help="Directory containing raw extracted videos")
    parser.add_argument("--output-base-dir", type=str, default="./data/processed", help="Base output directory")
    parser.add_argument("--keyframes-dir", type=str, default="./data/extracted/video batch 1/map-keyframes-aic25-b1/map-keyframes", help="Directory containing BTC keyframe CSVs")
    parser.add_argument("--media-info-dir", type=str, default="./data/extracted/video batch 1/media-info-aic25-b1/media-info", help="Directory containing BTC media info JSONs")
    parser.add_argument("--limit", type=int, default=50, help="Number of videos to run benchmark on")
    args = parser.parse_args()

    print("==========================================================================")
    print("🚀 STARTING SEQUENTIAL MULTIMODAL EXTRACTION (LIGHTWEIGHT MEMORY MANAGEMENT)")
    print("==========================================================================")

    # 1. Run OCR Pipeline
    print("\n--- 1/3 Running OCR Pipeline (PaddleOCR) ---")
    ocr_out = os.path.join(args.output_base_dir, "ocr")
    ocr_runner = OCRPipelineRunner(args.video_dir, ocr_out, args.keyframes_dir)
    ocr_runner.run_benchmark(args.limit)
    del ocr_runner
    clear_vram_cache()

    # 2. Run ASR Pipeline
    print("\n--- 2/3 Running ASR Pipeline (PhoWhisper) ---")
    asr_out = os.path.join(args.output_base_dir, "asr")
    asr_runner = ASRPipelineRunner(args.video_dir, asr_out, args.keyframes_dir, args.media_info_dir)
    asr_runner.run_benchmark(args.limit)
    del asr_runner
    clear_vram_cache()

    # 3. Run Object Detection Pipeline
    print("\n--- 3/3 Running Object Detection Pipeline (YOLO-World) ---")
    obj_out = os.path.join(args.output_base_dir, "objects")
    obj_runner = ObjectDetectionPipelineRunner(args.video_dir, obj_out, args.keyframes_dir)
    obj_runner.run_benchmark(args.limit)
    del obj_runner
    clear_vram_cache()

    print("\n==========================================================================")
    print("🎉 ALL 3 PIPELINES COMPLETED SUCCESSFULLY!")
    print(f"📁 Extracted database records saved under: {args.output_base_dir}")
    print("==========================================================================")

if __name__ == "__main__":
    main()
