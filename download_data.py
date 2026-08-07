import os
import argparse
from huggingface_hub import hf_hub_download

REPO_ID = "htkien95/DATA-AIC"
DEFAULT_LOCAL_DIR = "./data/raw"

# Phase 0 files: Queries and metadata
PHASE0_FILES = [
    "DATA AIC 2026/query/DanhSachTruyVanAIC_Chungket.xlsx",
    "DATA AIC 2026/query/query-p1-groupA.zip",
    "DATA AIC 2026/query/query_all.zip",
] + [f"DATA AIC 2026/map-keyframes/map-keyframes_L{i}.zip" for i in range(21, 31)] \
  + [f"DATA AIC 2026/media-info/media-info_L{i}.zip" for i in range(21, 31)]

# Phase 1 files: Pre-computed CLIP features and BTC Object Detections
PHASE1_FILES = [f"DATA AIC 2026/clip-features-32/clip-features-32_L{i}.zip" for i in range(21, 31)] \
             + [f"DATA AIC 2026/objects/objects_L{i}.zip" for i in range(21, 31)]

# Benchmark subset files: Videos (Only L21 and L22 as requested)
BENCHMARK_VIDEO_FILES = [
    "DATA AIC 2026/video batch 1/Videos_L21_a.zip",
    "DATA AIC 2026/video batch 1/Videos_L22_a.zip"
]

def download_file_list(files, phase_name, local_dir, token=None):
    print(f"\n--- Starting Download for {phase_name} ---")
    os.makedirs(local_dir, exist_ok=True)
    for filepath in files:
        print(f"Downloading {filepath}...")
        try:
            downloaded_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=filepath,
                repo_type="dataset",
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                token=token
            )
            print(f"✅ SUCCESS: Saved to: {downloaded_path}")
        except Exception as e:
            print(f"❌ ERROR: Failed to download {filepath}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download AI Challenge HCMC Datasets & Benchmark Packages")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_LOCAL_DIR, help="Local directory to store raw downloads")
    parser.add_argument("--phase", type=str, default="benchmark", choices=["all", "phase0", "phase1", "benchmark"],
                        help="Which data phase to download: phase0 (Queries/Meta), phase1 (CLIP/Objects), benchmark, all")
    parser.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN"), help="HuggingFace token for private datasets")
    args = parser.parse_args()

    if not args.hf_token:
        print("⚠️ Warning: No HF token provided. Private dataset downloads will likely fail.")
        print("Please provide it via --hf-token or set the HF_TOKEN environment variable.")

    if args.phase in ["phase0", "all"]:
        download_file_list(PHASE0_FILES, "Phase 0 (Queries & Metadata)", args.output_dir, args.hf_token)
    if args.phase in ["phase1", "all"]:
        download_file_list(PHASE1_FILES, "Phase 1 (CLIP Features & Object Detections)", args.output_dir, args.hf_token)
    if args.phase in ["benchmark", "all"]:
        download_file_list(BENCHMARK_VIDEO_FILES, "Benchmark Video Package", args.output_dir, args.hf_token)

    print("\n🎉 Download task completed!")

if __name__ == "__main__":
    main()
