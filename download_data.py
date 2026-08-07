import os
import argparse
from huggingface_hub import hf_hub_download

REPO_ID = "htkien95/DATA-AIC-2025"
DEFAULT_LOCAL_DIR = "./data/raw"

# Phase 0 files: Queries and metadata (~5 MB)
# PHASE0_FILES = [
#     "query/DanhSachTruyVanAIC_Chungket.xlsx",
#     "query/query-p1-groupA.zip",
#     "query/query-p2-groupA.zip",
#     "query/query-p3-groupA.zip",
#     "video batch 1/map-keyframes-aic25-b1.zip",
#     "video batch 1/media-info-aic25-b1.zip",
#     "video batch 2/map-keyframes-b2.zip",
#     "video batch 2/media-info-aic25-b2.zip"
# ]

# Phase 1 files: Pre-computed CLIP features and BTC Object Detections (~1.6 GB)
# PHASE1_FILES = [
#     "video batch 1/clip-features-32-aic25-b1.zip",
#     "video batch 1/objects-aic25-b1.zip",
#     "video batch 2/clip-features-32-aic25-b2.zip",
#     "video batch 2/objects-aic25-b2.zip"
# ]

# Benchmark subset files: Videos L21_a and L22_a (~650 videos, ~7.4 GB)
BENCHMARK_VIDEO_FILES = [
    "DATA AIC 2026/video batch 1/Videos_L21_a.zip",
    "DATA AIC 2026/video batch 1/Videos_L22_a.zip"
]

def download_file_list(files, phase_name, local_dir):
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
                local_dir_use_symlinks=False
            )
            print(f"✅ SUCCESS: Saved to: {downloaded_path}")
        except Exception as e:
            print(f"❌ ERROR: Failed to download {filepath}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download AI Challenge HCMC Datasets & Benchmark Packages")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_LOCAL_DIR, help="Local directory to store raw downloads")
    parser.add_argument("--phase", type=str, default="benchmark", choices=["all", "phase0", "phase1", "benchmark"],
                        help="Which data phase to download: phase0 (Queries/Meta), phase1 (CLIP/Objects), benchmark (650 videos), all")
    args = parser.parse_args()

    # if args.phase in ["phase0", "all"]:
    #     download_file_list(PHASE0_FILES, "Phase 0 (Queries & Metadata)", args.output_dir)
    # if args.phase in ["phase1", "all"]:
    #     download_file_list(PHASE1_FILES, "Phase 1 (CLIP Features & Object Detections)", args.output_dir)
    if args.phase in ["benchmark", "all"]:
        # download_file_list(PHASE0_FILES, "Phase 0 (Queries & Metadata)", args.output_dir)
        download_file_list(BENCHMARK_VIDEO_FILES, "Benchmark Video Package (~650 Videos)", args.output_dir)

    print("\n🎉 Download task completed!")

if __name__ == "__main__":
    main()
