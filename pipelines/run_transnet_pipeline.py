import os
import argparse
import glob
from tqdm import tqdm
import pandas as pd
import cv2
import sys

# Ensure correct Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extract.workers.transnet import TransNetV2Detector

def main():
    parser = argparse.ArgumentParser(description="Batch extract keyframes using TransNetV2")
    parser.add_argument("--video-dir", type=str, required=True, help="Directory containing raw videos")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the extracted keyframe CSVs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    video_files = glob.glob(os.path.join(args.video_dir, "*.mp4"))
    
    if not video_files:
        print(f"No .mp4 files found in {args.video_dir}")
        return

    print(f"Found {len(video_files)} videos. Loading TransNetV2 model...")
    detector = TransNetV2Detector(device="cuda")

    for video_path in tqdm(video_files, desc="Extracting Keyframes"):
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        out_csv = os.path.join(args.output_dir, f"{video_id}.csv")
        
        if os.path.exists(out_csv):
            print(f"Skipping {video_id}, already exists.")
            continue
            
        shots = detector.detect_shots(video_path, max_keyframes=1000)
        
        if not shots:
            print(f"Warning: No shots detected for {video_id}")
            continue
            
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        
        rows = []
        for shot in shots:
            rows.append({
                "n": shot["shot_id"],
                "pts_time": shot["start_sec"],
                "fps": fps,
                "frame_idx": shot["keyframe_id"]
            })
            
        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False)

    print(f"\n✅ Finished extracting {len(video_files)} videos. Keyframe CSVs saved to {args.output_dir}")

if __name__ == "__main__":
    main()
