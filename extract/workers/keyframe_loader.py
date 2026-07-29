import os
import glob
import pandas as pd
from typing import List, Dict, Any

class KeyframeLoader:
    """
    Loads pre-computed keyframes from BTC map-keyframes CSV files.
    This replaces TransNetV2 to ensure we process the exact frames 
    the BTC will evaluate us on.
    """
    def __init__(self, keyframes_root_dir: str):
        self.keyframes_root_dir = keyframes_root_dir
        self.csv_map = {}
        self._index_csv_files()

    def _index_csv_files(self):
        """Recursively find all map-keyframe CSV files and index them by video_id."""
        search_pattern = os.path.join(self.keyframes_root_dir, "**", "*.csv")
        csv_files = glob.glob(search_pattern, recursive=True)
        for fpath in csv_files:
            video_id = os.path.splitext(os.path.basename(fpath))[0]
            self.csv_map[video_id] = fpath
        print(f"✅ KeyframeLoader indexed {len(self.csv_map)} video CSVs from {self.keyframes_root_dir}")

    def load(self, video_id: str) -> List[Dict[str, Any]]:
        """
        Loads all keyframes for a video_id from its BTC CSV.
        Returns:
            List of dicts containing: shot_id, keyframe_id (frame_idx),
            start_sec (pts_time), end_sec (pts_time of next frame or +3s), fps
        """
        if video_id not in self.csv_map:
            print(f"⚠️ Warning: No keyframe CSV found for {video_id}. Returning empty.")
            return []

        csv_path = self.csv_map[video_id]
        try:
            df = pd.read_csv(csv_path)
            # Expected columns: n, pts_time, fps, frame_idx
            keyframes = []
            
            for i, row in df.iterrows():
                # End sec is next frame's start or current + 3.0s if it's the last frame
                # This gives a time_range that covers the gap to the next keyframe
                if i < len(df) - 1:
                    end_sec = df.iloc[i + 1]['pts_time']
                else:
                    end_sec = row['pts_time'] + 3.0
                
                keyframes.append({
                    "shot_id": int(row['n']),
                    "keyframe_id": int(row['frame_idx']),
                    "start_sec": float(row['pts_time']),
                    "end_sec": float(end_sec),
                    "fps": float(row['fps']),
                    "keyframe_n": int(row['n'])
                })
                
            return keyframes
        except Exception as e:
            print(f"❌ Error reading CSV {csv_path}: {e}")
            return []
