try:
    from prefect import task, flow
except ImportError:
    # Fallback to no-op decorators if prefect is not installed
    def task(fn): return fn
    def flow(fn): return fn

from extract.workers.dake import DakeSelector
from extract.workers.pe_core import VisualEmbeddingWorker
from app.service.encoder.visual_encoder import VisualEncoder
from app.core.logger import logger
from typing import List, Dict, Any
import os
import json

@task
def select_keyframes_task(video_path: str) -> List[int]:
    logger.info(f"Selecting keyframes for {video_path}")
    return DakeSelector.select_keyframes(video_path)

@task
def extract_embeddings_task(video_path: str, frame_indices: List[int]) -> List[Dict[str, Any]]:
    logger.info(f"Extracting embeddings for {video_path} at {len(frame_indices)} frames")
    encoder = VisualEncoder()
    worker = VisualEmbeddingWorker(encoder)
    return worker.extract_features(video_path, frame_indices)

@flow(name="Video Processing Flow")
def process_video_flow(video_path: str, output_dir: str):
    logger.info(f"Starting video processing flow for {video_path}")
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    
    # 1. Keyframe extraction
    frame_indices = select_keyframes_task(video_path)
    
    # 2. Embedding extraction
    features = extract_embeddings_task(video_path, frame_indices)
    
    # Save output metadata
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"{video_id}_features.json")
    with open(out_file, "w") as f:
        json.dump({
            "video_id": video_id,
            "keyframes": frame_indices,
            "embeddings": features
        }, f)
        
    logger.info(f"Video flow completed. Saved to {out_file}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        process_video_flow(sys.argv[1], sys.argv[2])
