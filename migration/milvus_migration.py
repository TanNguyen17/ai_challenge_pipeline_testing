import os
import glob
import numpy as np
import pandas as pd
from app.repository.milvus_repository import MilvusRepository
from app.core.logger import logger

def migrate_embeddings():
    logger.info("Starting Milvus embedding migration...")
    milvus_repo = MilvusRepository()
    
    # Create keyframes_pe_core collection (dim=512 for baseline CLIP features)
    collection_name = "keyframes_pe_core"
    milvus_repo.create_collection(collection_name, dim=512)
    
    # Search for all .npy files in both batch 1 and batch 2
    npy_files = glob.glob("d:/AI-HCMC/data/extracted/**/clip-features-32/*.npy", recursive=True)
    logger.info(f"Found {len(npy_files)} numpy embedding files.")
    
    for npy_path in npy_files:
        video_id = os.path.splitext(os.path.basename(npy_path))[0]
        
        # Find the corresponding map-keyframes CSV file
        csv_pattern = f"d:/AI-HCMC/data/extracted/**/map-keyframes/{video_id}.csv"
        csv_files = glob.glob(csv_pattern, recursive=True)
        
        if not csv_files:
            logger.warn(f"No keyframe map CSV found for {video_id}, skipping.")
            continue
            
        csv_path = csv_files[0]
        try:
            df = pd.read_csv(csv_path)
            embeddings = np.load(npy_path)
            
            if len(df) != len(embeddings):
                logger.error(f"Mismatch between CSV ({len(df)}) and numpy ({len(embeddings)}) for {video_id}!")
                continue
                
            entities = []
            for idx, row in df.iterrows():
                frame_idx = int(row["frame_idx"])
                emb = embeddings[idx].astype(float).tolist() # Milvus requires python float list
                entities.append({
                    "video_id": video_id,
                    "frame_idx": frame_idx,
                    "embedding": emb
                })
                
            # Batch insert in chunks of 500 to keep memory footprint low
            for i in range(0, len(entities), 500):
                chunk = entities[i:i+500]
                milvus_repo.insert(collection_name, chunk)
                
            logger.info(f"Successfully migrated {len(entities)} keyframes for {video_id}.")
        except Exception as e:
            logger.error(f"Failed to migrate embeddings for {video_id}: {e}")

if __name__ == "__main__":
    migrate_embeddings()
