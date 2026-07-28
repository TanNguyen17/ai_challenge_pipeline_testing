import os
import glob
import json
from app.repository.es_repository import ESRepository
from app.core.logger import logger

def migrate_lexical_data():
    logger.info("Starting Elasticsearch lexical data migration...")
    es_repo = ESRepository()
    
    # Create indices
    es_repo.create_index("ocr_text")
    es_repo.create_index("asr_transcripts")
    
    # We will populate these indices using information from media-info JSON files as a fallback baseline
    json_files = glob.glob("d:/AI-HCMC/data/extracted/**/media-info/*.json", recursive=True)
    logger.info(f"Found {len(json_files)} media info JSON files.")
    
    for json_path in json_files:
        video_id = os.path.splitext(os.path.basename(json_path))[0]
        
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            title = data.get("title", "")
            description = data.get("description", "")
            keywords = " ".join(data.get("keywords", []))
            
            combined_text = f"{title}. {description}. {keywords}"
            
            # Since we index by keyframe frame_idx, let's find the frames for this video from the map-keyframes CSV
            csv_pattern = f"d:/AI-HCMC/data/extracted/**/map-keyframes/{video_id}.csv"
            csv_files = glob.glob(csv_pattern, recursive=True)
            if not csv_files:
                continue
                
            import pandas as pd
            df = pd.read_csv(csv_files[0])
            
            # Index for each keyframe
            for _, row in df.iterrows():
                frame_idx = int(row["frame_idx"])
                doc_id = f"{video_id}_{frame_idx}"
                
                # Index into ASR and OCR index with the metadata for baseline testing
                es_repo.index_document("asr_transcripts", doc_id, video_id, frame_idx, combined_text)
                es_repo.index_document("ocr_text", doc_id, video_id, frame_idx, title) # use title as dummy OCR
                
            logger.info(f"Indexed lexical data for {video_id} (keyframes: {len(df)}).")
        except Exception as e:
            logger.error(f"Failed to migrate lexical data for {video_id}: {e}")

if __name__ == "__main__":
    migrate_lexical_data()
