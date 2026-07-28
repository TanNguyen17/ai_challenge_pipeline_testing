import os
import glob
import json
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
from app.models.media import VideoMetadata
from app.core.settings import settings
from app.core.logger import logger

async def init_db():
    client = AsyncIOMotorClient(settings.MONGO_URI)
    await init_beanie(
        database=client[settings.MONGO_DB_NAME],
        document_models=[VideoMetadata]
    )

async def migrate_video_metadata():
    logger.info("Initializing MongoDB for migration...")
    await init_db()
    
    # Search for all media-info JSON files
    json_files = glob.glob("d:/AI-HCMC/data/extracted/**/media-info/*.json", recursive=True)
    logger.info(f"Found {len(json_files)} media info JSON files.")
    
    for json_path in json_files:
        video_id = os.path.splitext(os.path.basename(json_path))[0]
        
        # Check if already migrated
        existing = await VideoMetadata.find_one(VideoMetadata.video_id == video_id)
        if existing:
            continue
            
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            # Create video metadata
            # length = duration in seconds
            video = VideoMetadata(
                video_id=video_id,
                video_path=f"data/raw/videos/{video_id}.mp4",
                duration_seconds=float(data.get("length", 0)),
                fps=30.0,  # default fallback
                total_frames=int(data.get("length", 0) * 30),
                summary=data.get("description", "")
            )
            await video.save()
            logger.info(f"Migrated VideoMetadata for {video_id}.")
        except Exception as e:
            logger.error(f"Failed to migrate metadata for {video_id}: {e}")

if __name__ == "__main__":
    asyncio.run(migrate_video_metadata())
