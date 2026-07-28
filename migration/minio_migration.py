import os
import glob
from app.repository.minio_repository import MinioRepository
from app.core.logger import logger

def migrate_images_to_minio():
    logger.info("Starting MinIO keyframe images migration...")
    minio_repo = MinioRepository()
    
    # Locate any JPG/PNG keyframe images
    image_files = glob.glob("d:/AI-HCMC/data/raw/keyframes/**/*.jpg", recursive=True)
    logger.info(f"Found {len(image_files)} keyframe image files to upload.")
    
    for img_path in image_files:
        filename = os.path.basename(img_path)
        video_id = os.path.basename(os.path.dirname(img_path))
        frame_name = os.path.splitext(filename)[0]
        
        try:
            frame_idx = int(frame_name)
            with open(img_path, "rb") as f:
                img_data = f.read()
                
            success = minio_repo.upload_frame(video_id, frame_idx, img_data)
            if success:
                logger.info(f"Uploaded keyframe {video_id}/{frame_idx} to MinIO.")
            else:
                logger.error(f"Failed to upload keyframe {video_id}/{frame_idx}.")
        except Exception as e:
            logger.error(f"Error migrating keyframe {img_path}: {e}")

if __name__ == "__main__":
    migrate_images_to_minio()
