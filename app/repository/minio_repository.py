from minio import Minio
from app.core.settings import settings
import io
from typing import Optional

class MinioRepository:
    def __init__(self):
        self.client = Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE
        )
        self.bucket = settings.MINIO_BUCKET_NAME
        self._ensure_bucket()

    def _ensure_bucket(self):
        try:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
        except Exception:
            pass

    def upload_frame(self, video_id: str, frame_idx: int, file_data: bytes) -> bool:
        object_name = f"{video_id}/{frame_idx}.jpg"
        try:
            self.client.put_object(
                self.bucket,
                object_name,
                io.BytesIO(file_data),
                len(file_data),
                content_type="image/jpeg"
            )
            return True
        except Exception:
            return False

    def get_frame(self, video_id: str, frame_idx: int) -> Optional[bytes]:
        object_name = f"{video_id}/{frame_idx}.jpg"
        try:
            response = self.client.get_object(self.bucket, object_name)
            data = response.read()
            response.close()
            response.release_conn()
            return data
        except Exception:
            return None
