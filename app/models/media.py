from beanie import Document, Indexed
from pydantic import Field
from typing import List, Optional, Dict, Any

class VideoMetadata(Document):
    video_id: Indexed(str, unique=True)
    video_path: str

    duration_seconds: float
    fps: float
    total_frames: int
    summary: Optional[str] = None

    class Settings:
        name = "videos"

class KeyframeMetadata(Document):
    video_id: str
    frame_idx: int
    pts_time: float
    ocr_text: Optional[str] = None
    asr_text: Optional[str] = None
    yolo_objects: List[Dict[str, Any]] = []
    caption: Optional[str] = None

    class Settings:
        name = "keyframes"
        indexes = [
            "video_id",
            ("video_id", "frame_idx")
        ]
