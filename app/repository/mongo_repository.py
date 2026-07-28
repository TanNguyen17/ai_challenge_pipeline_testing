from typing import List, Optional
from app.models.media import VideoMetadata, KeyframeMetadata

class MongoRepository:
    @staticmethod
    async def save_video(video: VideoMetadata) -> VideoMetadata:
        await video.save()
        return video

    @staticmethod
    async def get_video(video_id: str) -> Optional[VideoMetadata]:
        return await VideoMetadata.find_one(VideoMetadata.video_id == video_id)

    @staticmethod
    async def save_keyframe(keyframe: KeyframeMetadata) -> KeyframeMetadata:
        await keyframe.save()
        return keyframe

    @staticmethod
    async def get_keyframe(video_id: str, frame_idx: int) -> Optional[KeyframeMetadata]:
        return await KeyframeMetadata.find_one(
            KeyframeMetadata.video_id == video_id,
            KeyframeMetadata.frame_idx == frame_idx
        )

    @staticmethod
    async def get_keyframes_for_video(video_id: str) -> List[KeyframeMetadata]:
        return await KeyframeMetadata.find(KeyframeMetadata.video_id == video_id).to_list()
