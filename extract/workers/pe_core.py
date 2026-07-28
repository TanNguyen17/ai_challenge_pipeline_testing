from app.service.encoder.visual_encoder import VisualEncoder
from typing import List, Dict, Any
import cv2

class VisualEmbeddingWorker:
    def __init__(self, visual_encoder: VisualEncoder):
        self.visual_encoder = visual_encoder

    def extract_features(self, video_path: str, frame_indices: List[int]) -> List[Dict[str, Any]]:
        """
        Reads raw frame images from video at frame_indices, encodes them into dense vectors.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
            
        features = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Convert frame to JPEG bytes
            success, buffer = cv2.imencode('.jpg', frame)
            if not success:
                continue
            image_bytes = buffer.tobytes()
            
            # Encode image bytes to vector representation
            vec = self.visual_encoder.encode_image(image_bytes)
            features.append({
                "frame_idx": idx,
                "embedding": vec
            })
            
        cap.release()
        return features
