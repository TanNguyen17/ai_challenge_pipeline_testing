import cv2
import os
from typing import List

class DakeSelector:
    @staticmethod
    def select_keyframes(video_path: str, threshold: float = 12.0) -> List[int]:
        """
        Selects representative keyframes using frame differences (simplification of DAKE).
        video_path: path to raw mp4 video file
        threshold: sensitivity parameter for scene change detection
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
            
        frame_indices = []
        prev_frame = None
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (100, 100)) # resize to speed up comparison
            
            if prev_frame is None:
                frame_indices.append(frame_idx)
                prev_frame = gray
            else:
                # Calculate absolute difference
                diff = cv2.absdiff(gray, prev_frame)
                mean_diff = diff.mean()
                if mean_diff > threshold:
                    frame_indices.append(frame_idx)
                    prev_frame = gray
            
            frame_idx += 1
            
        cap.release()
        return frame_indices
