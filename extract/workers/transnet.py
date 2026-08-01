import os
import cv2
import numpy as np
import torch
from typing import List, Dict, Any
from transnetv2_pytorch import TransNetV2

class TransNetV2Detector:
    """
    SOTA TransNetV2 Shot Boundary Detector Engine.
    Executes 3D Convolutional Neural Network inference on GPU to predict exact video scene cuts.
    """
    def __init__(self, device: str = "cuda", weights_path: str = None):
        self.device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        self.model = None
        self.weights_path = weights_path or "./data/models/transnetv2_weights.pth"
        self._init_transnet_model()

    def _init_transnet_model(self):
        """Initializes TransNetV2 PyTorch 3D-CNN backbone and loads pre-trained weights."""
        try:
            self.model = TransNetV2()
            self.model.eval().to(self.device)

            # Load weights if available
            if os.path.exists(self.weights_path):
                state_dict = torch.load(self.weights_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print(f"✅ TransNetV2 real PyTorch weights loaded from {self.weights_path}.")
            else:
                print(f"⚠️ TransNetV2 weights missing at {self.weights_path}. Model will run but accuracy will be degraded without real weights. Please download transnetv2-pytorch-weights.pth.")
        except Exception as e:
            print(f"⚠️ TransNetV2 initialization error: {e}")

    def detect_shots(self, video_path: str, max_keyframes: int = 15, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Executes TransNetV2 3D-CNN inference on video tensor [1, 3, T, 27, 48] to detect scene cuts.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames = []
        frame_stride = 3 # Fast tensor batching

        while True:
            for _ in range(frame_stride - 1):
                if not cap.grab():
                    break
            ret, frame = cap.read()
            if not ret:
                break
            # Resize to TransNetV2 27x48 RGB tensor format
            resized = cv2.resize(frame, (48, 27))
            resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            frames.append(resized_rgb)

        cap.release()

        if not frames:
            return []

        # Convert to 3D tensor: [B=1, T, H=27, W=48, C=3] for official TransNetV2 PyTorch input format
        tensor_frames = np.array(frames, dtype=np.uint8)
        # transnetv2-pytorch requires inputs of shape (B, T, H, W, C) in RGB format (0-255 uint8)
        input_tensor = torch.tensor(tensor_frames).unsqueeze(0).to(self.device) # [1, T, H, W, C]

        # Run TransNetV2 3D-CNN Forward Pass on GPU
        shots = []
        with torch.no_grad():
            if self.model is not None:
                single_frame_pred, all_frames_pred = self.model(input_tensor)
                predictions = single_frame_pred[0].cpu().numpy() # [T]
            else:
                predictions = np.zeros(len(frames))

        start_frame = 0
        shot_id = 0

        for idx, prob in enumerate(predictions):
            actual_frame_idx = idx * frame_stride
            if prob > threshold or (actual_frame_idx - start_frame) > int(fps * 6):
                mid_frame = (start_frame + actual_frame_idx) // 2
                shots.append({
                    "shot_id": shot_id,
                    "start_frame": start_frame,
                    "end_frame": actual_frame_idx,
                    "start_sec": round(start_frame / fps, 2),
                    "end_sec": round(actual_frame_idx / fps, 2),
                    "keyframe_id": mid_frame,
                    "transition_probability": round(float(prob), 3)
                })
                shot_id += 1
                start_frame = actual_frame_idx

        if not shots and total_frames > 0:
            shots.append({
                "shot_id": 0,
                "start_frame": 0,
                "end_frame": total_frames,
                "start_sec": 0.0,
                "end_sec": round(total_frames / fps, 2),
                "keyframe_id": total_frames // 2,
                "transition_probability": 0.0
            })

        return shots[:max_keyframes]

if __name__ == "__main__":
    import argparse
    import pandas as pd
    parser = argparse.ArgumentParser(description="Run TransNetV2 to extract shot boundaries.")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, required=True, help="Directory to save CSV")
    args = parser.parse_args()

    detector = TransNetV2Detector()
    shots = detector.detect_shots(args.video, max_keyframes=1000)
    
    # Save to BTC-compatible CSV format
    os.makedirs(args.output, exist_ok=True)
    video_id = os.path.splitext(os.path.basename(args.video))[0]
    out_csv = os.path.join(args.output, f"{video_id}.csv")
    
    rows = []
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    
    for shot in shots:
        rows.append({
            "n": shot["shot_id"],
            "pts_time": shot["start_sec"],
            "fps": fps,
            "frame_idx": shot["keyframe_id"]
        })
        
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"✅ TransNetV2 extracted {len(shots)} shots. Saved to {out_csv}")
