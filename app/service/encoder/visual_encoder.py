import torch
from PIL import Image
import io
from app.core.settings import settings
from app.core.logger import logger
from typing import List
import numpy as np

class VisualEncoder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = settings.VISUAL_ENCODER_MODEL
        logger.info(f"Loading visual encoder {self.model_name} on {self.device}...")
        self.model = None
        self.preprocess = None
        self.dim = 512

        try:
            import clip
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            self.dim = 512
            logger.info("Visual encoder (OpenAI CLIP ViT-B/32) loaded successfully.")
        except ImportError:
            try:
                import open_clip
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    'ViT-B-32', pretrained='laion2b_s34b_b79k', device=self.device
                )
                self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
                self.dim = 512
                logger.info("Visual encoder (OpenCLIP ViT-B-32) loaded successfully.")
            except Exception as ex:
                logger.error(f"Error loading clip models: {ex}. Falling back to mock encoder.")

    def encode_text(self, text: str) -> List[float]:
        if self.model is None:
            # Return dummy normalized vector
            vec = np.random.randn(self.dim)
            vec /= np.linalg.norm(vec)
            return vec.tolist()
            
        try:
            # Check if using open_clip
            if hasattr(self, 'tokenizer'):
                import open_clip
                text_tokens = self.tokenizer([text]).to(self.device)
                with torch.no_grad():
                    text_features = self.model.encode_text(text_tokens)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    return text_features[0].cpu().numpy().tolist()
            else:
                import clip
                text_tokens = clip.tokenize([text]).to(self.device)
                with torch.no_grad():
                    text_features = self.model.encode_text(text_tokens)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    return text_features[0].cpu().numpy().tolist()
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            vec = np.random.randn(self.dim)
            vec /= np.linalg.norm(vec)
            return vec.tolist()

    def encode_image(self, image_bytes: bytes) -> List[float]:
        if self.model is None:
            vec = np.random.randn(self.dim)
            vec /= np.linalg.norm(vec)
            return vec.tolist()
            
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                return image_features[0].cpu().numpy().tolist()
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            vec = np.random.randn(self.dim)
            vec /= np.linalg.norm(vec)
            return vec.tolist()
