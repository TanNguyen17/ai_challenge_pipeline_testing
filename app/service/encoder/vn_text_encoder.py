import torch
from transformers import AutoTokenizer, AutoModel
from app.core.settings import settings
from app.core.logger import logger
from typing import List
import numpy as np

class VnTextEncoder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = settings.VN_TEXT_ENCODER_MODEL
        logger.info(f"Loading Vietnamese text encoder {self.model_name} on {self.device}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            self.dim = 768  # dangvantuan/vietnamese-embedding is 768-dim
            logger.info("Vietnamese text encoder loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading VN text encoder: {e}")
            self.model = None
            self.tokenizer = None
            self.dim = 768

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode(self, text: str) -> List[float]:
        if self.model is None or self.tokenizer is None:
            vec = np.random.randn(self.dim)
            vec /= np.linalg.norm(vec)
            return vec.tolist()
            
        try:
            encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(self.device)
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
            # Normalize embeddings
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            return sentence_embeddings[0].cpu().numpy().tolist()
        except Exception as e:
            logger.error(f"Error encoding VN text: {e}")
            vec = np.random.randn(self.dim)
            vec /= np.linalg.norm(vec)
            return vec.tolist()
