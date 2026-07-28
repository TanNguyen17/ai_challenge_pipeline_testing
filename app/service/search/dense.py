from app.repository.milvus_repository import MilvusRepository
from app.service.encoder.visual_encoder import VisualEncoder
from app.service.encoder.vn_text_encoder import VnTextEncoder
from typing import List, Dict, Any

class DenseSearchService:
    def __init__(self, milvus_repo: MilvusRepository, visual_encoder: VisualEncoder, vn_encoder: VnTextEncoder):
        self.milvus_repo = milvus_repo
        self.visual_encoder = visual_encoder
        self.vn_encoder = vn_encoder

    def search_visual(self, query_text: str, top_k: int = 100) -> List[Dict[str, Any]]:
        query_vector = self.visual_encoder.encode_text(query_text)
        return self.milvus_repo.search("keyframes_pe_core", query_vector, top_k)

    def search_transcript(self, query_text: str, top_k: int = 100) -> List[Dict[str, Any]]:
        query_vector = self.vn_encoder.encode(query_text)
        return self.milvus_repo.search("keyframes_dangvn", query_vector, top_k)
