from app.repository.es_repository import ESRepository
from typing import List, Dict, Any

class BM25SearchService:
    def __init__(self, es_repo: ESRepository):
        self.es_repo = es_repo

    def search_ocr(self, query_text: str, top_k: int = 100) -> List[Dict[str, Any]]:
        return self.es_repo.search("ocr_text", query_text, top_k)

    def search_asr(self, query_text: str, top_k: int = 100) -> List[Dict[str, Any]]:
        return self.es_repo.search("asr_transcripts", query_text, top_k)
