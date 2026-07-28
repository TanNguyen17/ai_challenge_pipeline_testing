from app.service.search.dense import DenseSearchService
from app.service.search.bm25 import BM25SearchService
from app.service.search.fusion import FusionService
from app.service.search.diversify import DiversifyService
from typing import List, Dict, Any

class MultiSignalRetriever:
    def __init__(self, dense_search: DenseSearchService, bm25_search: BM25SearchService):
        self.dense_search = dense_search
        self.bm25_search = bm25_search

    def search(self, query_text: str, top_k: int = 100, frame_gap: int = 450) -> List[Dict[str, Any]]:
        # 1. Parallel search on multiple channels
        # Retrieve 300 candidates per channel to ensure decent overlap for RRF
        visual_hits = self.dense_search.search_visual(query_text, top_k=300)
        transcript_hits = self.dense_search.search_transcript(query_text, top_k=300)
        ocr_hits = self.bm25_search.search_ocr(query_text, top_k=300)
        asr_hits = self.bm25_search.search_asr(query_text, top_k=300)

        # 2. Reciprocal Rank Fusion
        fused_hits = FusionService.reciprocal_rank_fusion(
            [visual_hits, transcript_hits, ocr_hits, asr_hits],
            k=60
        )

        # 3. Temporal deduplication
        final_hits = DiversifyService.temporal_deduplicate(fused_hits, frame_gap=frame_gap, top_k=top_k)

        return final_hits
