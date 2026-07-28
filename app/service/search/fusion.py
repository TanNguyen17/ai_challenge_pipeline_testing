from typing import List, Dict, Any

class FusionService:
    @staticmethod
    def reciprocal_rank_fusion(rankings: List[List[Dict[str, Any]]], k: int = 60) -> List[Dict[str, Any]]:
        """
        Applies Reciprocal Rank Fusion (RRF) on multiple ranked lists.
        Each hit in a list must contain: "video_id", "frame_idx" (and optionally "score")
        """
        fused_scores = {}
        
        for ranking in rankings:
            for rank, hit in enumerate(ranking):
                doc_key = (hit["video_id"], hit["frame_idx"])
                # rank is 0-indexed, so we add 1 to make it 1-indexed
                rrf_score = 1.0 / (k + rank + 1)
                fused_scores[doc_key] = fused_scores.get(doc_key, 0.0) + rrf_score

        # Sort the results based on the fused scores in descending order
        sorted_hits = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)

        return [
            {
                "video_id": key[0],
                "frame_idx": key[1],
                "score": score
            }
            for key, score in sorted_hits
        ]
