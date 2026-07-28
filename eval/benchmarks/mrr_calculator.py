from typing import List, Tuple

def calculate_mrr(retrieved_results: List[Tuple[str, int]], ground_truth: Tuple[str, int, int]) -> float:
    """
    Calculates Reciprocal Rank (RR) for a single query.
    retrieved_results: List of (video_id, frame_idx) sorted by rank (1-indexed).
    ground_truth: Tuple of (video_id, start_frame, end_frame).
    """
    gt_video, gt_start, gt_end = ground_truth
    
    for rank_idx, (video_id, frame_idx) in enumerate(retrieved_results, start=1):
        if video_id == gt_video and gt_start <= frame_idx <= gt_end:
            return 1.0 / rank_idx
            
    return 0.0

def calculate_mean_mrr(all_retrieved: List[List[Tuple[str, int]]], all_gt: List[Tuple[str, int, int]]) -> float:
    """
    Calculates Mean Reciprocal Rank (MRR) across all queries.
    """
    if not all_retrieved or not all_gt or len(all_retrieved) != len(all_gt):
        return 0.0
        
    mrr_scores = [calculate_mrr(ret, gt) for ret, gt in zip(all_retrieved, all_gt)]
    return sum(mrr_scores) / len(mrr_scores)
