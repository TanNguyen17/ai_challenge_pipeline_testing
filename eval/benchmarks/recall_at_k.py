from typing import List, Set, Tuple

def calculate_recall_at_k(retrieved_results: List[Tuple[str, int]], ground_truth: Tuple[str, int, int], k: int) -> float:
    """
    Calculates Recall@k for a single query.
    retrieved_results: List of (video_id, frame_idx) sorted by rank.
    ground_truth: Tuple of (video_id, start_frame, end_frame) representing the target moment.
    """
    gt_video, gt_start, gt_end = ground_truth
    
    # Look at top k retrieved results
    top_k_results = retrieved_results[:k]
    
    for video_id, frame_idx in top_k_results:
        if video_id == gt_video and gt_start <= frame_idx <= gt_end:
            return 1.0
            
    return 0.0

def calculate_mean_recall_at_k(all_retrieved: List[List[Tuple[str, int]]], all_gt: List[Tuple[str, int, int]], k: int) -> float:
    if not all_retrieved or not all_gt or len(all_retrieved) != len(all_gt):
        return 0.0
        
    recalls = [calculate_recall_at_k(ret, gt, k) for ret, gt in zip(all_retrieved, all_gt)]
    return sum(recalls) / len(recalls)
