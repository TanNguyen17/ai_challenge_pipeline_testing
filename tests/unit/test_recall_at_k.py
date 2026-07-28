from eval.benchmarks.recall_at_k import calculate_recall_at_k, calculate_mean_recall_at_k

def test_recall_at_k_hit():
    retrieved = [
        ("video1", 100),
        ("video2", 200),
        ("video3", 300)
    ]
    gt = ("video2", 150, 250)
    
    assert calculate_recall_at_k(retrieved, gt, k=1) == 0.0
    assert calculate_recall_at_k(retrieved, gt, k=2) == 1.0
    assert calculate_recall_at_k(retrieved, gt, k=5) == 1.0

def test_recall_at_k_miss():
    retrieved = [
        ("video1", 100),
        ("video2", 400),
        ("video3", 300)
    ]
    gt = ("video2", 150, 250)
    
    assert calculate_recall_at_k(retrieved, gt, k=3) == 0.0

def test_mean_recall():
    all_retrieved = [
        [("video1", 100), ("video2", 200)],
        [("video1", 100), ("video3", 300)]
    ]
    all_gt = [
        ("video2", 150, 250),
        ("video2", 150, 250)
    ]
    
    # First query hits at k=2, second query misses at k=2
    assert calculate_mean_recall_at_k(all_retrieved, all_gt, k=2) == 0.5
