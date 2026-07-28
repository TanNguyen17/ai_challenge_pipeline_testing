from eval.benchmarks.vn_mteb_runner import run_vn_mteb_benchmark
from app.core.logger import logger

def main():
    candidates = [
        {"name": "dangvantuan/vietnamese-embedding"},
        {"name": "GreenNode/GreenNode-Embedding-Large-VN-Mixed-V1"},
        {"name": "BAAI/bge-m3"},
        {"name": "intfloat/multilingual-e5-large"}
    ]
    
    logger.info("Initializing encoder benchmarking pipeline...")
    results = run_vn_mteb_benchmark(candidates)
    
    print("\n" + "="*80)
    print("                 VIETNAMESE TEXT ENCODER BENCHMARK RESULTS")
    print("="*80)
    print(f"{'Candidate Model':<50} | {'Recall@1':<8} | {'Recall@5':<8} | {'Latency':<8}")
    print("-"*80)
    for model_name, metrics in results.items():
        print(f"{model_name:<50} | {metrics['Recall@1']:.3f}    | {metrics['Recall@5']:.3f}    | {metrics['latency_ms']:.1f}ms")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
