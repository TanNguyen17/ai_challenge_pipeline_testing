# 🏆 AI Challenge HCMC — Multimodal Video Retrieval Pipeline Suite
> **Hệ thống Trích xuất & Benchmark Multimodal SOTA**: Video OCR, Vietnamese ASR, và Open-Vocabulary Object Detection dành cho RunPod Server sử dụng trình quản lý gói siêu tốc **`uv`**.

---

## 📌 1. BẢNG HƯỚNG DẪN CHI TIẾT TỪNG NHÁNH (PIPELINE GUIDES)

Hệ thống được tổ chức thành 3 pipeline độc lập, chuyên sâu với tài liệu hướng dẫn và mã nguồn hoàn chỉnh:

| Nhánh Multimodal | Công nghệ SOTA Nòng cốt | File Hướng dẫn Chi tiết & RunPod Blueprint |
| :--- | :--- | :--- |
| 🔤 **Video OCR** | TransNetV2 + **PP-OCRv5** + ByteTrack + RapidFuzz LCS | 📄 [OCR_PIPELINE_RUNPOD_GUIDE.md](file:///d:/AI-HCMC/OCR_PIPELINE_RUNPOD_GUIDE.md) |
| 🎙️ **Vietnamese ASR** | Silero VAD + **PhoWhisper-large (VinAI)** + WhisperX + ITN | 📄 [ASR_PIPELINE_RUNPOD_GUIDE.md](file:///d:/AI-HCMC/ASR_PIPELINE_RUNPOD_GUIDE.md) |
| 🎯 **Object Detection** | Spatial UI Masking + **YOLO-World v2** + RAM++ + **Florence-2** | 📄 [OBJECT_DETECTION_PIPELINE_RUNPOD_GUIDE.md](file:///d:/AI-HCMC/OBJECT_DETECTION_PIPELINE_RUNPOD_GUIDE.md) |

---

## 📁 2. CẤU TRÚC MÃ NGUỒN DỰ ÁN (PROJECT STRUCTURE)

```
d:\AI-HCMC\
├── pyproject.toml                      # File cấu hình dependencies cho `uv`
├── download_data.py                    # Tải bộ dữ liệu & queries từ HuggingFace
├── extract_data.py                     # Giải nén các tập dữ liệu zip (.mp4 / metadata)
│
├── pipelines/                          # BỘ EXECUTION SCRIPTS 3 PIPELINES
│   ├── run_ocr_pipeline.py             # Script chạy 5-Stage SOTA Video OCR
│   ├── run_asr_pipeline.py             # Script chạy 5-Stage SOTA Vietnamese ASR
│   ├── run_obj_detection_pipeline.py   # Script chạy 5-Stage SOTA Object Detection
│   └── run_all_pipelines.py            # Master runner chạy cả 3 pipelines nối tiếp
│
├── eval/                               # BỘ BENCHMARK & GROUND TRUTH EVALUATION SUITE
│   ├── benchmarks/
│   │   ├── recall_at_k.py              # Script tính Recall@1, Recall@5, Recall@10
│   │   └── mrr_calculator.py           # Script tính MRR (Mean Reciprocal Rank) chuẩn BTC
│   └── evaluate_benchmark.py           # Master runner đánh giá Ground Truth & A/B Testing
│
├── OCR_PIPELINE_RUNPOD_GUIDE.md        # Hướng dẫn chi tiết & RunPod Blueprint cho OCR
├── ASR_PIPELINE_RUNPOD_GUIDE.md        # Hướng dẫn chi tiết & RunPod Blueprint cho ASR
└── OBJECT_DETECTION_PIPELINE_RUNPOD_GUIDE.md # Hướng dẫn chi tiết & RunPod Blueprint cho Obj Det
```

---

## 🚀 3. HƯỚNG DẪN CHẠY THỰC NGHIỆM TRÊN RUNPOD VỚI `uv` (QUICKSTART)

### 3.1 Cài đặt `uv` & Đồng bộ Môi trường (Environment Sync)
```bash
# 1. Cài đặt uv (nếu server chưa có)
pip install uv

# 2. Tạo virtualenv và cài tự động 100% dependencies bằng uv sync
uv sync
```

### 3.2 Tải & Giải nén Dữ liệu Benchmark (~650 Videos, ~7.4 GB) bằng `uv run`
```bash
# Tải gói dữ liệu Benchmark
uv run python download_data.py --phase benchmark

# Giải nén dữ liệu video
uv run python extract_data.py --raw-dir ./data/raw --extract-dir ./data/extracted
```

### 3.3 Chạy Trích xuất Dữ liệu cho Cả 3 Pipelines bằng `uv run`
```bash
# Chạy cả 3 pipeline OCR, ASR, Object Detection trên 500 video benchmark
uv run python pipelines/run_all_pipelines.py --video-dir ./data/extracted --output-base-dir ./data/processed --limit 500
```

### 3.4 Chạy Đánh giá Ground Truth Evaluation & A/B Testing bằng `uv run`
```bash
# Đánh giá hiệu quả tìm kiếm Recall@K & MRR dựa trên bộ câu hỏi đề thi
uv run python eval/evaluate_benchmark.py --processed-dir ./data/processed --query-dir ./data/raw/query
```

---

## 📊 4. PHƯƠNG PHÁP ĐÁNH GIÁ ĐỘ HIỆU QUẢ (GROUND TRUTH BENCHMARKING)

Để khẳng định tính hiệu quả của các pipeline trên bài toán thi đấu, dự án áp dụng kịch bản **A/B Testing đối soát với Ground Truth**:
1. **Recall@K (K=1, 5, 10, 100)**: Đo tỉ lệ tìm thấy khoảng khung hình Ground Truth `[start_frame, end_frame]` trong Top-K kết quả.
2. **MRR (Mean Reciprocal Rank)**: Đo vị trí xuất hiện của đáp án đúng trong bảng xếp hạng.
3. **A/B Testing Delta**: Đo mức tăng trưởng điểm Recall@5 khi bật từng nhánh pipeline so với Baseline (Vector Only). Kỳ vọng tăng **+25% - +40%** điểm Recall@5.
