# 🏆 SOTA Video OCR Pipeline & RunPod Deployment Guide
> **Dự án**: AI Challenge HCMC (Video Retrieval Engine)  
> **Chuyên mục**: Video OCR Extraction, Text Tracking, Temporal Aggregation & Ground Truth Evaluation via **`uv`**

---

## 📌 1. TỔNG QUAN KIẾN TRÚC PIPELINE (5-STAGE ARCHITECTURE)

Hệ thống xử lý OCR video được thiết kế theo kiến trúc **Hybrid SOTA (Deep Learning Perception + Classical Temporal Aggregation)** nhằm giải quyết dứt điểm các bẫy dữ liệu: *lặp chữ thừa, xé lẻ câu chữ chạy, nhiễu chữ overlay vs scene, và chi phí tính toán GPU*.

```
[VIDEO GỐC MP4]
       │
       ▼
[STAGE 1: TransNetV2 Shot Boundary Detection]
       └── Cắt cảnh video, trích 1-2 Keyframe/Shot (Giảm 85% tải GPU)
       │
       ▼
[STAGE 2: PP-OCRv5 Detection & Recognition]
       ├── Detection: Large-Kernel PAN (Tìm Bounding Box chữ nhỏ/mờ)
       └── Recognition: SVTR_LCNetV3 (Đọc chữ Tiếng Việt có dấu)
       │
       ▼
[STAGE 3: ByteTrack Text Tracking]
       └── Gom các dải chữ lặp qua các frames liên tiếp vào chung 1 Tracklet_ID
       │
       ▼
[STAGE 4: Text Alignment & LCS Substring Stitching]
       ├── Chữ đứng yên: Character-level Majority Voting (Lọc nhiễu chính tả)
       └── Chữ chạy (Ticker): RapidFuzz / LCS Substring Stitching (Ghép câu chữ chạy)
       │
       ▼
[STAGE 5: Dynamic Layout Classifier & Elasticsearch Indexing]
       ├── Phân loại: ocr_overlay | ocr_scene | ocr_system
       └── Index vào DB: ocr_raw + ocr_no_accent (Tiếng Việt không dấu) + ocr_ngram
```

---

## 📋 2. INPUT / OUTPUT CONTRACT CHO TỪNG STAGE

### Stage 1: TransNetV2 (Shot Boundary Detection & Sampling)
- **Input**: Video MP4 gốc.
- **Output**:
```json
[
  {
    "shot_id": 0,
    "start_frame": 0,
    "end_frame": 120,
    "start_sec": 0.0,
    "end_sec": 4.8,
    "keyframe_id": 60,
    "keyframe_path": "./keyframes/L21_V001/shot_0_frame_60.webp"
  }
]
```

### Stage 2: PP-OCRv5 (Text Spotting)
- **Input**: Keyframe image (.webp / .jpg).
- **Output**:
```json
[
  {
    "keyframe_id": 60,
    "timestamp_sec": 2.4,
    "ocr_raw_detections": [
      {
        "bbox": [[120, 950], [980, 950], [980, 1020], [120, 1020]],
        "text": "9 TRIỆU ĐẾN NHA TRANG",
        "confidence": 0.96
      },
      {
        "bbox": [[1600, 40], [1820, 40], [1820, 90], [1600, 90]],
        "text": "HTV7",
        "confidence": 0.99
      }
    ]
  }
]
```

### Stage 3: ByteTrack (Text Tracking)
- **Input**: Sequential OCR detections across keyframes.
- **Output**:
```json
[
  {
    "tracklet_id": "TRK_001",
    "shot_id": 0,
    "first_seen_sec": 0.0,
    "last_seen_sec": 4.8,
    "observations": [
      {"keyframe_id": 30, "text": "9 TRIỆU ĐẾN NHA TR...", "confidence": 0.91},
      {"keyframe_id": 60, "text": "9 TRIỆU ĐẾN NHA TRANG", "confidence": 0.96},
      {"keyframe_id": 90, "text": "TRIỆU ĐẾN NHA TRANG - KHÁNH HÒA", "confidence": 0.94}
    ]
  }
]
```

### Stage 4: Text Alignment & LCS Stitching
- **Input**: Tracklets from Stage 3.
- **Output**:
```json
[
  {
    "shot_id": 0,
    "tracklet_id": "TRK_001",
    "time_range": {"start_sec": 0.0, "end_sec": 4.8},
    "stitched_clean_text": "9 TRIỆU ĐẾN NHA TRANG - KHÁNH HÒA",
    "primary_bbox": [120, 950, 980, 1020],
    "avg_confidence": 0.95
  }
]
```

### Stage 5: Dynamic Layout & Elasticsearch Document
- **Input**: Stitched text + Primary BBox + Duration.
- **Output (Final Database Document)**:
```json
{
  "video_id": "L21_V001",
  "shot_id": 0,
  "time_range": {"start_sec": 0.0, "end_sec": 4.8},
  "keyframe_ids": [30, 60, 90],
  
  "ocr_overlay": "9 TRIỆU ĐẾN NHA TRANG - KHÁNH HÒA",
  "ocr_scene": null,
  "ocr_system": "HTV7",
  
  "ocr_no_accent": "9 TRIEU DEN NHA TRANG KHANH HOA HTV7",
  "ocr_full_combined": "9 TRIỆU ĐẾN NHA TRANG - KHÁNH HÒA HTV7"
}
```

---

## 🚀 3. HƯỚNG DẪN TRIỂN KHAI THỰC NGHIỆM TRÊN RUNPOD VỚI `uv`

### 3.1 Cài đặt & Đồng bộ Môi trường với `uv`
```bash
pip install uv
uv sync
```

### 3.2 Kịch bản Chạy Lệnh Thực Nghiệm bằng `uv run`

```bash
# 1. Tải dữ liệu Benchmark (~650 videos, ~7.4GB) từ HuggingFace
uv run python download_data.py --phase benchmark

# 2. Giải nén dữ liệu video
uv run python extract_data.py --raw-dir ./data/raw --extract-dir ./data/extracted

# 3. Chạy Pipeline OCR 5-Stage
uv run python pipelines/run_ocr_pipeline.py --video-dir ./data/extracted --output-dir ./data/processed/ocr --limit 500

# 4. Chạy Đánh giá Ground Truth Evaluation & A/B Testing
uv run python eval/evaluate_benchmark.py --processed-dir ./data/processed --query-dir ./data/raw/query
```

---

## 📊 4. KHUNG ĐO ĐẠC HIỆU NĂNG & ĐÁNH GIÁ GROUND TRUTH (GT EVALUATION)

### 4.1 Chỉ số Đo đạc Từng Stage
| Stage | Chỉ số cần ghi Log (Metric) | Công thức / Đo bằng | Giá trị Kỳ vọng |
| :--- | :--- | :--- | :--- |
| **Stage 1 (TransNetV2)** | Frame Reduction Ratio (%) | $1 - \frac{\text{Keyframes}}{\text{Total Frames}}$ | **85% – 90%** reduction |
| **Stage 2 (PP-OCRv5)** | Inference Latency & VRAM | `time.time()` & `torch.cuda.max_memory()` | **10–15ms/frame**, VRAM $< 4\text{GB}$ |
| **Stage 3 (ByteTrack)** | Tracklet Compression Ratio | $\frac{\text{Raw BBoxes}}{\text{Unique Tracklets}}$ | **3x – 8x** compression |
| **Stage 4 (LCS Stitching)**| Text Deduplication Ratio | $1 - \frac{\text{Clean Sentences}}{\text{Raw Extracted Text}}$ | **75% – 85%** deduplicated |
| **Stage 5 (Indexing)** | Top-5 Retrieval Recall | Test 20 KIS Queries | **$> 90\%$ Recall** |

### 4.2 Đánh giá Thực tế bằng Bộ Đề Thi Ground Truth (A/B Testing)
Để đo đạc độ hiệu quả thực tế trên đề thi cuộc thi:
1. **Recall@K (K=1, 5, 10)**: Kiểm tra xem `(video_id, frame_id)` tìm ra có trùng khớp với `[start_frame, end_frame]` của Ground Truth hay không.
2. **MRR (Mean Reciprocal Rank)**: Đo vị trí xếp hạng trung bình của đáp án đúng.
3. **A/B Testing Delta**: So sánh **Kịch bản A (Chưa có OCR)** vs **Kịch bản B (Đã bật OCR SOTA)** trên 50 câu hỏi KIS OCR để đo mức tăng trưởng điểm **Recall@5** (Kỳ vọng tăng **+25% - +35%**).
