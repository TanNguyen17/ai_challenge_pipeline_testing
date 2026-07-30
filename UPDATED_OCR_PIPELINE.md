# 🏆 SOTA Video OCR Pipeline (VLM Edition) & RunPod Deployment Guide
> **Dự án**: AI Challenge HCMC (Video Retrieval Engine)  
> **Chuyên mục**: VLM OCR Extraction, Temporal Aggregation & Ground Truth Evaluation via **`uv`**

---

## 📌 1. TỔNG QUAN KIẾN TRÚC PIPELINE

Hệ thống xử lý OCR video hiện tại được thiết kế theo kiến trúc **Vision-Language Model (VLM)** sử dụng **Qwen2-VL-2B-Instruct**, nhằm đơn giản hóa pipeline, tăng khả năng đọc hiểu ngữ cảnh trực tiếp trên ảnh nguyên bản thay vì các bước tracking text phức tạp truyền thống.

```
[VIDEO GỐC MP4]
       │
       ▼
[STAGE 1: BTC Keyframe Loading]
       └── Đọc danh sách keyframes trực tiếp từ `map-keyframes/*.csv` của Ban Tổ Chức.
       │
       ▼
[STAGE 2: Qwen2-VL-2B-Instruct Extraction]
       └── Chạy batch inference ảnh Keyframe qua VLM Qwen2-VL-2B để trích xuất văn bản tiếng Việt có dấu.
       │
       ▼
[STAGE 3: JSONL Document Export (Span & Shot)]
       ├── Lưu `doc_type = "span"` cho mỗi khung hình (frame)
       └── Rollup gom nhóm `doc_type = "shot"` chứa tất cả text trong phạm vi shot đó.
```

---

## 📋 2. INPUT / OUTPUT CONTRACT CHO TỪNG STAGE

### Stage 1: Keyframe Loading
- **Input**: Video MP4 gốc và CSV keyframes của BTC.
- **Output**: List các khung hình (frames) cần trích xuất.

### Stage 2 & 3: Qwen2-VL Extraction & Document Export
- **Input**: Keyframe image RGB.
- **Output (Final Database Documents)**:

**1. Per-Frame Span Document (`doc_type: "span"`)**
```json
{
  "doc_type": "span",
  "video_id": "L21_V001",
  "shot_id": 1,
  "tracklet_id": "TRK_00000",
  "frame_idx": 0,
  "keyframe_n": 1,
  "time_range": {"start_sec": 0.0, "end_sec": 3.0},
  "ocr_raw_full": "HTV9 HD",
  "ocr_no_accent": "htv9 hd",
  "ocr_system": "qwen2-vl-2b",
  "confidence": 1.0
}
```

**2. Per-Shot Rollup Document (`doc_type: "shot"`)**
```json
{
  "doc_type": "shot",
  "video_id": "L21_V001",
  "shot_id": 1,
  "time_range": {"start_sec": 0.0, "end_sec": 3.0},
  "ocr_full_combined": "HTV9 HD | Thời sự"
}
```

---

## 🚀 3. HƯỚNG DẪN TRIỂN KHAI THỰC NGHIỆM TRÊN RUNPOD VỚI `uv`

### 3.1 Cài đặt Môi trường
Cần cài đặt bộ thư viện `transformers` và `qwen-vl-utils`:
```bash
uv pip install transformers>=4.45.0 qwen-vl-utils
```

### 3.2 Kịch bản Chạy Lệnh Thực Nghiệm
```bash
# Chạy Pipeline OCR bằng Qwen2-VL
uv run python pipelines/run_ocr_pipeline.py --video-dir ./data/extracted --output-dir ./data/processed/ocr --keyframes-dir ./data/extracted/map-keyframes --limit 50
```

---

## 📊 4. KHUNG ĐO ĐẠC HIỆU NĂNG

| Pipeline | Model | Vai trò | Metric Ghi nhận |
| :--- | :--- | :--- | :--- |
| **Stage 1 (Keyframes)** | BTC CSV | Đồng bộ mốc thời gian | Số lượng frames / Video |
| **Stage 2 (VLM)** | Qwen2-VL-2B | Trích xuất Text Tiếng Việt | Latency inference (GPU) & Số records |
| **Stage 3 (Export)** | JSONL | Lưu DB Span + Shot | Tổng file size jsonl |
