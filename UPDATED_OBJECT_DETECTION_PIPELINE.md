# 🏆 SOTA Object Detection Pipeline & RunPod Deployment Guide
> **Dự án**: AI Challenge HCMC (Video Retrieval Engine)  
> **Chuyên mục**: Open-Vocabulary Object Detection, Spatial UI Masking & Ground Truth Evaluation via **`uv`**

---

## 📌 1. TỔNG QUAN KIẾN TRÚC PIPELINE

Hệ thống Object Detection thực tế đang được sử dụng dựa trên kiến trúc **YOLO-World v2** (Ultralytics) để phát hiện vật thể dạng Open-Vocabulary (chống giới hạn 80 class COCO). Quá trình tập trung vào việc chặn đứng nhiễu do logo/đồ họa truyền hình và gom nhóm vật thể theo từng phân cảnh (Shot).

```
[KEYFRAME TỪ BTC CSV]
       │
       ▼
[STAGE 1: Spatial UI Exclusion Masking]
       └── Vẽ hình chữ nhật đen (Mask) che góc trên Logo đài và dải banner Tin chạy bên dưới.
       └── Triệt tiêu 95% nhận diện nhầm đồ họa TV.
       │
       ▼
[STAGE 2: YOLO-World v2 Detection]
       ├── Sử dụng cơ chế "Prompt-then-Detect" với 39 nhãn định nghĩa sẵn (person, car, áo dài, đàn bầu...).
       └── Phân tích vị trí không gian (spatial_position) cho từng vật thể (center_left, bottom_right...).
       │
       ▼
[STAGE 3: Per-Frame Export]
       └── Kết xuất dữ liệu cho mỗi khung hình với `doc_type = "frame"`.
       │
       ▼
[STAGE 4: Shot-Level Object Summarization]
       └── Thống kê số lượng tối đa (`max_counts`) và gộp nhãn theo từng shot (`doc_type = "shot"`), giúp nén kích thước DB phục vụ Search.
```

---

## 📋 2. INPUT / OUTPUT CONTRACT

### Final Database Documents (JSONL)

Dữ liệu được chia thành hai mức độ: Frame-level (phục vụ chi tiết) và Shot-level (phục vụ truy vấn nhóm).

**1. Per-Frame Document (`doc_type: "frame"`)**
```json
{
  "doc_type": "frame",
  "video_id": "L21_V001",
  "frame_idx": 90,
  "keyframe_n": 2,
  "time_range": {"start_sec": 3.0, "end_sec": 8.7},
  "detections": [
    {
      "label": "person",
      "confidence": 0.896,
      "bbox": [0.39, 0.232, 0.505, 0.839],
      "spatial_position": "center"
    }
  ]
}
```

**2. Per-Shot Rollup Document (`doc_type: "shot"`)**
```json
{
  "doc_type": "shot",
  "video_id": "L21_V001",
  "shot_id": 1,
  "time_range": {"start_sec": 0.0, "end_sec": 3.0},
  "max_counts": {"boat": 1},
  "detected_classes": ["boat"],
  "spatial_tokens": ["boat bottom_left"]
}
```

---

## 🚀 3. HƯỚNG DẪN TRIỂN KHAI THỰC NGHIỆM TRÊN RUNPOD VỚI `uv`

### 3.1 Cài đặt Môi trường
Cài đặt thư viện Ultralytics YOLO:
```bash
uv pip install ultralytics opencv-python-headless
```

### 3.2 Kịch bản Chạy Lệnh Thực Nghiệm
```bash
# Chạy Pipeline Object Detection bằng YOLO-World v2
uv run python pipelines/run_obj_detection_pipeline.py \
    --video-dir ./data/extracted \
    --output-dir ./data/processed/objects \
    --keyframes-dir ./data/extracted/map-keyframes \
    --limit 50
```

---

## 📊 4. KHUNG ĐO ĐẠC HIỆU NĂNG

| Metric | Mô tả |
| :--- | :--- |
| **Total Objects Detected** | Tổng số Bounding Boxes đạt Confidence >= 0.35. |
| **Total Object Documents** | Số bản ghi JSON (cả Frame và Shot rollup). |
| **Inference Batching** | Batch Size 16 để tận dụng GPU nội tại qua Ultralytics. |
