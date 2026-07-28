# 🏆 SOTA Object Detection Pipeline & RunPod Deployment Guide
> **Dự án**: AI Challenge HCMC (Video Retrieval Engine)  
> **Chuyên mục**: Open-Vocabulary Object Detection, Spatial UI Masking, ROI Crop Vectoring & Ground Truth Evaluation via **`uv`**

---

## 📌 1. TỔNG QUAN KIẾN TRÚC PIPELINE (5-STAGE ARCHITECTURE)

Hệ thống Object Detection được xây dựng theo kiến trúc **Two-Tier SOTA (Offline Open-Vocab Detection + Online Vision-Language Grounding)**. Kiến trúc này giải quyết dứt điểm các bẫy dữ liệu: *hạn chế 80 lớp COCO chuẩn, nhận diện nhầm vật thể trên Logo/Quảng cáo TV, phình dữ liệu qua cảnh tĩnh, và chi phí tính toán GPU*.

```
[KEYFRAME TỪ TRANSNETV2]
       │
       ▼
[STAGE 1: Spatial UI Exclusion Masking]
       └── Che (Mask) góc trên Logo & dải Tin chạy -> Triệt tiêu 95% nhận diện nhầm trên đồ họa TV
       │
       ▼
[STAGE 2: YOLO-World v2 + RAM++ Multi-Label Tagging]
       ├── YOLO-World v2: Dùng cơ chế "Prompt-then-Detect" -> Nhận diện Open-Vocab siêu nhanh (74 FPS)
       └── RAM++: Tự động gán nhãn hàng nghìn khái niệm cảnh quan (Stage, River, Crowd, Daytime...)
       │
       ▼
[STAGE 3: Crop ROI & CLIP Crop Vector Encoding]
       └── Cắt ảnh vùng Bounding Box (ROI) -> Mã hóa thành 512d CLIP Vector (Phục vụ VQA thuộc tính)
       │
       ▼
[STAGE 4: Shot-Level Object Summarization]
       └── Thống kê vật thể theo Shot Cut -> Lưu max_count, spatial_position (Nén 90% DB size)
       │
       ▼ (ONLINE SEARCH PHASE)
[STAGE 5: Microsoft Florence-2 / Grounding DINO 1.5 Reranker]
       └── Mô hình VLM siêu nhẹ của Microsoft (0.23B params) -> Soi lại vị trí không gian cho Top-20 frames VQA
```

---

## 📋 2. INPUT / OUTPUT CONTRACT CHO TỪNG STAGE

### Stage 1: Spatial UI Exclusion Masking
- **Input**: Raw Keyframe Image (.webp / .jpg).
- **Output**: Masked image with excluded top-right logo and bottom ticker zones.

### Stage 2 & 3: YOLO-World v2 + RAM++ Tagging & Crop ROI Vectoring
- **Input**: Masked Keyframe image.
- **Output**:
```json
{
  "keyframe_id": 400,
  "timestamp_sec": 16.0,
  "yolo_world_detections": [
    {
      "object_id": "OBJ_01",
      "label": "red_aodai",
      "bbox": [0.25, 0.30, 0.55, 0.85],
      "confidence": 0.92,
      "spatial_position": "center_left",
      "roi_crop_vector": [0.082, -0.015, 0.241, 0.115]
    },
    {
      "object_id": "OBJ_02",
      "label": "dan_bau",
      "bbox": [0.40, 0.60, 0.70, 0.90],
      "confidence": 0.88,
      "spatial_position": "center_bottom",
      "roi_crop_vector": [-0.104, 0.052, 0.183, 0.091]
    }
  ],
  "ram_concept_tags": ["stage", "indoor", "performance", "traditional_music"]
}
```

### Stage 4 & 5: Shot-Level Summarization & Elasticsearch Document (Lưu Database)
- **Input**: Detected objects + ROI vectors per keyframe.
- **Output (Final Database Document)**:
```json
{
  "video_id": "L21_V001",
  "shot_id": 2,
  "time_range": {"start_sec": 15.0, "end_sec": 28.0},
  "keyframe_ids": [400, 420],

  "object_summary": {
    "detected_classes": ["person", "red_aodai", "car", "dan_bau"],
    "counts": {
      "person": 2,
      "red_aodai": 1,
      "car": 3,
      "dan_bau": 1
    },
    "objects_detail": [
      {
        "object_id": "OBJ_01",
        "label": "red_aodai",
        "confidence": 0.92,
        "bbox": [0.25, 0.30, 0.55, 0.85],
        "spatial_position": "center_left"
      }
    ]
  },

  "scene_tags": ["stage", "indoor", "performance", "traditional_music"]
}
```

---

## 📊 3. SO SÁNH HIỆU NĂNG MÔ HÌNH (SOTA MODEL BENCHMARKS)

| Mô hình (Model) | Loại Kiến trúc | Tốc độ Inference | Vai trò trong Pipeline |
| :--- | :--- | :--- | :--- |
| **YOLOv8 / YOLOv11** | Fixed 80-Class COCO | ~5ms / frame | ❌ Không dùng (Bị giới hạn 80 lớp cố định) |
| 🏆 **YOLO-World v2** | **Open-Vocab Real-time** | **~13-15ms / frame (74 FPS)** | 🟩 **SOTA Số 1 cho Offline Batching** |
| 🏆 **RAM++** | Multi-label Concept Tagging | ~20ms / frame | 🟩 **SOTA cho Gán nhãn toàn cảnh** |
| 🏆 **Florence-2 (Microsoft)** | Vision-Language Foundation | ~40-60ms / frame | 🟩 **SOTA Số 1 cho Online VQA Rerank** |

---

## 🚀 4. HƯỚNG DẪN TRIỂN KHAI THỰC NGHIỆM TRÊN RUNPOD VỚI `uv`

### 4.1 Cài đặt & Đồng bộ Môi trường với `uv`
```bash
pip install uv
uv sync
```

### 4.2 Kịch bản Chạy Lệnh Thực Nghiệm bằng `uv run`

```bash
# 1. Tải dữ liệu Benchmark (~650 videos, ~7.4GB) từ HuggingFace
uv run python download_data.py --phase benchmark

# 2. Giải nén dữ liệu video
uv run python extract_data.py --raw-dir ./data/raw --extract-dir ./data/extracted

# 3. Chạy Pipeline Object Detection 5-Stage (YOLO-World v2 + RAM++ + Florence-2)
uv run python pipelines/run_obj_detection_pipeline.py --video-dir ./data/extracted --output-dir ./data/processed/objects --limit 500

# 4. Chạy Đánh giá Ground Truth Evaluation & A/B Testing
uv run python eval/evaluate_benchmark.py --processed-dir ./data/processed --query-dir ./data/raw/query
```

---

## 📈 5. KHUNG ĐO ĐẠC HIỆU NĂNG & ĐÁNH GIÁ GROUND TRUTH (GT EVALUATION)

### 5.1 Chỉ số Đo đạc Từng Stage
| Stage | Chỉ số cần ghi Log (Metric) | Công thức / Đo bằng | Giá trị Kỳ vọng |
| :--- | :--- | :--- | :--- |
| **Stage 1 (UI Masking)** | UI False Positive Filter (%) | $1 - \frac{\text{UI Boxes}}{\text{Raw Boxes}}$ | **$> 95\%$** UI noise eliminated |
| **Stage 2 (YOLO-World v2)** | Throughput & Latency | FPS & `time.time()` | **50–70 FPS**, Latency $< 20\text{ms}$ |
| **Stage 3 (Crop ROI)** | Crop Vector Extraction Time | `time.time()` per crop | **$< 5\text{ms}$** per crop |
| **Stage 4 (Summarization)** | Shot Compression Ratio | $\frac{\text{Raw Boxes}}{\text{Shot Summarized Records}}$ | **8x – 12x** DB compression |
| **Stage 5 (Reranking)** | Top-5 VQA Spatial Recall | Test 20 Object VQA Queries | **$> 90\%$ Recall@5** |

### 5.2 Đánh giá Thực tế bằng Bộ Đề Thi Ground Truth (A/B Testing)
Để đo đạc độ hiệu quả thực tế trên đề thi cuộc thi:
1. **Recall@K (K=1, 5, 10)**: So sánh kết quả tìm kiếm vật thể với khoảng khung hình Ground Truth `[start_frame, end_frame]`.
2. **MRR (Mean Reciprocal Rank)**: Đo vị trí xếp hạng trung bình của đáp án đúng.
3. **A/B Testing Delta**: So sánh **Kịch bản A (Chưa có Object Detection)** vs **Kịch bản B (Đã bật YOLO-World v2 + Florence-2)** trên 50 câu hỏi truy vấn vật thể/đếm/vị trí không gian để đo mức tăng trưởng điểm **Recall@5** (Kỳ vọng tăng **+25% - +35%**).
