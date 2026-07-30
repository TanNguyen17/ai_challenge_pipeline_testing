# 🏆 SOTA Vietnamese ASR Pipeline & RunPod Deployment Guide
> **Dự án**: AI Challenge HCMC (Video Retrieval Engine)  
> **Chuyên mục**: Automatic Speech Recognition (ASR), VAD Filtering & Ground Truth Evaluation via **`uv`**

---

## 📌 1. TỔNG QUAN KIẾN TRÚC PIPELINE

Hệ thống ASR tiếng Việt được xây dựng dựa trên mô hình **PhoWhisper-large (VinAI)** kết hợp với engine **Faster-Whisper (CTranslate2)** để tăng tối đa tốc độ inference thực nghiệm trên GPU.

```
[VIDEO AUDIO TRACK (MP4 / WAV)]
       │
       ▼
[STAGE 1: Audio Extraction]
       └── Trích xuất audio 16kHz mono bằng `ffmpeg`. Xử lý fail-fast nếu không có file âm thanh.
       │
       ▼
[STAGE 2: Faster-Whisper (PhoWhisper) & Silero VAD]
       ├── Đọc model `vinai/PhoWhisper-large` định dạng CT2.
       ├── Lọc im lặng tự động (min_silence_duration=500ms) bằng VAD filter.
       └── Nạp Hot-word từ metadata `media-info` JSON của BTC.
       │
       ▼
[STAGE 3: Hallucination Filtering]
       └── Loại bỏ các lỗi lặp từ (hallucination loops: lặp 4 lần liên tiếp) do nhiễu nhạc nền.
       │
       ▼
[STAGE 4: Temporal Mapping & Export]
       └── Dùng thời gian đoạn VAD làm `window_id`, map với danh sách `frame_indices` và `shot_ids` từ BTC keyframes.
```

---

## 📋 2. INPUT / OUTPUT CONTRACT

### Final Database Document (JSONL)

Dữ liệu được lưu dưới dạng file `.jsonl` tăng dần. Cấu trúc output thực tế từ code hiện tại:

```json
{
  "video_id": "L21_V001",
  "window_id": 0,
  "time_range": {"start_sec": 4.43, "end_sec": 38.06},
  "mapped_frame_indices": [261, 351, 411, 531],
  "mapped_shot_ids": [3, 4, 5, 6],
  "asr_data": {
    "transcript_normalized": "chào mừng quý vị đến với chương trình sáu mươi giây",
    "transcript_raw": "chào mừng quý vị đến với chương trình sáu mươi giây",
    "asr_no_accent": "chao mung quy vi den voi chuong trinh sau muoi giay"
  },
  "word_timestamps": [
    {"word": " chào", "start": 4.43, "end": 4.95, "probability": 0.5},
    {"word": " mừng", "start": 4.95, "end": 5.05, "probability": 1.0}
  ],
  "confidence_score": 0.966
}
```

---

## 🚀 3. HƯỚNG DẪN TRIỂN KHAI THỰC NGHIỆM TRÊN RUNPOD VỚI `uv`

### 3.1 Cài đặt Môi trường
```bash
uv pip install faster-whisper
```

### 3.2 Kịch bản Chạy Lệnh Thực Nghiệm
```bash
# Chạy Pipeline ASR
uv run python pipelines/run_asr_pipeline.py \
    --video-dir ./data/extracted \
    --output-dir ./data/processed/asr \
    --keyframes-dir ./data/extracted/map-keyframes \
    --media-info-dir ./data/extracted/media-info \
    --limit 50
```

---

## 📊 4. KHUNG ĐO ĐẠC HIỆU NĂNG

| Metric | Mô tả | 
| :--- | :--- | 
| **Total VAD Segments** | Số lượng đoạn thoại trích xuất được sau khi chạy Silero VAD (Loại bỏ im lặng/nhạc). |
| **ASR Documents** | Tổng số doc ghi nhận vào database (Loại bỏ các doc lặp do Hallucination). |
| **Inference Time** | Thời gian thực thi trung bình trên mỗi video bằng CTranslate2. |
