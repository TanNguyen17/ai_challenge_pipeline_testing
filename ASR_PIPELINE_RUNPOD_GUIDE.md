# 🏆 SOTA Vietnamese ASR Pipeline & RunPod Deployment Guide
> **Dự án**: AI Challenge HCMC (Video Retrieval Engine)  
> **Chuyên mục**: Automatic Speech Recognition (ASR), Audio VAD, Phoneme Alignment & Ground Truth Evaluation via **`uv`**

---

## 📌 1. TỔNG QUAN KIẾN TRÚC PIPELINE (5-STAGE ARCHITECTURE)

Hệ thống ASR tiếng Việt được xây dựng dựa trên mô hình **PhoWhisper-large (VinAI)** kết hợp với hạ tầng **WhisperX Alignment & CTranslate2 Engine**. Kiến trúc này giải quyết dứt điểm các bẫy dữ liệu: *lặp từ ảo giác do nhạc nền (Hallucination), sai tên riêng/địa danh, thiếu dấu câu/chữ số, và lệch mốc thời gian với khung hình*.

```
[VIDEO AUDIO TRACK (MP4 / WAV)]
       │
       ▼
[STAGE 1: Silero VAD (Voice Activity Detection)]
       └── Lọc sạch 100% nhạc nền/im lặng -> Chống lặp từ ảo giác & giảm 30% thời lượng Audio
       │
       ▼
[STAGE 2: PhoWhisper-large (VinAI) + CTranslate2 Engine]
       ├── Model Backbone: vinai/PhoWhisper-large (VinAI fine-tune trên 844h tiếng Việt 3 miền)
       └── Engine: faster-whisper (CTranslate2 FP16) -> Tốc độ 20x-25x Real-time
       │
       ▼
[STAGE 3: WhisperX Phoneme Alignment (Wav2Vec2 Alignment)]
       └── Căn chỉnh mốc thời gian từng TỪ (Word-level timestamps) chính xác cấp millisecond
       │
       ▼
[STAGE 4: Inverse Text Normalization (ITN) & Hot-words]
       ├── Hot-words Prompt: Nạp địa danh/tên riêng (Cần Giờ, Tân Bình, HTV, CSGT, PCCC...)
       └── ITN Normalization: "chín triệu" -> "9 triệu", "mười chín tháng bảy" -> "19/07"
       │
       ▼
[STAGE 5: Temporal Sliding Window (20s) & Shot-Mapping]
       └── Gom ASR theo Cửa sổ trượt 20s (overlap 5s) & Map khớp với Shot ID từ TransNetV2
```

---

## 📋 2. INPUT / OUTPUT CONTRACT CHO TỪNG STAGE

### Stage 1: Silero VAD (Voice Activity Detection)
- **Input**: File âm thanh nguyên bản trích xuất từ Video.
- **Output**:
```json
[
  {"speech_start": 4.2, "speech_end": 18.5},
  {"speech_start": 21.0, "speech_end": 45.2}
]
```

### Stage 2 & 3: PhoWhisper + WhisperX Word Alignment
- **Input**: Audio segments đã lọc VAD.
- **Output**:
```json
{
  "transcript": "Tổ chức lễ đón vị khách du lịch thứ 9 triệu đến Nha Trang Khánh Hòa",
  "word_timestamps": [
    {"word": "Tổ", "start": 4.20, "end": 4.35},
    {"word": "chức", "start": 4.36, "end": 4.50},
    {"word": "lễ", "start": 4.51, "end": 4.65},
    {"word": "đón", "start": 4.66, "end": 4.85},
    {"word": "9 triệu", "start": 5.10, "end": 5.60}
  ],
  "avg_confidence": 0.95
}
```

### Stage 4 & 5: ITN Normalization & Elasticsearch Document (Lưu Database)
- **Input**: Word timestamps + Normalized text.
- **Output (Final Database Document)**:
```json
{
  "video_id": "L21_V001",
  "window_id": 2,
  "time_range": {"start_sec": 20.0, "end_sec": 40.0},
  "mapped_shot_ids": [1, 2],

  "asr_data": {
    "transcript_normalized": "Tổ chức lễ đón vị khách du lịch thứ 19 triệu đến Nha Trang Khánh Hòa ngày 19/07",
    "transcript_raw": "tổ chức lễ đón vị khách du lịch thứ mười chín triệu đến nha trang khánh hòa ngày mười chín tháng bảy",
    "asr_no_accent": "To chuc le don vi khach du lich thu 19 trieu den Nha Trang Khanh Hoa ngay 19/07"
  },

  "asr_stats": {
    "language": "vi",
    "confidence_score": 0.95,
    "vad_speech_ratio": 0.85
  }
}
```

---

## 📊 3. THÔNG SỐ THỰC NGHIỆM KHOA HỌC (EMPIRICAL BENCHMARKS)

### Tỉ lệ lỗi từ (Word Error Rate - WER) cho Tiếng Việt:
| Model ASR | Bộ dữ liệu VIVOS (WER) | Bộ dữ liệu VLSP 2020 (WER) | Đánh giá |
| :--- | :--- | :--- | :--- |
| OpenAI Whisper-large-v3 | 12.5% | 22.4% | Hay sai chính tả & mất dấu tiếng Việt |
| PhoWhisper-small | 6.33% | 15.93% | Rất nhẹ, chạy siêu nhanh |
| 🏆 **PhoWhisper-large (VinAI)** | **4.67%** | **13.75%** | **SOTA Tiếng Việt Số 1** (Giảm >50% lỗi) |

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

# 3. Chạy Pipeline ASR 5-Stage (PhoWhisper + WhisperX)
uv run python pipelines/run_asr_pipeline.py --video-dir ./data/extracted --output-dir ./data/processed/asr --limit 500

# 4. Chạy Đánh giá Ground Truth Evaluation & A/B Testing
uv run python eval/evaluate_benchmark.py --processed-dir ./data/processed --query-dir ./data/raw/query
```

---

## 📈 5. KHUNG ĐO ĐẠC HIỆU NĂNG & ĐÁNH GIÁ GROUND TRUTH (GT EVALUATION)

### 5.1 Chỉ số Đo đạc Từng Stage
| Stage | Chỉ số cần ghi Log (Metric) | Công thức / Đo bằng | Giá trị Kỳ vọng |
| :--- | :--- | :--- | :--- |
| **Stage 1 (Silero VAD)** | VAD Speech Ratio (%) & Noise Filter | $\frac{\text{Speech Duration}}{\text{Total Audio Duration}}$ | **25%–35%** audio noise trimmed |
| **Stage 2 (PhoWhisper)** | Inference Speed & VRAM | `time.time()` & `torch.cuda.max_memory()` | **20x Real-time**, VRAM $< 4.5\text{GB}$ |
| **Stage 3 (WhisperX)** | Word Timestamp Error (ms) | Ground Truth Alignment Offset | **$< 100\text{ms}$** word precision |
| **Stage 4 (ITN)** | Number/Date Normalization % | Matches on regex entities | **$> 95\%$** entities normalized |
| **Stage 5 (Sliding Window)**| KIS Voice Search Recall | Test 20 Spoken KIS Queries | **$> 92\%$ Recall** |

### 5.2 Đánh giá Thực tế bằng Bộ Đề Thi Ground Truth (A/B Testing)
Để đo đạc độ hiệu quả thực tế trên đề thi cuộc thi:
1. **Recall@K (K=1, 5, 10)**: So sánh kết quả tìm kiếm tiếng nói ASR với khoảng khung hình Ground Truth `[start_frame, end_frame]`.
2. **MRR (Mean Reciprocal Rank)**: Đo vị trí xuất hiện của câu thoại chính xác trong bảng xếp hạng.
3. **A/B Testing Delta**: So sánh **Kịch bản A (Chưa có ASR)** vs **Kịch bản B (Đã bật PhoWhisper SOTA)** trên 50 câu hỏi truy vấn tiếng nói để đo mức tăng trưởng điểm **Recall@5** (Kỳ vọng tăng **+30% - +40%**).
