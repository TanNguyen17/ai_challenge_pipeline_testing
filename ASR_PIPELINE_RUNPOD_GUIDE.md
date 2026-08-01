# 🏆 SOTA Video ASR Pipeline (Qwen3-ASR Edition) & RunPod Deployment Guide
> **Dự án**: AI Challenge HCMC (Video Retrieval Engine)  
> **Chuyên mục**: Qwen3-ASR-1.7B, WhisperX VAD, Word-Level Alignment, Two-Tier JSON Indexing

---

## 📌 1. TỔNG QUAN KIẾN TRÚC PIPELINE VÀ JUSTIFICATION (TẠI SAO DÙNG QWEN3-ASR?)

### 💡 Justification: Tại sao dùng Qwen3-ASR thay vì bản Qwen3-Audio?
Mặc dù Qwen3-Audio-7B là mô hình rất mạnh nhưng nó là một Audio-LLM (nghĩ giống như ChatGPT biết nghe). Việc dùng nó chỉ để "chép chính tả" (Transcription) là quá lãng phí tài nguyên và làm chậm pipeline. Ngược lại, **Qwen3-ASR-1.7B** là mô hình chuyên dụng cho chép lời (ASR Encoder-Decoder):
1. **Tối ưu Hóa Tốc độ & VRAM:** Với 1.7 tỷ tham số, nó chạy nhanh gấp nhiều lần và ngốn chưa tới 5GB VRAM, cho phép chạy High-throughput cực đỉnh.
2. **Loại bỏ Ảo giác (Hallucinations):** Vẫn kế thừa khả năng kháng nhiễu và hiểu tiếng Việt của dòng Qwen, khắc phục triệt để lỗi "lặp từ vô tận" của Whisper/PhoWhisper.
3. **Cấu trúc linh hoạt với WhisperX:** Qwen3-ASR dịch ra text thô siêu chuẩn, sau đó ta kết hợp VAD (cắt đoạn) và Phoneme Alignment của WhisperX để tạo ra một Pipeline Hybrid hoàn hảo: Văn bản chuẩn của Qwen3-ASR + Mốc thời gian chuẩn từng miligiây của WhisperX.

### 🏗️ Kiến trúc Pipeline Cải tiến (4-Stage Architecture)

```text
[VIDEO GỐC MP4] ──(Tách Audio WAV 16kHz)──┐
                                          │
                                          ▼
[STAGE 1: WhisperX VAD (Chunking)]
       └── Chạy mô hình VAD lọc bỏ khoảng lặng/nhạc nền, cắt âm thanh thành các đoạn <30s.
                                          │
                                          ▼
[STAGE 2: Qwen3-ASR-1.7B (High-throughput Specialized ASR)]
       └── Đọc hiểu âm thanh và trích xuất thành văn bản tiếng Việt hoàn chỉnh, triệt tiêu ảo giác.
                                          │
                                          ▼
[STAGE 3: WhisperX Phoneme/Word Alignment]
       └── Đối chiếu đoạn text từ Qwen3-ASR ngược lại với âm thanh bằng Wav2Vec2 để lấy mốc thời gian.
                                          │
                                          ▼
[STAGE 4: Two-Tier DB Schema Export]
       └── Lưu vào JSONL (1 file cho `asr_span` và 1 file cho `shot` rollup).
```

---

## 🔍 2. CHI TIẾT TRIỂN KHAI TỪNG STAGE

Việc trích xuất giọng nói cho Video Retrieval đòi hỏi phải khớp nối hoàn hảo với dữ liệu hình ảnh (Visual). Dưới đây là chiến lược xử lý chuyên sâu:

### STAGE 1 & 2: VAD Chunking và Qwen3-ASR
- **Kỹ thuật Cắt Âm Thanh:** Qwen3-ASR hoạt động cực tốt với các đoạn âm thanh dưới 30 giây. Sử dụng công cụ VAD của WhisperX, âm thanh dài cả tiếng đồng hồ sẽ được băm nhỏ thành các khối (chunks) tinh khiết (chỉ có giọng người).
- **Trích xuất chuyên sâu:** Truyền thẳng Array âm thanh vào `AutoModelForCausalLM` với mô hình Qwen3-ASR. Quá trình này chuyên trị đa ngôn ngữ và tạp âm.

### STAGE 3: WhisperX Phoneme/Word Alignment
- **Cách hoạt động:** Dù Qwen3-ASR ra Text rất tốt, ta đẩy ngược văn bản đó cùng Array âm thanh qua hàm `whisperx.align(wx_segments, align_model)`.
- **Lý do & Dẫn chứng:** Mô hình Wav2Vec2 của WhisperX sẽ đo đạc lại sóng âm (Acoustic) và gán chính xác `{"word": "tôi", "start": 1.25, "end": 1.40}`. Mốc thời gian chuẩn là yếu tố cốt lõi để ăn trọn điểm tIoU (không dư không thiếu) trong bài thi Known-Item Search (KIS), BGK yêu cầu tìm đúng 1 giây mà nhân vật nói ra từ khóa (VD: *"Khoảnh khắc MC nói từ 'Festival'"*). Việc có timestamp cho từng từ giúp hệ thống cắt đúng chính xác 1 giây đó để nộp, đảm bảo ăn trọn điểm tIoU (Temporal Intersection over Union).

### STAGE 4: DB Export & Shot Mapping (Chiến lược Multimodal)
- **Cách hoạt động:** Dữ liệu ASR không đứng một mình. Hệ thống lấy thời gian xuất hiện của câu nói so khớp với ranh giới Cảnh quay (Shot Boundaries) từ TransNetV2 (đã làm ở pipeline OCR). Câu nói thuộc giây nào sẽ được "gắn" (map) vào Shot ID của giây đó. Mọi dữ liệu xuất ra JSONL cho Elasticsearch.

---

## 💡 3. HƯỚNG DẪN TỐI ƯU CHIẾN LƯỢC QUERY & TIOU

Tương tự như OCR, ASR cũng phải tuân thủ triệt để mô hình **Two-Tier Schema (Coarse-to-Fine)** để đáp ứng 2 dạng câu hỏi AVS và KIS:

**Ví dụ thực tiễn (Dựa trên JSON bên dưới):**
Giả sử BGK ra đề đa phương thức (Multimodal): *"Tìm cảnh quay có Logo HTV9 (Hình ảnh) VÀ MC đọc từ 'Khánh Hòa' (Âm thanh)"*.
- **Bước 1 (Tìm kiếm AVS bằng Shot - Maximize Recall):** Nhờ cơ chế Shot Mapping ở Stage 4, chữ *"Khánh Hòa"* đã được hệ thống tự động gộp vào chung một rổ `doc_type: "shot"` (Shot ID: 26) cùng với dữ liệu hình ảnh Logo HTV9. Elasticsearch dễ dàng truy vấn chéo (Cross-modal) trên cùng một file JSON và tìm ra ngay Shot 26!
- **Bước 2 (Chốt thời gian KIS bằng Span - Maximize tIoU):** Dù Shot 26 dài 10 giây, nhưng MC chỉ đọc từ *"Khánh Hòa"* trong đúng 0.5 giây. Hệ thống code của bạn sẽ lật ngược về file `doc_type: "asr_span"`, nhìn vào mảng `word_timestamps` của chữ "Khánh Hòa" để trích xuất exaclty mốc `4.51s` đến `4.85s` nộp cho BTC. Điểm tIoU tuyệt đối!

---

## 📋 4. FINAL DATABASE DOCUMENTS (JSONL CONTRACT)

**1. ASR Span Document (Phục vụ truy vấn chính xác Word-level)**
```json
{
  "doc_type": "asr_span",
  "video_id": "L21_V001",
  "shot_id": 26,
  "time_range": {"start_sec": 4.20, "end_sec": 5.60},
  "asr_data": {
    "transcript": "lễ đón khách đến Khánh Hòa",
    "word_timestamps": [
      {"word": "lễ", "start": 4.20, "end": 4.35},
      {"word": "đón", "start": 4.36, "end": 4.50},
      {"word": "Khánh", "start": 4.51, "end": 4.70},
      {"word": "Hòa", "start": 4.71, "end": 4.85}
    ]
  },
  "confidence_score": 0.95
}
```

**2. Multimodal Shot Document (Phục vụ truy vấn AVS BM25 chéo)**
```json
{
  "doc_type": "shot",
  "video_id": "L21_V001",
  "shot_id": 26,
  "time_range": {"start_sec": 0.0, "end_sec": 10.0},
  "combined_data": {
    "asr_transcript": "tổ chức lễ đón khách đến khánh hòa",
    "ocr_overlay_text": "HTV9 Tin Tức",
    "object_tags": ["person", "microphone"]
  }
}
```

---

## 🚀 5. HƯỚNG DẪN TRIỂN KHAI THỰC NGHIỆM TRÊN RUNPOD VỚI `uv`

### 5.1 Cài đặt & Đồng bộ Môi trường
Cần thư viện whisperx và các dependencies cho Qwen2-Audio:
```bash
uv pip install transformers torch accelerate whisperx
```

### 5.2 Kịch bản Chạy Lệnh Thực Nghiệm
```bash
# 2. Chạy Pipeline ASR (Qwen3-ASR + WhisperX)
uv run python pipelines/run_asr_pipeline.py \
    --video-dir ./data/extracted \
    --output-dir ./data/processed/asr \
    --model-name Qwen/Qwen3-ASR-1.7B \
    --apply-vad true \
    --limit 500

# 2. Chạy kịch bản Shot-Mapping (Hợp nhất ASR vào Shot ID)
uv run python pipelines/merge_multimodal.py \
    --asr-dir ./data/processed/asr \
    --ocr-dir ./data/processed/ocr \
    --output-dir ./data/processed/database
```

---

## 📊 6. KHUNG ĐO ĐẠC HIỆU NĂNG & A/B TESTING

| Tiêu chí thi đấu AI Challenge | Kịch bản A (Whisper V3 Gốc) | Kịch bản B (Qwen3-ASR + WhisperX) |
| :--- | :--- | :--- |
| **Xử lý tiếng Việt Thời sự** | Lỗi dấu, sai từ địa phương | 🟢 SOTA Audio LLM, hiểu cả ngữ cảnh |
| **Ảo giác (Hallucination)** | Bị lặp từ vô tận khi có khoảng lặng | 🟢 Triệt tiêu hoàn toàn nhờ Chat Prompt |
| **Mốc thời gian (tIoU)** | Cấp độ câu (10s), dễ mất điểm | 🟢 Phoneme Alignment cấp độ từ (0.1s) |
| **Khả năng Truy vấn Cross-modal** | Âm thanh lệch pha Hình ảnh | 🟢 Khớp hoàn hảo nhờ Shot Mapping Database |
