# 🏆 SOTA Video OCR Pipeline (VLM Edition) & RunPod Deployment Guide
> **Dự án**: AI Challenge HCMC (Video Retrieval Engine)  
> **Chuyên mục**: Vision-Language Model (VLM) Text Extraction, UI Masking & Structured Document Indexing

---

## 📌 1. TỔNG QUAN KIẾN TRÚC PIPELINE VÀ JUSTIFICATION (TẠI SAO DÙNG QWEN3-VL?)

### 💡 Justification: Tại sao thay thế PaddleOCR bằng Qwen3-VL?
Theo các nghiên cứu mới nhất về **Video Document Understanding** (như top đầu benchmark trên [DocVQA](https://www.docvqa.org/) và [OCRBench](https://github.com/Yuliang-Liu/MultimodalOCR)), các mô hình VLM như **Qwen3-VL** đã vượt qua các pipeline OCR truyền thống nhờ khả năng **đọc hiểu ngữ cảnh (Semantic Comprehension)**. Hơn nữa, tập dữ liệu của AI Challenge chứa lượng lớn **Video Thời sự (HTV9) với chữ chạy (News ticker) và bảng hiệu tiếng Việt phức tạp**. Qwen3-VL là mô hình SOTA hỗ trợ Zero-shot tiếng Việt vượt trội so với PaddleOCR (vốn hay bị lỗi dấu ở phông chữ khó):
1. **Semantic Grouping (Gom nhóm ngữ nghĩa):** PaddleOCR trả về một "rổ chữ" lộn xộn (bag-of-words). Qwen3-VL có khả năng phân biệt đâu là dòng tiêu đề thời sự, đâu là chữ ngẫu nhiên trên áo nhân vật.
2. **End-to-End Pipeline:** Loại bỏ hoàn toàn sự phức tạp của ByteTrack (Tracking) và Layout Classifier. VLM xử lý trực tiếp từ ảnh sang JSON có cấu trúc.
3. **Khắc phục điểm yếu (Hallucinations):** Quá trình phân tích log cho thấy Qwen3-VL thường bị "ảo giác" lặp từ (VD: lặp lại mốc thời gian `06:31:37` hàng chục lần). Để đưa pipeline này vào thi đấu thực tế, kiến trúc bên dưới đã được **cải tiến đặc biệt** bằng kỹ thuật *Spatial UI Masking* và *Structured Prompting*.

### 🏗️ Kiến trúc Pipeline Cải tiến (3-Stage Architecture)

```
[VIDEO GỐC MP4]
       │
       ▼
[STAGE 1: TransNetV2 & Sparse Multi-Frame Sampling]
       └── Xác định ranh giới cảnh và trích xuất mẫu thưa (2-3 frames/shot).
       │
       ▼
[STAGE 2: VLM Structured Prompting & VQA]
       └── Sử dụng `Qwen3-VL-7B-Instruct` đọc hiểu văn bản và ép xuất JSON theo format.
       │
       ▼
[STAGE 3: DB Export & Indexing (Two-Tier Schema)]
       └── Lưu mốc thời gian chi tiết (`span`) và gộp toàn bộ ngữ cảnh (`shot`).
```

---

## 🔍 2. CHI TIẾT TRIỂN KHAI TỪNG STAGE

Việc vận hành một VLM khổng lồ trên lượng dữ liệu video đa dạng đòi hỏi sự linh hoạt và tối ưu khắt khe ở từng bước. Dưới đây là cơ chế xử lý chuyên sâu:

### STAGE 1: TransNetV2 & Sparse Multi-Frame Sampling
- **Cách hoạt động:** Pipeline không sử dụng file keyframe tĩnh `1-frame/shot` của BTC. Thay vào đó, video được đưa qua mô hình Deep Learning `TransNetV2` để phân tích động lực học của từng pixel, từ đó nội suy ra chính xác mốc `start_frame` và `end_frame` của mỗi phân cảnh.
- **Cơ chế lấy mẫu (Sparse Sampling):** Dựa trên thời lượng của shot, hệ thống tự động trích xuất **2 đến 3 khung hình** cách đều nhau. 
- **Lý do & Dẫn chứng:** Theo báo cáo nghiên cứu SOTA của [Qwen-VL (arXiv:2308.12966)](https://arxiv.org/abs/2308.12966) và [Video-LLaVA (arXiv:2311.10122)](https://arxiv.org/abs/2311.10122), chiến lược lấy mẫu thưa (Sparse Sampling) giải quyết bài toán cốt lõi của VLM: Chi phí tính toán cực cao (có thể tốn 300ms/frame). Việc chỉ lấy 2-3 frames cân bằng hoàn hảo giữa tốc độ và độ bao phủ thông tin. Ngược lại, nếu chỉ dựa vào 1 frame ngẫu nhiên từ BTC, hệ thống vô cùng dễ trúng vào flash-frame (cảnh chớp sáng) hoặc frame mờ, dẫn đến việc VLM từ chối phục vụ (báo lỗi *"Sorry, I can't assist"*). Ngoài ra, nhiều frame giúp tóm gọn được chữ cuộn (scrolling ticker) từ đầu đến cuối shot.

### STAGE 2: VLM Structured Prompting & VQA
- **Cách hoạt động:** Ta trực tiếp đưa ảnh gốc vào `Qwen3-VL-7B-Instruct` mà không dùng kỹ thuật vẽ box đen (UI Masking) cứng nhắc, vốn dễ làm mất chữ quan trọng khi chạy trên các video Vlog/Phim không có logo đài.
- **Giải quyết Ảo giác (Hallucination):** Dùng kỹ thuật **Zero-shot Prompt Engineering** ép khuôn JSON đầu ra: *"Hãy phân loại chữ thành 'overlay_text' và 'scene_text'. ĐẶC BIỆT PHỚT LỜ các logo nhỏ, đồng hồ, và mốc thời gian. Chỉ trả về JSON."*
- **Lý do & Dẫn chứng (Tối ưu Truy vấn KIS):** Việc yêu cầu VLM phân loại rõ `'overlay_text'` (chữ đồ họa chèn thêm) và `'scene_text'` (chữ tự nhiên trong cảnh) là vũ khí cực mạnh cho AI Challenge. Ví dụ, nếu đề thi KIS yêu cầu *"Tìm bảng hiệu tiệm bánh mì"*, Elasticsearch của bạn có thể tăng trọng số (boost weight) cho trường `'scene_text'` và phớt lờ các tin thời sự chạy bên dưới (`'overlay_text'`). Hơn nữa, ép định dạng JSON triệt tiêu hoàn toàn rác hội thoại của AI (như *"Theo hình ảnh..."*), giữ Database luôn sạch sẽ.

### STAGE 3: DB Export & Indexing (Two-Tier Schema)
- **Cách hoạt động:** Kết quả JSON từ Stage 2 được đối chiếu lại với danh sách thời gian. Mỗi frame sinh ra một document mức `span`. Sau đó, hệ thống gom tất cả chữ của các `span` trong cùng một shot, áp dụng thuật toán lọc trùng lặp, tạo thành một document mức `shot`. Mọi dữ liệu xuất ra JSONL cho Elasticsearch.
- **Lý do & Dẫn chứng (Đặc thù Cuộc thi & Dataset):** Trong các giải đấu như [VBS](https://videobrowsershowdown.org/), dữ liệu thường là video không cấu trúc (News, CCTV). Đề thi luôn chia làm 2 dạng: 
  1) **AVS (Ad-hoc Video Search):** Tìm ngữ cảnh chung chung (VD: *"Người biểu tình mang biểu ngữ ngã ra đường"*). Document `shot` là bắt buộc vì nó gom đủ mọi "từ khóa" rải rác trong một cảnh dài để Elasticsearch dễ dàng BM25 matching (Maximize Recall).
  2) **KIS (Known-Item Search):** Tìm chính xác một khoảnh khắc chớp nhoáng (VD: *"Khoảnh khắc biển số xe 63-AM xẹt qua"*). Document `span` là vũ khí tối thượng cung cấp mốc thời gian chuẩn tới từng miligiây để nộp cho BTC, tránh bị phạt điểm tIoU (Maximize Precision). Lược đồ Two-Tier này là thiết kế tiêu chuẩn của các đội vô địch VBS/TRECVID.

---

## 📋 3. INPUT / OUTPUT CONTRACT & PROMPT ENGINEERING

### Kỹ thuật Structured Prompting cho Qwen3-VL
Để tránh VLM mô tả miên man, Prompt phải được thiết kế để ép ra JSON:
> *"Hãy trích xuất tất cả văn bản tiếng Việt xuất hiện trong bức ảnh này. Phân loại chúng thành hai nhóm: 'overlay_text' (chữ được chèn lên video như tiêu đề, tin chạy) và 'scene_text' (chữ tự nhiên trong cảnh như biển hiệu, áo). Bỏ qua các logo nhỏ. Chỉ trả về JSON format: {"overlay_text": "...", "scene_text": "..."}. Không giải thích."*

### Final Database Documents (JSONL) & Two-Tier Query Justification

**💡 Hướng dẫn: Tại sao lại cần cả `span` và `shot`? (Chiến lược Tối ưu điểm tIoU)**

Trong các giải đấu quốc tế như [VBS (Video Browser Showdown)](https://videobrowsershowdown.org/) hay [TRECVID](https://trecvid.nist.gov/), hệ thống chấm điểm dựa trên **tIoU (Temporal Intersection over Union)**. Tức là: Bạn phải nộp đáp án chứa đúng sự kiện, và mốc thời gian nộp (start, end) phải khớp sát nhất với thời gian thực tế. Việc xuất ra cả 2 level `span` và `shot` (mô hình **Multi-Stage Temporal Grounding**) là công thức cốt lõi để chiến thắng:

**Nếu chỉ dùng 1 loại Document, bạn sẽ thất bại:**
- ❌ **Nếu chỉ dùng Frame/Span:** Giả sử đề bài tìm *"Cảnh có Logo HTV9 và chữ Bão"*. Frame 1 có Logo HTV9 (nhưng chưa có chữ Bão). Frame 50 có chữ Bão (nhưng mất Logo). Elasticsearch sẽ không bao giờ tìm ra video này vì không có frame nào chứa cả hai từ khóa cùng lúc!
- ❌ **Nếu chỉ dùng Shot:** Bằng cách gộp toàn bộ chữ của shot vào một Document, Elasticsearch sẽ tìm ra ngay phân cảnh chứa cả "HTV9" và "Bão". Tuy nhiên, nếu Shot dài 30 giây, mà chữ "Bão" chỉ xuất hiện đúng 3 giây, việc bạn nộp đáp án "0 đến 30 giây" sẽ bị BTC phạt điểm tIoU rất nặng vì quá dư thừa thời gian (độ chính xác thấp).

**✅ Giải pháp của chúng ta: Chiến lược 2 bước (Coarse-to-Fine)**
Đây là cách thiết kế schema giúp bạn vừa không bỏ sót video, vừa đạt điểm tIoU tuyệt đối:

1. **Bước 1: Tìm kiếm bao quát (Tìm bằng Shot Document - Maximize Recall)**
   Đẩy câu truy vấn vào Elasticsearch tìm trên các file `doc_type: "shot"`. Vì `shot` chứa tất cả chữ của cả một phân cảnh (như một rổ từ khóa khổng lồ), nó đảm bảo bạn **chắc chắn tìm trúng video** có chứa các yếu tố mong muốn dù chúng không xuất hiện cùng lúc trên một khung hình.
   
2. **Bước 2: Cắt gọt thời gian chính xác (Soi bằng Span Document - Maximize tIoU)**
   Sau khi Elasticsearch trả về danh sách các `shot_id` phù hợp nhất, hệ thống code của bạn sẽ lật lại Database, truy vấn vào các `doc_type: "span"` thuộc đúng những `shot_id` đó. Lúc này, bạn sẽ biết chính xác chữ "Bão" bắt đầu ở giây nào và kết thúc ở giây nào để nộp mốc thời gian cực kỳ chuẩn xác cho BTC!

---
**💡 Ví dụ thực tiễn (Dựa trên 2 Documents mẫu bên dưới):**
Giả sử BGK ra đề: *"Tìm cảnh có nói về học bổng ở Vũng Tàu và Festival hoa Đà Lạt"*.
- Nếu bạn search bằng `span` (Document 1): Bạn sẽ trượt! Vì frame 2500 chỉ hiện câu *"Vũng Tàu trao học bổng"*, thông tin về *"Festival"* chưa xuất hiện trên màn hình.
- Nếu bạn search bằng `shot` (Document 2): Elasticsearch sẽ tìm thấy `shot_id: 26` ngay lập tức vì nó đã gộp toàn bộ text của cả đoạn video dài 6 giây (từ `99.96s` đến `105.70s`).
- Tuy nhiên, nếu bạn nộp đáp án dài 6 giây này cho BTC, bạn sẽ bị trừ điểm tIoU vì chữ *"Vũng Tàu trao học bổng"* chỉ xuất hiện đúng 0.5 giây. Vì vậy, ở bước 2, bạn lấy `shot_id: 26` truy vấn ngược vào Database `span` (để ra Document 1) và nộp đáp án chính xác tuyệt đối là từ `99.96s` đến `100.46s`!

**1. Per-Frame Span Document (Định vị chính xác thời gian)**
```json
{
  "doc_type": "span",
  "video_id": "L21_V019",
  "shot_id": 26,
  "tracklet_id": "TRK_0025",
  "frame_idx": 2500,
  "time_range": {"start_sec": 99.96, "end_sec": 100.46},
  "ocr_data": {
    "overlay_text": "Vũng Tàu trao học bổng cho 48 học sinh",
    "scene_text": ""
  },
  "confidence": 0.95
}
```

**2. Per-Shot Rollup Document (Tối ưu tìm kiếm ngữ cảnh BM25)**
```json
{
  "doc_type": "shot",
  "video_id": "L21_V019",
  "shot_id": 26,
  "time_range": {"start_sec": 99.96, "end_sec": 105.70},
  "ocr_data_combined": {
    "overlay_text": "Vũng Tàu trao học bổng cho 48 học sinh. Festival hoa Đà Lạt sẽ diễn ra gần 1 tháng",
    "scene_text": "Trường tiểu học",
    "ocr_no_accent_combined": "vung tau trao hoc bong cho 48 hoc sinh festival hoa da lat se dien ra gan 1 thang truong tieu hoc"
  }
}
```

---

## 🚀 4. HƯỚNG DẪN TRIỂN KHAI THỰC NGHIỆM TRÊN RUNPOD VỚI `uv`

### 4.1 Cài đặt Môi trường
Cần cài đặt bộ thư viện `transformers` mới nhất, `qwen-vl-utils` và môi trường cho TransNetV2 (TensorFlow):
```bash
uv pip install transformers>=4.45.0 qwen-vl-utils tensorflow==2.15.0 opencv-python-headless
```

### 4.2 Kịch bản Chạy Lệnh Thực Nghiệm
```bash
# 1. (Tùy chọn) Chạy trích xuất Shot Boundaries tự động bằng TransNetV2
uv run python extract/workers/transnet.py --video ./data/raw/L21_V019.mp4 --output ./data/extracted/keyframes

# 2. Chạy Pipeline OCR bằng Qwen3-VL
uv run python pipelines/run_ocr_pipeline.py \
    --video-dir ./data/extracted \
    --output-dir ./data/processed/ocr \
    --use-qwen-vl true \
    --limit 50
```

---

## 📊 5. KHUNG ĐO ĐẠC HIỆU NĂNG & A/B TESTING

Để chứng minh VLM hiệu quả hơn OCR truyền thống trong cuộc thi này, thực hiện A/B Testing:

| Tiêu chí thi đấu AI Challenge | Kịch bản A (PaddleOCR Cũ) | Kịch bản B (Qwen3-VL Cải tiến) |
| :--- | :--- | :--- |
| **Lọc nhiễu Timestamp (Tăng Precision)** | Vẫn bị đọc (VD: `06:31:37`) | 🟢 Đã bị lọc nhờ Zero-shot Prompting |
| **Truy vấn AVS (Gom nhóm ngữ nghĩa)** | Rời rạc từng Box, dễ đứt đoạn | 🟢 VLM tự động gom thành câu hoàn chỉnh |
| **Xử lý tiếng Việt (Bảng hiệu phức tạp)**| Hay lỗi dấu tiếng Việt | 🟢 SOTA Multilingual, đọc chính xác dấu |
| **Điểm số tIoU (Định vị thời gian)** | Chỉ được 1 frame/shot (Dễ tIoU thấp) | 🟢 Multi-frame Sampling tối ưu tIoU |
| **Chi phí tính toán** | 🟢 Thấp (chạy được nhiều frames) | 🟡 Cao (Chỉ chạy 2-3 keyframes / shot) |
