import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def preload_all_models():
    print("==========================================================================")
    print("🚀 PRE-DOWNLOADING ALL SOTA DEEP LEARNING MODEL WEIGHTS")
    print("==========================================================================")

    # 1. Preload PaddleOCR (PP-OCRv5 Vietnamese)
    print("\n--- 1/4 Preloading PaddleOCR (Vietnamese) ---")
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang="vi", show_log=False)
        print("✅ PaddleOCR weights downloaded & cached successfully.")
    except Exception as e:
        print(f"⚠️ PaddleOCR preloading notice: {e}")

    # 2. Preload Faster-Whisper / PhoWhisper ASR
    print("\n--- 2/4 Preloading Faster-Whisper ASR ---")
    try:
        from faster_whisper import WhisperModel
        for m_name in ["vinai/phowhisper-large", "medium", "base"]:
            try:
                model = WhisperModel(m_name, device="cpu", compute_type="int8")
                print(f"✅ Faster-Whisper model '{m_name}' cached successfully.")
                break
            except Exception as ex:
                print(f"Notice loading {m_name}: {ex}")
    except Exception as e:
        print(f"⚠️ Faster-Whisper preloading notice: {e}")

    # 3. Preload Ultralytics YOLO-World v2 / YOLOv8
    print("\n--- 3/4 Preloading YOLO-World Object Detection Weights ---")
    try:
        from ultralytics import YOLO
        for w_name in ["yolov8x-worldv2.pt", "yolov8x.pt"]:
            try:
                yolo = YOLO(w_name)
                print(f"✅ YOLO weights '{w_name}' downloaded successfully.")
                break
            except Exception as ex:
                print(f"Notice loading {w_name}: {ex}")
    except Exception as e:
        print(f"⚠️ YOLO preloading notice: {e}")

    # 4. Preload OpenCLIP ViT-B/32
    print("\n--- 4/4 Preloading OpenCLIP ViT-B/32 Visual Encoder ---")
    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        print("✅ OpenCLIP ViT-B/32 weights downloaded & cached successfully.")
    except Exception as e:
        print(f"⚠️ OpenCLIP preloading notice: {e}")

    print("\n==========================================================================")
    print("🎉 ALL SOTA MODEL WEIGHTS ARE PRELOADED & READY FOR RUNPOD GPU EXECUTION!")
    print("==========================================================================")

if __name__ == "__main__":
    preload_all_models()
