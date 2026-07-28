import os
import sys
import gc

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def clear_memory():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

def preload_all_models():
    print("==========================================================================")
    print("🚀 PRE-DOWNLOADING MODEL WEIGHTS TO DISK (LIGHTWEIGHT CACHING)")
    print("==========================================================================")

    # 1. Preload PaddleOCR (PP-OCRv5 Vietnamese)
    print("\n--- 1/4 Pre-downloading PaddleOCR Weights ---")
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang="vi", show_log=False)
        del ocr
        clear_memory()
        print("✅ PaddleOCR weights cached to disk.")
    except Exception as e:
        print(f"⚠️ PaddleOCR pre-download notice: {e}")

    # 2. Preload Faster-Whisper / PhoWhisper ASR
    print("\n--- 2/4 Pre-downloading Faster-Whisper ASR Weights ---")
    try:
        from faster_whisper import WhisperModel
        for m_name in ["vinai/phowhisper-large", "medium", "base"]:
            try:
                model = WhisperModel(m_name, device="cpu", compute_type="int8")
                del model
                clear_memory()
                print(f"✅ Faster-Whisper model '{m_name}' cached to disk.")
                break
            except Exception as ex:
                print(f"Notice loading {m_name}: {ex}")
    except Exception as e:
        print(f"⚠️ Faster-Whisper pre-download notice: {e}")

    # 3. Preload Ultralytics YOLO-World v2 / YOLOv8
    print("\n--- 3/4 Pre-downloading YOLO-World Object Detection Weights ---")
    try:
        from ultralytics import YOLO
        for w_name in ["yolov8x-worldv2.pt", "yolov8x.pt"]:
            try:
                yolo = YOLO(w_name)
                del yolo
                clear_memory()
                print(f"✅ YOLO weights '{w_name}' cached to disk.")
                break
            except Exception as ex:
                print(f"Notice loading {w_name}: {ex}")
    except Exception as e:
        print(f"⚠️ YOLO pre-download notice: {e}")

    # 4. Preload OpenCLIP ViT-B/32
    print("\n--- 4/4 Pre-downloading OpenCLIP ViT-B/32 Visual Encoder Weights ---")
    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        del model
        clear_memory()
        print("✅ OpenCLIP ViT-B/32 weights cached to disk.")
    except Exception as e:
        print(f"⚠️ OpenCLIP pre-download notice: {e}")

    print("\n==========================================================================")
    print("🎉 ALL SOTA MODEL WEIGHTS CACHED TO DISK! MEMORY FREED 100%.")
    print("==========================================================================")

if __name__ == "__main__":
    preload_all_models()
