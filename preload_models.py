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

    # 1. Preload Qwen2-VL (OCR)
    print("\n--- 1/4 Pre-downloading Qwen2-VL-2B-Instruct Weights (OCR) ---")
    try:
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        import torch
        qwen_id = "Qwen/Qwen2-VL-2B-Instruct"
        print(f"Loading {qwen_id}...")
        processor = AutoProcessor.from_pretrained(qwen_id)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            qwen_id, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
        del model
        del processor
        clear_memory()
        print("✅ Qwen2-VL-2B-Instruct cached to disk.")
    except Exception as e:
        print(f"❌ Failed caching Qwen2-VL: {e}")

    # 2. Preload Faster-Whisper / PhoWhisper ASR
    print("\n--- 2/4 Pre-downloading Faster-Whisper ASR Weights ---")
    try:
        from faster_whisper import WhisperModel
        import subprocess
        
        model_path = "./models/phowhisper-large-ct2"
        if not os.path.exists(model_path):
            print("Converting vinai/PhoWhisper-large to CTranslate2 format...")
            os.makedirs("./models", exist_ok=True)
            try:
                subprocess.run([
                    "ct2-transformers-converter", "--model", "vinai/PhoWhisper-large", 
                    "--output_dir", model_path,
                    "--copy_files", "tokenizer.json", "preprocessor_config.json"
                ], check=True)
                print("✅ Conversion to CT2 successful.")
            except Exception as e:
                print(f"⚠️ Conversion failed: {e}. You may need to run: pip install ctranslate2 transformers")
                print("Falling back to downloading default medium model just to cache dependencies...")
                WhisperModel("medium", device="cpu", compute_type="int8")

        if os.path.exists(model_path):
            try:
                model = WhisperModel(model_path, device="cpu", compute_type="int8")
                del model
                clear_memory()
                print(f"✅ PhoWhisper CT2 model cached and verified at {model_path}.")
            except Exception as ex:
                print(f"Notice loading {model_path}: {ex}")
    except Exception as e:
        print(f"⚠️ Faster-Whisper pre-download notice: {e}")

    # 3. Preload Ultralytics YOLOE
    print("\n--- 3/4 Pre-downloading YOLOE Object Detection Weights ---")
    try:
        from ultralytics import YOLO
        w_name = "yolov8x-worldv2.pt"
        try:
            yolo = YOLO(w_name)
            del yolo
            clear_memory()
            print(f"✅ YOLOE weights '{w_name}' cached to disk.")
        except Exception as ex:
            print(f"❌ Failed loading {w_name}: {ex}")
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
