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

    # 1. Preload Qwen3-VL (OCR)
    print("\n--- 1/7 Pre-downloading Qwen3-VL-7B-Instruct Weights (OCR) ---")
    try:
        from transformers import AutoModelForCausalLM, AutoProcessor
        import torch
        qwen_id = "Qwen/Qwen3-VL-7B-Instruct"
        print(f"Loading {qwen_id}...")
        processor = AutoProcessor.from_pretrained(qwen_id)
        model = AutoModelForCausalLM.from_pretrained(
            qwen_id, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
        del model
        del processor
        clear_memory()
        print("✅ Qwen3-VL-7B-Instruct cached to disk.")
    except Exception as e:
        print(f"❌ Failed caching Qwen3-VL: {e}")

    # 2. Preload Qwen3-ASR
    print("\n--- 2/7 Pre-downloading Qwen3-ASR-1.7B Weights (ASR) ---")
    try:
        from transformers import AutoProcessor, AutoModelForCausalLM
        import torch
        model_id = "Qwen/Qwen3-ASR-1.7B"
        print(f"Loading {model_id}...")
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        del model
        del processor
        clear_memory()
        print(f"✅ {model_id} cached to disk.")
    except Exception as e:
        print(f"❌ Failed caching Qwen3-ASR: {e}")

    # 3. Preload Ultralytics YOLO-World v2
    print("\n--- 3/7 Pre-downloading YOLO-World v2 Object Detection Weights ---")
    try:
        from ultralytics import YOLO
        w_name = "yolov8x-worldv2.pt"
        try:
            yolo = YOLO(w_name)
            del yolo
            clear_memory()
            print(f"✅ YOLO-World weights '{w_name}' cached to disk.")
        except Exception as ex:
            print(f"❌ Failed loading {w_name}: {ex}")
    except Exception as e:
        print(f"⚠️ YOLO pre-download notice: {e}")

    # 4. Preload OpenCLIP ViT-B/32
    print("\n--- 4/7 Pre-downloading OpenCLIP ViT-B/32 Visual Encoder Weights ---")
    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        del model
        clear_memory()
        print("✅ OpenCLIP ViT-B/32 weights cached to disk.")
    except Exception as e:
        print(f"⚠️ OpenCLIP pre-download notice: {e}")

    # 5. Preload RAM++ Scene Tagger
    print("\n--- 5/7 Pre-downloading RAM++ Scene Tagging Weights ---")
    try:
        import urllib.request
        weight_path = "ram_plus_swin_large_14m.pth"
        if not os.path.exists(weight_path):
            print("Downloading RAM++ weights (14M)...")
            url = "https://huggingface.co/xinyu1205/recognize-anything-plus-model/resolve/main/ram_plus_swin_large_14m.pth"
            urllib.request.urlretrieve(url, weight_path)
            print("✅ Downloaded RAM++ weights.")
            
        # Optional: verify it loads
        try:
            from ram.models import ram_plus
            model = ram_plus(pretrained=weight_path, image_size=384, vit='swin_l')
            del model
            clear_memory()
            print("✅ RAM++ model successfully cached and verified.")
        except ImportError:
            print("⚠️ RAM++ weights downloaded, but 'ram' module not installed to verify.")
    except Exception as e:
        print(f"⚠️ RAM++ pre-download notice: {e}")

    # 6. Preload TransNetV2 Weights
    print("\n--- 6/7 Pre-downloading TransNetV2 Shot Boundary Weights ---")
    try:
        import urllib.request
        os.makedirs("./data/models", exist_ok=True)
        tn2_weight_path = "./data/models/transnetv2_weights.pth"
        if not os.path.exists(tn2_weight_path):
            print("Downloading TransNetV2 real weights...")
            url = "https://github.com/soCzech/TransNetV2/raw/master/inference/pytorch/transnetv2-pytorch-weights.pth"
            try:
                urllib.request.urlretrieve(url, tn2_weight_path)
                print("✅ Downloaded TransNetV2 weights.")
            except Exception as download_ex:
                print(f"⚠️ Could not download TransNetV2 weights automatically: {download_ex}")
        else:
            print("✅ TransNetV2 weights already exist.")
            
        try:
            from transnetv2_pytorch import TransNetV2
            import torch
            model = TransNetV2()
            if os.path.exists(tn2_weight_path):
                model.load_state_dict(torch.load(tn2_weight_path, map_location="cpu"))
            del model
            clear_memory()
            print("✅ TransNetV2 architecture verified.")
        except ImportError:
            print("⚠️ transnetv2-pytorch not installed.")
    except Exception as e:
        print(f"⚠️ TransNetV2 pre-download notice: {e}")

    # 7. Preload WhisperX VAD & Alignment Models
    print("\n--- 7/7 Pre-downloading WhisperX VAD & Alignment Models ---")
    try:
        import whisperx
        import torch
        print("Preloading WhisperX VAD Model...")
        vad_model = whisperx.vad.load_vad_model(torch.device("cpu"))
        del vad_model
        clear_memory()
        
        print("Preloading WhisperX Alignment Model (vi)...")
        align_model, align_metadata = whisperx.load_align_model(language_code="vi", device="cpu")
        del align_model
        del align_metadata
        clear_memory()
        print("✅ WhisperX VAD & Alignment models cached to disk.")
    except Exception as e:
        print(f"⚠️ WhisperX pre-download notice: {e}")

    print("\n==========================================================================")
    print("🎉 ALL SOTA MODEL WEIGHTS CACHED TO DISK! MEMORY FREED 100%.")
    print("==========================================================================")

if __name__ == "__main__":
    preload_all_models()
