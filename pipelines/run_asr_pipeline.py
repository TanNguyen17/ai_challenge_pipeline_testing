import os
import sys
import time
import json
import argparse
import tempfile
import subprocess
import torch
import whisperx
from transformers import AutoProcessor, AutoModelForCausalLM, Qwen3ASRForConditionalGeneration

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
from typing import List, Dict, Any
from extract.workers.keyframe_loader import KeyframeLoader

class ASRPipelineRunner:
    """
    SOTA 4-Stage Vietnamese Video ASR Pipeline:
    Stage 1: WhisperX VAD (Voice Activity Detection Chunking)
    Stage 2: Qwen3-ASR-1.7B (High-throughput Specialized ASR)
    Stage 3: WhisperX Phoneme/Word-level Timestamp Alignment
    Stage 4: Two-Tier DB Schema Export (asr_span / shot)
    """
    def __init__(self, video_dir: str, output_dir: str, keyframes_dir: str, media_info_dir: str, device: str = "cuda"):
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.keyframes_dir = keyframes_dir
        self.media_info_dir = media_info_dir
        self.device = device
        os.makedirs(output_dir, exist_ok=True)
        self.keyframe_loader = KeyframeLoader(keyframes_dir)
        self._init_asr_model()
        self._init_whisperx()

    def _init_asr_model(self):
        """Initializes Qwen3-ASR-1.7B ASR model engine."""
        self.asr_model = None
        self.asr_processor = None
        self.vad_model = None
        try:
            print("Loading Qwen3-ASR-1.7B...")
            model_id = "Qwen/Qwen3-ASR-1.7B"
            use_gpu = self.device == "cuda"
            
            self.asr_processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            self.asr_model = Qwen3ASRForConditionalGeneration.from_pretrained(
                model_id, 
                torch_dtype=torch.bfloat16 if (use_gpu and torch.cuda.is_bf16_supported()) else torch.float16,
                device_map="cuda" if use_gpu else "cpu",
                trust_remote_code=True
            )
            self.asr_model.eval()
            print(f"✅ ASR Model '{model_id}' loaded successfully.")
            
            # Load VAD model from WhisperX for robust chunking
            print("Loading WhisperX VAD model...")
            self.vad_model = whisperx.vad.load_vad_model(torch.device(self.device))
            print("✅ VAD Model loaded.")
                
        except Exception as e:
            raise RuntimeError(f"❌ FATAL: Qwen3-ASR engine failed to initialize: {e}")

    def _init_whisperx(self):
        """Initializes WhisperX Alignment Model for exact Phoneme/Word timestamps."""
        self.align_model = None
        self.align_metadata = None
        try:
            print("Loading WhisperX Alignment Model...")
            self.align_model, self.align_metadata = whisperx.load_align_model(language_code="vi", device=self.device)
            print("✅ WhisperX Alignment Model loaded successfully.")
        except Exception as e:
            print(f"⚠️ Warning: Cannot load WhisperX alignment model: {e}")

    def _extract_audio_wav(self, video_path: str, temp_wav_path: str) -> bool:
        """Extracts 16kHz mono WAV audio track from video file using ffmpeg/moviepy."""
        try:
            cmd = [
                "ffmpeg", "-y", "-i", video_path,
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                temp_wav_path
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            if not (os.path.exists(temp_wav_path) and os.path.getsize(temp_wav_path) > 0):
                raise RuntimeError(f"ffmpeg produced empty audio for {video_path}")
            return True
        except FileNotFoundError:
            raise RuntimeError("❌ FATAL: ffmpeg not found. Please install it (e.g. apt-get install -y ffmpeg)")
        except subprocess.CalledProcessError as e:
            print(f"❌ ffmpeg failed for {video_path}: {e}")
            return False

    def process_audio(self, video_path: str) -> Dict[str, Any]:
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        start_time = time.time()
        temp_wav_path = os.path.join(self.output_dir, f"{video_id}_temp.wav")

        documents = []
        vad_segment_count = 0

        audio_extracted = self._extract_audio_wav(video_path, temp_wav_path)

        if audio_extracted and self.asr_model is not None:
            try:
                # 1. VAD Chunking
                audio_np = whisperx.load_audio(temp_wav_path)
                vad_segments = whisperx.vad.find_vad(audio_np, self.vad_model)
                # Ensure segments are manageable (e.g. max 30 seconds)
                vad_segments = whisperx.vad.merge_chunks(vad_segments, chunk_size=30)
                
                vad_segment_count = len(vad_segments)
                wx_segments = []
                
                # 2. Qwen3-ASR Transcription
                for seg in vad_segments:
                    start_sec = seg["start"]
                    end_sec = seg["end"]
                    
                    # Slice audio array (16kHz)
                    start_sample = int(start_sec * 16000)
                    end_sample = int(end_sec * 16000)
                    chunk_audio = audio_np[start_sample:end_sample]
                    
                    if len(chunk_audio) < 1600: # Skip very short noise (<0.1s)
                        continue
                        
                    conversation = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": [
                            {"type": "audio", "audio_url": "placeholder.wav"},
                            {"type": "text", "text": "Trích xuất chính xác văn bản tiếng Việt từ đoạn âm thanh này. Chỉ trả về văn bản, không giải thích, không dịch."}
                        ]}
                    ]
                    
                    prompt = self.asr_processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
                    inputs = self.asr_processor(
                        text=prompt,
                        audios=chunk_audio,
                        return_tensors="pt",
                        sampling_rate=16000
                    ).to(self.device)
                    
                    with torch.no_grad():
                        generated_ids = self.asr_model.generate(**inputs, max_length=256)
                        
                    generated_ids = generated_ids[:, inputs.input_ids.size(1):]
                    transcription = self.asr_processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                    
                    raw_text = transcription.strip()
                    if not raw_text: continue
                    
                    # Hallucination Filtering (Repeated word loops)
                    words_list = raw_text.split()
                    if len(words_list) > 10:
                        is_hallucination = False
                        for i in range(len(words_list) - 5):
                            if words_list[i] == words_list[i+1] == words_list[i+2] == words_list[i+3]:
                                is_hallucination = True
                                break
                        if is_hallucination:
                            print(f"⚠️ Filtered hallucination loop in {video_id} at {start_sec}s")
                            continue
                            
                    wx_segments.append({
                        "text": raw_text,
                        "start": start_sec,
                        "end": end_sec,
                        "words": []
                    })

                word_level_data = []
                
                # 3. WhisperX Alignment (Real SOTA Phoneme alignment)
                if hasattr(self, 'align_model') and self.align_model is not None and wx_segments:
                    try:
                        audio_np = whisperx.load_audio(temp_wav_path)
                        result = whisperx.align(wx_segments, self.align_model, self.align_metadata, audio_np, self.device, return_char_alignments=False)
                        for seg in result["segments"]:
                            for w in seg.get("words", []):
                                if "start" in w and "end" in w:
                                    word_level_data.append({
                                        "word": w["word"],
                                        "start": w["start"],
                                        "end": w["end"],
                                        "score": w.get("score", 0.9)
                                    })
                    except Exception as e:
                        print(f"WhisperX alignment error for {video_id}: {e}")
                
                # Fallback to chunk timestamps if WhisperX fails
                if not word_level_data:
                    for seg in wx_segments:
                        # Since we don't have exact word timestamps from Qwen3-ASR, distribute time evenly
                        words = seg["text"].split()
                        if not words: continue
                        duration = seg["end"] - seg["start"]
                        w_dur = duration / len(words)
                        for idx, w in enumerate(words):
                            word_level_data.append({
                                "word": w,
                                "start": round(seg["start"] + idx * w_dur, 2),
                                "end": round(seg["start"] + (idx + 1) * w_dur, 2),
                                "score": 0.9
                            })
                            
                # 4. Shot Mapping & Two-Tier Document Construction
                keyframes = self.keyframe_loader.load(video_id)
                shots_data = {}
                
                for kf in keyframes:
                    shot_id = kf["keyframe_n"]
                    if shot_id not in shots_data:
                        shots_data[shot_id] = {
                            "time_range": {"start_sec": kf.get("start_sec", 0.0), "end_sec": kf.get("end_sec", 0.0)},
                            "words": []
                        }
                    else:
                        shots_data[shot_id]["time_range"]["start_sec"] = min(shots_data[shot_id]["time_range"]["start_sec"], kf.get("start_sec", 0.0))
                        shots_data[shot_id]["time_range"]["end_sec"] = max(shots_data[shot_id]["time_range"]["end_sec"], kf.get("end_sec", 0.0))
                        
                for w in word_level_data:
                    for shot_id, s_data in shots_data.items():
                        # Find which shot this word belongs to
                        if s_data["time_range"]["start_sec"] <= w["start"] <= s_data["time_range"]["end_sec"]:
                            s_data["words"].append(w)
                            
                            doc = {
                                "doc_type": "asr_span",
                                "video_id": video_id,
                                "shot_id": shot_id,
                                "word": w["word"],
                                "time_range": {"start_sec": round(w["start"], 2), "end_sec": round(w["end"], 2)},
                                "confidence": round(w["score"], 3)
                            }
                            documents.append(doc)
                            break
                            
                for shot_id, s_data in shots_data.items():
                    if not s_data["words"]:
                        continue
                    
                    full_transcript = " ".join([w["word"] for w in s_data["words"]])
                    
                    shot_doc = {
                        "doc_type": "shot",
                        "video_id": video_id,
                        "shot_id": shot_id,
                        "time_range": s_data["time_range"],
                        "asr_data_combined": {
                            "transcript": full_transcript,
                            "asr_no_accent": self._remove_accents(full_transcript)
                        }
                    }
                    documents.append(shot_doc)

            except Exception as ex:
                print(f"Error transcribing audio for {video_id}: {ex}")

        # Cleanup temporary audio file
        if os.path.exists(temp_wav_path):
            try:
                os.remove(temp_wav_path)
            except Exception:
                pass

        elapsed_sec = round(time.time() - start_time, 2)
        return {
            "video_id": video_id,
            "elapsed_sec": elapsed_sec,
            "num_vad_segments": vad_segment_count,
            "num_documents": len(documents),
            "documents": documents
        }

    @staticmethod
    def _remove_accents(input_str: str) -> str:
        s = input_str.lower()
        accents = {
            'a': 'àáảạãăằắẳặẵâầấẩậẫ',
            'd': 'đ',
            'e': 'èéẻẹẽêềếểệễ',
            'i': 'ìíỉịĩ',
            'o': 'òóỏọõôồốổộỗơờớởợỡ',
            'u': 'ùúủụũưừứửựữ',
            'y': 'ỳýỷỵỹ'
        }
        for char, accented_chars in accents.items():
            for a in accented_chars:
                s = s.replace(a, char)
        return s

    def run_benchmark(self, limit_videos: int = 50):
        print(f"\n🚀 Running SOTA ASR Pipeline (Qwen3-ASR) on up to {limit_videos} videos...")
        video_files = []
        for root, _, files in os.walk(self.video_dir):
            for f in files:
                if f.lower().endswith(('.mp4', '.mkv', '.avi', '.webm')):
                    video_files.append(os.path.join(root, f))
        video_files = video_files[:limit_videos]

        if not video_files:
            print("⚠️ No video files found in directory.")
            return

        out_jsonl = os.path.join(self.output_dir, "asr_extracted_documents.jsonl")
        processed_video_ids = set()
        if os.path.exists(out_jsonl):
            with open(out_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        doc = json.loads(line)
                        processed_video_ids.add(doc.get("video_id"))
        
        pending_videos = []
        for v in video_files:
            v_id = os.path.splitext(os.path.basename(v))[0]
            if v_id not in processed_video_ids:
                pending_videos.append(v)
            else:
                print(f"⏭️ Skipping {v_id} - already processed (resume).")
                
        video_files = pending_videos

        if not video_files:
            print("⚠️ No pending video files found to process (all done).")
            return

        total_time = 0.0
        total_vad = 0
        total_docs = 0

        # Open in append mode for true fail-safe streaming output
        with open(out_jsonl, "a", encoding="utf-8") as f:
            for idx, v_path in enumerate(video_files):
                print(f"[{idx+1}/{len(video_files)}] Processing ASR for {os.path.basename(v_path)}...")
                res = self.process_audio(v_path)
                
                # Write immediately
                for doc in res["documents"]:
                    f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                f.flush()
                
                total_time += res["elapsed_sec"]
                total_vad += res.get("num_vad_segments", 0)
                total_docs += res.get("num_documents", 0)

        benchmark_report = {
            "pipeline": "SOTA 4-Stage Vietnamese ASR (Qwen3-ASR)",
            "videos_processed": len(video_files),
            "total_elapsed_sec": round(total_time, 2),
            "avg_time_per_video_sec": round(total_time / len(video_files), 3) if video_files else 0,
            "total_vad_segments": total_vad,
            "total_asr_documents": total_docs,
            "output_jsonl_path": out_jsonl
        }

        report_path = os.path.join(self.output_dir, "asr_benchmark_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(benchmark_report, f, indent=2, ensure_ascii=False)

        print("\n📊 --- ASR BENCHMARK REPORT ---")
        print(json.dumps(benchmark_report, indent=2, ensure_ascii=False))
        print(f"✅ Saved ASR extracted documents to: {out_jsonl}")

def exp_prob(logprob: float) -> float:
    import math
    try:
        return min(1.0, math.exp(logprob))
    except Exception:
        return 0.90

def main():
    parser = argparse.ArgumentParser(description="Run 5-Stage SOTA Vietnamese ASR Pipeline")
    parser.add_argument("--video-dir", type=str, default="./data/extracted", help="Directory containing raw videos")
    parser.add_argument("--output-dir", type=str, default="./data/processed/asr", help="Output directory for ASR records")
    parser.add_argument("--keyframes-dir", type=str, default="./data/extracted/video batch 1/map-keyframes-aic25-b1/map-keyframes", help="Directory containing BTC keyframe CSVs")
    parser.add_argument("--media-info-dir", type=str, default="./data/extracted/video batch 1/media-info-aic25-b1/media-info", help="Directory containing BTC media-info JSONs")
    parser.add_argument("--limit", type=int, default=50, help="Maximum number of videos to process in benchmark")
    args = parser.parse_args()

    runner = ASRPipelineRunner(args.video_dir, args.output_dir, args.keyframes_dir, args.media_info_dir)
    runner.run_benchmark(args.limit)

if __name__ == "__main__":
    main()
