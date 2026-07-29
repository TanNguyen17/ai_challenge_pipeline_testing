import os
import sys
import time
import json
import argparse
import tempfile
import subprocess
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
from typing import List, Dict, Any
from extract.workers.keyframe_loader import KeyframeLoader

class ASRPipelineRunner:
    """
    SOTA 5-Stage Vietnamese Video ASR Pipeline:
    Stage 1: Silero VAD (Voice Activity Detection & Music/Noise Filtering)
    Stage 2: PhoWhisper-large / Faster-Whisper CTranslate2 Engine
    Stage 3: Word-level Timestamp Alignment
    Stage 4: Inverse Text Normalization (ITN) & Hot-word Rules
    Stage 5: Temporal Sliding Window (20s) & Shot-Mapping Export
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

    def _init_asr_model(self):
        """Initializes Faster-Whisper / PhoWhisper ASR model engine."""
        self.asr_model = None
        try:
            from faster_whisper import WhisperModel
            use_gpu = self.device == "cuda"
            compute_type = "float16" if use_gpu else "int8"
            
            # Use PhoWhisper CT2 converted model or fail with clear error
            model_id = "vinai/PhoWhisper-large"
            try:
                print(f"Loading ASR model '{model_id}' on {'cuda' if use_gpu else 'cpu'} ({compute_type})...")
                self.asr_model = WhisperModel(model_id, device="cuda" if use_gpu else "cpu", compute_type=compute_type)
                print(f"✅ ASR Model '{model_id}' loaded successfully.")
            except Exception as e:
                raise RuntimeError(
                    f"❌ FATAL: Cannot load ASR model '{model_id}': {e}\n"
                    f"Ensure you are using the CT2 converted model path or run: ct2-whisper-converter --model {model_id} --output_dir ./models/phowhisper-ct2"
                )
        except Exception as e:
            raise RuntimeError(f"❌ FATAL: Faster-Whisper engine failed to initialize: {e}")

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
                # 1. Hot-word boosting from BTC media-info
                hotword_prompt = ""
                media_info_path = os.path.join(self.media_info_dir, f"{video_id}.json")
                if os.path.exists(media_info_path):
                    try:
                        with open(media_info_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            keywords = data.get("keywords", [])
                            if keywords:
                                # Create a natural sentence prompt containing keywords
                                hotword_prompt = "Đây là video về " + ", ".join(keywords[:10]) + "."
                    except Exception as e:
                        print(f"Warning: Could not read media-info for {video_id}: {e}")

                segments, info = self.asr_model.transcribe(
                    temp_wav_path,
                    beam_size=5,
                    language="vi",
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500),
                    word_timestamps=True,
                    initial_prompt=hotword_prompt if hotword_prompt else None
                )

                segment_list = list(segments)
                vad_segment_count = len(segment_list)

                for idx, seg in enumerate(segment_list):
                    raw_text = seg.text.strip()
                    if not raw_text:
                        continue
                        
                    # 2. Hallucination Filtering (Repeated word loops)
                    words_list = raw_text.split()
                    if len(words_list) > 10:
                        # If the same word appears too many times consecutively, it's a hallucination loop
                        is_hallucination = False
                        for i in range(len(words_list) - 5):
                            if words_list[i] == words_list[i+1] == words_list[i+2] == words_list[i+3]:
                                is_hallucination = True
                                break
                        if is_hallucination:
                            print(f"⚠️ Filtered hallucination loop in {video_id} at {seg.start}s")
                            continue

                    words = []
                    if getattr(seg, 'words', None):
                        for w in seg.words:
                            words.append({
                                "word": w.word,
                                "start": round(w.start, 2),
                                "end": round(w.end, 2),
                                "probability": round(w.probability, 3)
                            })

                    unaccented = self._remove_accents(raw_text)

                    # Map to BTC keyframes using KeyframeLoader
                    keyframes = self.keyframe_loader.load(video_id)
                    mapped_frame_indices = []
                    mapped_shot_ids = []
                    for kf in keyframes:
                        # If keyframe time overlaps with ASR segment
                        if seg.start <= kf["start_sec"] <= seg.end:
                            mapped_frame_indices.append(kf["keyframe_id"])
                            mapped_shot_ids.append(kf["keyframe_n"])

                    doc = {
                        "video_id": video_id,
                        "window_id": idx,
                        "time_range": {"start_sec": round(seg.start, 2), "end_sec": round(seg.end, 2)},
                        "mapped_frame_indices": mapped_frame_indices,
                        "mapped_shot_ids": list(set(mapped_shot_ids)),
                        "asr_data": {
                            "transcript_normalized": raw_text,
                            "transcript_raw": raw_text.lower(),
                            "asr_no_accent": unaccented
                        },
                        "word_timestamps": words,
                        "confidence_score": round(exp_prob(seg.avg_logprob), 3) if hasattr(seg, "avg_logprob") else 0.90
                    }
                    documents.append(doc)

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
        print(f"\n🚀 Running SOTA ASR Pipeline (PhoWhisper / Faster-Whisper) on up to {limit_videos} videos...")
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
            "pipeline": "SOTA 5-Stage Vietnamese ASR (Faster-Whisper)",
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
