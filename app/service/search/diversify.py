from typing import List, Dict, Any

class DiversifyService:
    @staticmethod
    def temporal_deduplicate(hits: List[Dict[str, Any]], frame_gap: int = 450, top_k: int = 100) -> List[Dict[str, Any]]:
        """
        Deduplicates keyframes from the same video that are too close temporally (i.e. within frame_gap).
        """
        selected_hits = []
        video_to_frames = {}  # video_id -> list of selected frame indices

        for hit in hits:
            video_id = hit["video_id"]
            frame_idx = hit["frame_idx"]

            if video_id not in video_to_frames:
                video_to_frames[video_id] = [frame_idx]
                selected_hits.append(hit)
            else:
                # Check if this frame is too close to any already selected frame in the same video
                too_close = False
                for selected_frame in video_to_frames[video_id]:
                    if abs(frame_idx - selected_frame) < frame_gap:
                        too_close = True
                        break
                
                if not too_close:
                    video_to_frames[video_id].append(frame_idx)
                    selected_hits.append(hit)

            if len(selected_hits) >= top_k:
                break

        return selected_hits
