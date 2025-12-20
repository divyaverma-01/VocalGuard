import os
import pandas as pd

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

STRESS_MAP = {
    "neutral": 0,
    "calm": 0,
    "happy": 0,
    "sad": 0,
    "angry": 1,
    "fearful": 1,
    "disgust": 1,
    "surprised": None,
}

def build_metadata(ravdess_root):
    records = []

    for actor_dir in sorted(os.listdir(ravdess_root)):
        if not actor_dir.startswith("Actor_"):
            continue

        actor_path = os.path.join(ravdess_root, actor_dir)
        speaker_id = actor_dir.split("_")[1]

        for fname in os.listdir(actor_path):
            if not fname.endswith(".wav"):
                continue

            parts = fname.replace(".wav", "").split("-")
            emotion_code = parts[2]
            emotion = EMOTION_MAP[emotion_code]
            label = STRESS_MAP[emotion]

            if label is None:
                continue

            records.append({
                "file_path": os.path.join(actor_path, fname),
                "speaker_id": speaker_id,
                "emotion": emotion,
                "label": label,
            })

    return pd.DataFrame(records)
