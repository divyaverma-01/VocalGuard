import os
import librosa
import soundfile as sf
import numpy as np
import pandas as pd


def preprocess_and_window(
    metadata_path,
    splits_dir,
    output_root,
    windows_csv_path="data/processed/windows.csv",
    sample_rate=16000,
    window_duration=4.0,
    hop_duration=4.0,
):

    # Load metadata (ground truth)
    df = pd.read_csv(metadata_path, dtype={"speaker_id": str})
    df["speaker_id"] = df["speaker_id"].str.zfill(2)

    window_samples = int(window_duration * sample_rate)
    hop_samples = int(hop_duration * sample_rate)

    all_records = []

    for split in ["train", "val", "test"]:
        # Load speaker split
        with open(os.path.join(splits_dir, f"{split}_speakers.txt")) as f:
            speakers = set(f.read().splitlines())

        split_df = df[df["speaker_id"].isin(speakers)]

        out_dir = os.path.join(output_root, split)
        os.makedirs(out_dir, exist_ok=True)

        for _, row in split_df.iterrows():
            # Load audio
            y, sr = librosa.load(
                row["file_path"],
                sr=sample_rate,
                mono=True,
            )

            # Trim silence
            y, _ = librosa.effects.trim(y, top_db=20)

            # Amplitude normalization
            if np.max(np.abs(y)) > 0:
                y = y / np.max(np.abs(y))

            total_len = len(y)

            # Ensure at least one window
            num_windows = max(
                1,
                int(np.ceil((total_len - window_samples) / hop_samples)) + 1,
            )

            base_name = os.path.basename(row["file_path"]).replace(".wav", "")

            for i in range(num_windows):
                start = i * hop_samples
                end = start + window_samples
                window = y[start:end]

                if len(window) < window_samples:
                    window = np.pad(
                        window,
                        (0, window_samples - len(window)),
                        mode="constant",
                    )

                fname = f"{row['speaker_id']}_{base_name}_win{i}.wav"
                out_path = os.path.join(out_dir, fname)

                sf.write(out_path, window, sample_rate)

                all_records.append({
                    "file_path": out_path,
                    "speaker_id": row["speaker_id"],
                    "label": int(row["label"]),
                    "split": split,
                })

    # Save window-level metadata
    windows_df = pd.DataFrame(all_records)
    os.makedirs(os.path.dirname(windows_csv_path), exist_ok=True)
    windows_df.to_csv(windows_csv_path, index=False)

    return windows_df
