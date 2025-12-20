import pandas as pd
import random
import os

def make_speaker_splits(
    metadata_path,
    output_dir,
    train_ratio=0.7,
    val_ratio=0.15,
    seed=42
):
    df = pd.read_csv(metadata_path)

    speakers = sorted(df["speaker_id"].unique())
    random.seed(seed)
    random.shuffle(speakers)

    n_total = len(speakers)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)

    train_speakers = speakers[:n_train]
    val_speakers = speakers[n_train:n_train + n_val]
    test_speakers = speakers[n_train + n_val:]

    os.makedirs(output_dir, exist_ok=True)

    def save_split(name, speaker_list):
        path = os.path.join(output_dir, f"{name}_speakers.txt")
        with open(path, "w") as f:
            for s in speaker_list:
                f.write(f"{s}\n")

    save_split("train", train_speakers)
    save_split("val", val_speakers)
    save_split("test", test_speakers)

    return {
        "train": train_speakers,
        "val": val_speakers,
        "test": test_speakers
    }
