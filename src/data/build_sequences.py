import pandas as pd
import json
from pathlib import Path

SEQUENCE_LENGTH = 3   # K
STRIDE = 1            # overlapping sequences


def extract_utterance_id(path):
    fname = Path(path).stem
    return fname.replace("_win0", "")

def compute_escalation_label(labels):
    """
    Late-onset sustained stress escalation.
    labels: List[int] (e.g., [0, 0, 1])
    """
    K = len(labels)

    # Need at least 2 windows to talk about "sustained"
    if K < 2:
        return 0

    # Late-onset sustained stress
    # Current dataset(RAVDESS) has K=3
    if labels[-1] == 1 and labels[-2] == 1:
        if sum(labels[:-2]) < len(labels[:-2]):  # at least one calm earlier
            return 1

    return 0

def build_sequences(windows_csv: str, output_csv: str):
    df = pd.read_csv(windows_csv)

    # Derive utterance_id from file_path
    df["utterance_id"] = df["file_path"].apply(extract_utterance_id)

    sequences = []
    seq_counter = 0

    for speaker_id, spk_df in df.groupby("speaker_id"):
        # Preserve temporal order by filename (RAVDESS is ordered)
        spk_df = spk_df.sort_values("utterance_id").reset_index(drop=True)

        split = spk_df["split"].iloc[0]
        utterances = spk_df["utterance_id"].tolist()

        seq_idx = 0
        for i in range(0, len(utterances) - SEQUENCE_LENGTH + 1, STRIDE):
            slice_df = spk_df.iloc[i : i + SEQUENCE_LENGTH]

            window_files = slice_df["file_path"].tolist()
            labels = slice_df["label"].tolist()

            sequence_label = int(max(labels))

            escalation_label = compute_escalation_label(labels)

            sequences.append({
                "sequence_id": f"seq_{seq_counter:06d}",
                "speaker_id": speaker_id,
                "sequence_index": seq_idx,
                "utterance_ids": json.dumps(slice_df["utterance_id"].tolist()),
                "window_files": json.dumps(window_files),
                "labels": json.dumps(labels),
                "sequence_label": sequence_label,
                "escalation_label": escalation_label,
                "split": split,
            })

            seq_counter += 1
            seq_idx += 1

    seq_df = pd.DataFrame(sequences)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    seq_df.to_csv(output_csv, index=False)

    print(f"✅ Built {len(seq_df)} sequences")
    print(seq_df["split"].value_counts())
    print(seq_df["sequence_label"].value_counts())
    print(seq_df["escalation_label"].value_counts())

    return seq_df


if __name__ == "__main__":
    build_sequences(
        windows_csv="data/processed/windows.csv",
        output_csv="data/processed/sequences.csv",
    )
