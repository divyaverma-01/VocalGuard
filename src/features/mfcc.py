import numpy as np
import librosa


def extract_mfcc(
    y,
    sr,
    n_mfcc=13,
    n_fft=2048,
    hop_length=512,
    max_frames=128,
):
    """
    Returns MFCC + delta + delta-delta
    Shape: (3, n_mfcc, max_frames)
    """

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    # Delta features
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # Stack channels
    features = np.stack([mfcc, delta, delta2], axis=0)

    # Time normalization
    if features.shape[2] < max_frames:
        pad_width = max_frames - features.shape[2]
        features = np.pad(
            features,
            ((0, 0), (0, 0), (0, pad_width)),
            mode="constant",
        )
    else:
        features = features[:, :, :max_frames]

    return features.astype(np.float32)
