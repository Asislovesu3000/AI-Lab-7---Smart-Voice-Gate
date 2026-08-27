"""
voice_embedding.py — Speaker Voice Embedding Module

Uses SpeechBrain's pretrained ECAPA-TDNN model to extract voice embeddings
and compare them for speaker verification.

AI Technique: Speaker Verification (Voice Biometrics)
"""

from speechbrain.inference.speaker import EncoderClassifier
import torch
import os
import shutil
import soundfile as sf
import numpy as np
from pathlib import Path

# ──────────────────────────────────────────────
# Fix: Windows does not allow symlinks without admin privileges.
# Patch SpeechBrain's file linking to use copy instead of symlink.
# ──────────────────────────────────────────────
import speechbrain.utils.fetching as _sb_fetch

def _copy_instead_of_symlink(src, dst, strategy=None):
    """Copy file instead of creating a symlink (Windows fix)."""
    src, dst = Path(src), Path(dst)
    if dst.exists():
        dst.unlink()
    shutil.copy2(str(src), str(dst))

_sb_fetch.link_with_strategy = _copy_instead_of_symlink

# ──────────────────────────────────────────────
# Step 1: Load the Pretrained Speaker Embedding Model
# ──────────────────────────────────────────────
# ECAPA-TDNN is a state-of-the-art model for speaker verification.
# It converts any voice clip into a fixed-size embedding vector (192-dim).

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
)


# ──────────────────────────────────────────────
# Step 2: Extract Embedding from an Audio File
# ──────────────────────────────────────────────
def extract_embedding(audio_path):
    """
    Takes a path to a WAV audio file, loads it,
    and returns a speaker embedding vector.

    Args:
        audio_path (str): Path to the audio file (.wav)

    Returns:
        torch.Tensor: Speaker embedding vector (1 x 192)
    """
    # Load audio using soundfile (avoids torchaudio/torchcodec issues)
    data, sample_rate = sf.read(audio_path)

    # Convert to numpy float32
    data = np.array(data, dtype=np.float32)

    # Convert to mono if stereo
    if data.ndim == 2:
        data = np.mean(data, axis=1)

    # Resample to 16kHz if needed (model expects 16kHz)
    if sample_rate != 16000:
        # Simple resampling using numpy interpolation
        duration = len(data) / sample_rate
        target_length = int(duration * 16000)
        data = np.interp(
            np.linspace(0, len(data), target_length),
            np.arange(len(data)),
            data
        )

    # Convert to torch tensor with shape (1, num_samples)
    signal = torch.tensor(data, dtype=torch.float32).unsqueeze(0)

    # Extract embedding using the pretrained model
    embedding = classifier.encode_batch(signal)

    return embedding.squeeze()


# ──────────────────────────────────────────────
# Step 3: Compare Two Embeddings (Cosine Similarity)
# ──────────────────────────────────────────────
def compare_embeddings(embedding1, embedding2, threshold=0.25):
    """
    Compares two speaker embeddings using cosine similarity.

    Args:
        embedding1 (torch.Tensor): First speaker embedding
        embedding2 (torch.Tensor): Second speaker embedding
        threshold (float): Minimum similarity score to consider a match

    Returns:
        tuple: (is_match: bool, similarity_score: float)
    """
    # Compute cosine similarity between the two embeddings
    similarity = torch.nn.functional.cosine_similarity(
        embedding1.unsqueeze(0),
        embedding2.unsqueeze(0)
    )

    score = similarity.item()
    is_match = score >= threshold

    return is_match, score
