"""
gate_logic.py — Gate Verification Logic

Combines speaker verification and speech recognition
to decide whether to grant or deny access.

Both conditions must pass:
  1. Speaker identity matches an enrolled user
  2. Voice command matches the passphrase
"""

from voice_embedding import extract_embedding, compare_embeddings
from speech_recognition_module import transcribe_audio, check_passphrase


# ──────────────────────────────────────────────
# Step 1: Verify Access (Main Gate Logic)
# ──────────────────────────────────────────────
def verify_access(audio_path, target_name, target_embedding, target_passphrase):
    """
    1-to-1 Verification pipeline:
    1. Extract voice embedding from the audio.
    2. Compare it with the target user's enrolled embedding.
    3. Transcribe the audio and check if it matches the target user's custom passphrase.

    Args:
        audio_path (str): Path to the verification audio file
        target_name (str): The name of the user claiming access
        target_embedding (torch.Tensor): Enrolled embedding tensor of the target user
        target_passphrase (str): Enrolled custom passphrase of the target user

    Returns:
        dict: Result with keys:
            - access_granted (bool)
            - speaker_matched (bool)
            - speaker_name (str)
            - speaker_score (float)
            - command_matched (bool)
            - transcribed_text (str)
            - command_score (float)
    """

    result = {
        "access_granted": False,
        "speaker_matched": False,
        "speaker_name": target_name,
        "speaker_score": 0.0,
        "command_matched": False,
        "transcribed_text": "",
        "command_score": 0.0,
    }

    # ── Step 2: Extract embedding from verification audio ──
    try:
        test_embedding = extract_embedding(audio_path)
    except Exception as e:
        result["transcribed_text"] = f"[ERROR] Embedding extraction failed: {e}"
        return result

    # ── Step 3: Compare with the target user's enrolled embedding (1-to-1) ──
    is_match, score = compare_embeddings(test_embedding, target_embedding)
    result["speaker_score"] = round(score, 4)
    result["speaker_matched"] = is_match

    # ── Step 4: Transcribe audio and check custom passphrase ──
    transcribed = transcribe_audio(audio_path)
    result["transcribed_text"] = transcribed

    if not transcribed.startswith("[ERROR]"):
        cmd_match, cmd_score = check_passphrase(transcribed, target_passphrase)
        result["command_matched"] = cmd_match
        result["command_score"] = round(cmd_score, 4)

    # ── Step 5: Grant access only if BOTH checks pass ──
    if result["speaker_matched"] and result["command_matched"]:
        result["access_granted"] = True

    return result

