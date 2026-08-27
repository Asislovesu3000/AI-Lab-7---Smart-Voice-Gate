"""
speech_recognition_module.py — Speech-to-Text Module

Uses the SpeechRecognition library with Google Web Speech API
to transcribe spoken audio into text.

AI Technique: Speech Recognition (NLP)
"""

import speech_recognition as sr


# ──────────────────────────────────────────────
# Step 1: Initialize the Speech Recognizer
# ──────────────────────────────────────────────
recognizer = sr.Recognizer()


# ──────────────────────────────────────────────
# Step 2: Transcribe Audio File to Text
# ──────────────────────────────────────────────
def transcribe_audio(audio_path):
    """
    Takes a path to a WAV audio file and returns the transcribed text.
    Uses Google Web Speech API (free, no API key needed).

    Args:
        audio_path (str): Path to the audio file (.wav)

    Returns:
        str: Transcribed text (lowercase), or error message
    """
    try:
        # Load the audio file
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)

        # Transcribe using Google Web Speech API
        text = recognizer.recognize_google(audio_data)
        return text.lower().strip()

    except sr.UnknownValueError:
        return "[ERROR] Could not understand the audio"

    except sr.RequestError as e:
        return f"[ERROR] Speech Recognition API error: {e}"


# ──────────────────────────────────────────────
# Step 3: Check if Transcribed Text Matches Passphrase
# ──────────────────────────────────────────────
def check_passphrase(transcribed_text, passphrase="open the gate"):
    """
    Compares the transcribed text with the expected passphrase
    using fuzzy matching (to tolerate small transcription errors).

    Args:
        transcribed_text (str): Text from speech recognition
        passphrase (str): Expected passphrase command

    Returns:
        tuple: (is_match: bool, similarity_ratio: float)
    """
    from difflib import SequenceMatcher

    # Clean both strings
    transcribed_clean = transcribed_text.lower().strip()
    passphrase_clean = passphrase.lower().strip()

    # Calculate similarity ratio (0.0 to 1.0)
    ratio = SequenceMatcher(None, transcribed_clean, passphrase_clean).ratio()

    # Match if similarity is above 60%
    is_match = ratio >= 0.6

    return is_match, ratio
