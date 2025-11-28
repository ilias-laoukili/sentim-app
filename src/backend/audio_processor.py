"""
Audio Processing Module
Handles audio file loading, processing, and conversion operations.
"""

import os
import tempfile
import subprocess
from typing import Optional, Tuple, Union, Any

import librosa
import soundfile as sf
import numpy as np


def safe_unlink(path: Optional[str]) -> None:
    """Safely delete a file."""
    try:
        if path and os.path.exists(path):
            os.unlink(path)
    except Exception:
        pass


def _convert_to_wav_with_ffmpeg(
    input_path: str,
) -> Tuple[Optional[np.ndarray], Optional[int], Optional[bytes], Optional[str]]:
    """
    Convert audio file to WAV using ffmpeg.

    Args:
        input_path: Path to the input audio file.

    Returns:
        Tuple containing (audio_array, sample_rate, audio_bytes, error_message).
    """
    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_wav_path = tmp_wav.name
    tmp_wav.close()

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", input_path, tmp_wav_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        y, sr = librosa.load(tmp_wav_path, sr=None, mono=True)
        # Read converted wav bytes for playback
        with open(tmp_wav_path, "rb") as f:
            audio_bytes = f.read()
        safe_unlink(tmp_wav_path)
        return y, sr, audio_bytes, None
    except FileNotFoundError:
        safe_unlink(tmp_wav_path)
        return (
            None,
            None,
            None,
            "`ffmpeg` not found. Install ffmpeg to support MP3/M4A uploads, or upload WAV/FLAC files.",
        )
    except subprocess.CalledProcessError as cpe:
        stderr = (
            cpe.stderr.decode("utf-8") if hasattr(cpe, "stderr") and cpe.stderr is not None else ""
        )
        safe_unlink(tmp_wav_path)
        return None, None, None, f"ffmpeg failed to convert the uploaded file.\n{stderr}"
    except Exception as e:
        safe_unlink(tmp_wav_path)
        return None, None, None, f"Error during ffmpeg conversion: {str(e)}"


def load_audio_file(
    uploaded_bytes: bytes, filename: str
) -> Tuple[Optional[np.ndarray], Optional[int], Optional[bytes], Optional[str]]:
    """
    Load audio from uploaded bytes.

    Args:
        uploaded_bytes: The uploaded file bytes
        filename: Original filename

    Returns:
        tuple: (audio_array, sample_rate, audio_bytes_for_playback, error_message)
    """
    # Save uploaded file to a temporary file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix="_" + filename)
    tmp.write(uploaded_bytes)
    tmp.flush()
    tmp_path = tmp.name
    tmp.close()

    # Check file size
    try:
        file_size = os.path.getsize(tmp_path)
    except Exception:
        file_size = None

    # Reject obviously-broken uploads (very small files)
    if file_size is not None and file_size < 1024:
        safe_unlink(tmp_path)
        return (
            None,
            None,
            None,
            f"Uploaded file appears too small ({file_size} bytes). Please re-upload a valid audio file.",
        )

    # Try loading with librosa directly
    try:
        y, sr = librosa.load(tmp_path, sr=None, mono=True)
        safe_unlink(tmp_path)
        return y, sr, uploaded_bytes, None
    except Exception as exc:
        # Handle audioread backend missing or other format issues by attempting ffmpeg conversion
        try:
            import audioread

            is_no_backend = (
                isinstance(exc, audioread.NoBackendError)
                or exc.__class__.__name__ == "NoBackendError"
            )
        except Exception:
            is_no_backend = False

        if is_no_backend or isinstance(exc, Exception):
            result = _convert_to_wav_with_ffmpeg(tmp_path)
            safe_unlink(tmp_path)
            return result
        else:
            safe_unlink(tmp_path)
            return None, None, None, f"Could not load the uploaded audio: {exc}"


def process_audio(
    y: np.ndarray,
    sr: int,
    speed: float = 1.0,
    pitch: int = 0,
    carrier_freq: float = 0.0,
    echo_delay: float = 0.0,
    echo_decay: float = 0.5,
) -> Tuple[Optional[np.ndarray], Optional[bytes]]:
    """
    Apply DSP effects to audio.

    Args:
        y: Input audio array
        sr: Sample rate
        speed: Time-stretch rate
        pitch: Pitch shift in semitones
        carrier_freq: Robotization carrier frequency
        echo_delay: Echo delay time in milliseconds (0 = off)
        echo_decay: Echo decay factor (0.0 to 1.0)

    Returns:
        tuple: (processed_audio_array, processed_audio_bytes) or (None, None) on error
    """
    try:
        from src.backend import dsp_utils

        # Start from original
        y_proc = np.copy(y)

        # Apply pitch shift first
        if pitch != 0:
            y_proc = dsp_utils.pitch_shift(y_proc, n_steps=pitch)

        # Apply time-stretch (speed)
        if speed != 1.0:
            y_proc = dsp_utils.time_stretch(y_proc, rate=speed)

        # Apply robotization if carrier_freq > 0
        if carrier_freq > 0.0:
            y_proc = dsp_utils.robotize(y_proc, rate=1.0, carrier_freq=carrier_freq, sr=sr)

        # Apply echo effect if echo_delay > 0
        if echo_delay > 0.0:
            y_proc = dsp_utils.echo(y_proc, sr, delay_ms=echo_delay, decay=echo_decay)

        # Ensure dtype float32 and normalize
        y_proc = np.asarray(y_proc, dtype=np.float32)
        max_val = np.max(np.abs(y_proc))
        if max_val > 1.0:
            y_proc = y_proc / max_val

        # Write to temporary file and read bytes
        try:
            tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tmp_out_path = tmp_out.name
            tmp_out.close()
            sf.write(tmp_out_path, y_proc, sr, format="WAV")
            with open(tmp_out_path, "rb") as f:
                audio_bytes = f.read()
            safe_unlink(tmp_out_path)
            return y_proc, audio_bytes
        except Exception:
            # Fallback: try writing to an in-memory buffer
            import io

            buf = io.BytesIO()
            try:
                sf.write(buf, y_proc, sr, format="WAV")
                audio_bytes = buf.getvalue()
                return y_proc, audio_bytes
            except Exception:
                return None, None
    except Exception as e:
        return None, None


def encode_audio_for_playback(audio_array: np.ndarray, sr: int) -> Optional[bytes]:
    """
    Encode audio array to WAV bytes for playback.

    Args:
        audio_array: Audio data as numpy array
        sr: Sample rate

    Returns:
        bytes or None
    """
    try:
        # Normalize
        audio_norm = np.asarray(audio_array, dtype=np.float32)
        max_val = np.max(np.abs(audio_norm))
        if max_val > 1.0:
            audio_norm = audio_norm / max_val
        elif max_val > 0 and max_val < 0.1:
            # Scale up very quiet audio
            audio_norm = audio_norm / max_val

        # Write to temporary file
        tmp_synth = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp_synth_path = tmp_synth.name
        tmp_synth.close()

        sf.write(tmp_synth_path, audio_norm, sr, format="WAV")
        with open(tmp_synth_path, "rb") as f:
            audio_bytes = f.read()
        safe_unlink(tmp_synth_path)
        return audio_bytes
    except Exception:
        return None
