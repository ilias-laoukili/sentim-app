import numpy as np
import librosa
from typing import Optional, Tuple, Union, Callable, Any

# Streamlit caching: use `st.cache_data` when Streamlit is available, otherwise no-op.
try:
    import streamlit as st

    cache = st.cache_data
except Exception:

    def cache(func: Callable) -> Callable:
        return func


def _hann_window(M: int, sym: bool = True) -> np.ndarray:
    """
    Numpy implementation of a Hann window, similar to scipy.signal.windows.hann.
    `sym=False` is used for signal processing applications (periodic).
    """
    if M < 1:
        return np.array([])
    if M == 1:
        return np.ones(1, dtype=float)

    if sym:
        # This case is not used in the project but included for completeness
        return 0.5 * (1 - np.cos(2 * np.pi * np.arange(M) / (M - 1)))
    else:  # Periodic window
        return 0.5 * (1 - np.cos(2 * np.pi * np.arange(M) / M))


@cache
def compute_stft(
    y: np.ndarray,
    n_fft: int = 2048,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    center: bool = True,
    pad_mode: str = "reflect",
    return_mag_phase: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Compute the Short-Time Fourier Transform (STFT) of a real-valued audio signal using a Hann window.

    This wrapper uses `librosa.stft` for stability but constructs a Hann window via
    `scipy.signal.windows.hann` so the returned complex spectrogram is safe for phase
    manipulation (phase vocoder style processing).

    Parameters
    - y: time-domain audio signal (1D numpy array)
    - n_fft: number of FFT bins (frame size). The resulting spectrogram has shape `(n_fft//2+1, n_frames)`.
    - hop_length: number of samples between successive frames. If `None`, defaults to `win_length // 4`.
    - win_length: length of the analysis window in samples. If `None`, defaults to `n_fft`.
    - center: whether to pad `y` so that frames are centered (passed to `librosa.stft`).
    - pad_mode: padding mode for `librosa.stft` when `center=True`.
    - return_mag_phase: if True, also return magnitude and phase arrays.

    Returns
    - If `return_mag_phase` is False: complex STFT matrix `S` (shape `(n_fft//2+1, n_frames)`).
    - If True: tuple `(S, magnitude, phase)` where `magnitude = abs(S)` and `phase = angle(S)`.

    Notes
    - Returning the complex STFT allows direct phase manipulation (e.g., for phase vocoder).
    """
    if win_length is None:
        win_length = n_fft
    if hop_length is None:
        hop_length = max(1, win_length // 4)

    window = _hann_window(win_length, sym=False)
    S = librosa.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
    )
    if return_mag_phase:
        magnitude = np.abs(S)
        phase = np.angle(S)
        return S, magnitude, phase
    return S


def compute_istft(
    S: np.ndarray,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    center: bool = True,
    length: Optional[int] = None,
) -> np.ndarray:
    """
    Reconstruct a time-domain signal from a complex STFT matrix using the inverse STFT.

    Parameters
    - S: complex STFT (shape `(n_fft//2+1, n_frames)`).
    - hop_length: hop length used during STFT. If `None`, defaults to `win_length // 4`.
    - win_length: window length used during STFT. If `None`, defaults to `n_fft` inferred from `S`.
    - center: same `center` semantics as `librosa.istft`.
    - length: optionally force the length of the output signal (useful when trimming/padding to original length).

    Returns
    - y: reconstructed time-domain signal (1D numpy array)

    Notes
    - This function infers `n_fft` from the number of rows in `S` when `win_length` is not provided.
    - The Hann window used here matches `compute_stft` so overlap-add reconstruction is consistent.
    """
    # Infer n_fft from the number of frequency bins
    n_fft = (S.shape[0] - 1) * 2
    if win_length is None:
        win_length = n_fft
    if hop_length is None:
        hop_length = max(1, win_length // 4)

    window = _hann_window(win_length, sym=False)
    y = librosa.istft(
        S,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        length=length,
    )
    return y


def _resample_fft(x: np.ndarray, num: int) -> np.ndarray:
    """
    Resamples a signal `x` to `num` samples using the FFT method.
    This is a numpy-based equivalent to scipy.signal.resample.
    """
    if x.size == 0:
        return np.array([], dtype=float)

    # Forward FFT
    X = np.fft.rfft(x)

    # Create new frequency array and zero-pad or truncate
    new_X = np.zeros(num // 2 + 1, dtype=X.dtype)
    n = min(len(X), len(new_X))
    new_X[:n] = X[:n]

    return np.fft.irfft(new_X, n=num) * (num / len(x))


@cache
def time_stretch(
    y: np.ndarray,
    rate: float,
    n_fft: int = 2048,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
) -> np.ndarray:
    """
    Time-stretch an audio signal using the Phase Vocoder algorithm without changing pitch.

    Parameters
    - y: time-domain audio signal (1D numpy array)
    - rate: stretch factor. Values > 1 speed up (shorten) the signal, values < 1 slow down (lengthen).
    - n_fft, hop_length, win_length: STFT parameters. If `hop_length` or `win_length` are None,
      reasonable defaults are chosen (win_length = n_fft, hop_length = win_length // 4).

    Returns
    - y_stretched: time-domain stretched signal

    Implementation notes
    - We compute the complex STFT using `compute_stft`, then apply `librosa.phase_vocoder`
      which performs phase-advance correction to maintain phase coherence between frames.
    - Finally, we reconstruct with `compute_istft` using the same window/hop settings.
    """
    if win_length is None:
        win_length = n_fft
    if hop_length is None:
        hop_length = max(1, win_length // 4)

    S = compute_stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=True)
    S_stretched = librosa.phase_vocoder(S, rate=rate, hop_length=hop_length)
    target_length = None
    try:
        target_length = int(np.round(len(y) / rate))
    except Exception:
        target_length = None
    y_stretched = compute_istft(
        S_stretched, hop_length=hop_length, win_length=win_length, length=target_length
    )
    return y_stretched


@cache
def pitch_shift(
    y: np.ndarray,
    n_steps: float,
    n_fft: int = 2048,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
) -> np.ndarray:
    """
    Pitch-shift an audio signal by `n_steps` semitones without changing duration.

    Algorithm (per project guidelines):
    1. Compute a time-stretched version of the signal with factor = 1 / (2**(n_steps/12)).
    2. Resample the time-stretched signal back to the original number of samples to restore
       the original duration while changing pitch.

    Parameters
    - y: input time-domain signal
    - n_steps: number of semitones to shift (positive raises pitch, negative lowers)
    - n_fft, hop_length, win_length: passed to the internal `time_stretch` call

    Returns
    - y_shifted: pitch-shifted signal with the same length as `y` (approximately)
    """
    # pitch factor (frequency multiplier)
    factor = 2.0 ** (n_steps / 12.0)

    # time-stretch by reciprocal of factor (makes signal longer for upward shifts)
    rate = 1.0 / factor

    y_ts = time_stretch(y, rate=rate, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

    try:
        target_len = len(y)
        y_shifted = _resample_fft(y_ts, target_len)
    except Exception:
        y_shifted = y_ts

    return np.real(y_shifted)


@cache
def robotize(
    y: np.ndarray,
    rate: Optional[float],
    carrier_freq: float,
    sr: Optional[int] = None,
    n_fft: int = 2048,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
) -> np.ndarray:
    """
    Robotize (ring-modulate) an audio signal.

    Implementation:
    - Optionally time-stretch the input using the phase vocoder with `rate`.
    - Multiply the (possibly time-stretched) signal by a complex exponential
      `exp(-j*2*pi*fc*t)` where `fc` is `carrier_freq`.
    - Return the real part of the modulated signal.

    Parameters
    - y: input time-domain signal
    - rate: time-stretch rate passed to `time_stretch`. Use `1.0` to skip stretching.
    - carrier_freq: carrier frequency in Hz (if `sr` provided) or in cycles-per-sample (if `sr` is None)
    - sr: sample rate in Hz. If provided, `carrier_freq` is interpreted in Hz; if None,
          `carrier_freq` is interpreted as cycles-per-sample.
    - n_fft, hop_length, win_length: forwarded to `time_stretch` if used

    Returns
    - y_robot: real-valued robotized signal
    """
    # Optionally time-stretch first
    if rate is None:
        rate = 1.0
    if rate != 1.0:
        y_proc = time_stretch(
            y, rate=rate, n_fft=n_fft, hop_length=hop_length, win_length=win_length
        )
    else:
        y_proc = y

    n = len(y_proc)
    if sr is None:
        t = np.arange(n)
        omega = 2.0 * np.pi * carrier_freq * t
    else:
        t = np.arange(n) / float(sr)
        omega = 2.0 * np.pi * carrier_freq * t

    carrier = np.cos(omega)
    y_mod = y_proc * carrier
    return y_mod


@cache
def echo(y: np.ndarray, sr: int, delay_ms: float = 250.0, decay: float = 0.5) -> np.ndarray:
    """
    Apply an echo (delay) effect to an audio signal.

    Implementation:
    - Creates a delayed copy of the signal attenuated by the decay factor.
    - Adds the delayed signal to the original to create the echo effect.
    - Multiple echoes naturally occur due to feedback in the delay line.

    Parameters
    - y: input time-domain signal (1D numpy array)
    - sr: sample rate in Hz
    - delay_ms: delay time in milliseconds (default: 250ms)
    - decay: attenuation factor for the echo (0.0 to 1.0, default: 0.5)
          Values closer to 1.0 create more pronounced and longer-lasting echoes.

    Returns
    - y_echo: signal with echo effect applied
    """
    y = np.asarray(y, dtype=float)

    # Convert delay from milliseconds to samples
    delay_samples = int((delay_ms / 1000.0) * sr)

    # Ensure delay is at least 1 sample
    delay_samples = max(1, delay_samples)

    # Create output array with extra space for the echo tail
    output_length = len(y) + delay_samples
    y_echo = np.zeros(output_length, dtype=float)

    # Copy original signal
    y_echo[: len(y)] = y

    # Add delayed (echoed) signal
    y_echo[delay_samples : delay_samples + len(y)] += y * decay

    # Normalize to prevent clipping
    max_val = np.max(np.abs(y_echo))
    if max_val > 1.0:
        y_echo = y_echo / max_val

    return y_echo


__all__ = ["compute_stft", "compute_istft", "time_stretch", "pitch_shift", "robotize", "echo"]
