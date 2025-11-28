"""
Emotion Analysis Module

Provides emotion recognition and transformation capabilities using:
- Heuristic-based emotion analysis from audio features
- Machine learning classification with Random Forest
- Prosody-based emotion transformation
"""

import numpy as np
import joblib
import librosa
import librosa.display
import matplotlib
from typing import Dict, Tuple, List, Optional, Any, Union

from scipy import signal as scipy_signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# Set backend to Agg before importing pyplot to avoid GUI issues on servers
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from .dsp_utils import pitch_shift, time_stretch


def extract_features(audio_signal, sample_rate) -> Dict[str, float]:
    """
    Extract classic DSP features for emotion heuristics.

    Args:
        audio_signal: Audio time-series data
        sample_rate: Sample rate of the audio

    Returns:
        dict: Extracted features (rms, centroid, f0_median, f0_mean, f0_std)
    """
    y = np.asarray(audio_signal, dtype=float)

    # RMS Energy
    hop_length = 512
    frame_length = 2048
    # Librosa returns shape (1, t), we take [0] to get 1D array
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_mean = float(np.mean(rms))

    # Spectral Centroid (Brightness)
    # Ensure n_fft is not larger than the signal itself
    n_fft_safe = min(2048, y.size)
    win_length_safe = min(1024, y.size)
    S = librosa.stft(y, n_fft=n_fft_safe, hop_length=hop_length, win_length=win_length_safe)
    mag = np.abs(S)
    # Pass S to spectral_centroid to avoid re-computing STFT
    centroid = librosa.feature.spectral_centroid(S=mag, sr=sample_rate)[0]
    centroid_mean = float(np.mean(centroid))

    # Pitch (F0) using YIN
    fmin, fmax = 50.0, 500.0
    try:
        # For very low sample rates, reduce frame_length
        if sample_rate < 8000 and frame_length > y.size:
            frame_length = 1024 if y.size >= 1024 else 512

        f0 = librosa.yin(
            y,
            fmin=fmin,
            fmax=fmax,
            sr=sample_rate,
            frame_length=frame_length,
            hop_length=hop_length,
        )
        # Filter out unvoiced segments (NaNs)
        f0_valid = f0[~np.isnan(f0)]

        if f0_valid.size == 0:
            f0_median, f0_mean, f0_std = 0.0, 0.0, 0.0
        else:
            f0_median = float(np.median(f0_valid))
            f0_mean = float(np.mean(f0_valid))
            f0_std = float(np.std(f0_valid))
    except Exception:
        f0_median, f0_mean, f0_std = 0.0, 0.0, 0.0

    return {
        "rms": rms_mean,
        "centroid": centroid_mean,
        "f0_median": f0_median,
        "f0_mean": f0_mean,
        "f0_std": f0_std,
    }


def emotion_analysis_heuristic(audio_signal, sample_rate) -> Tuple[str, float, Dict[str, float]]:
    """
    Baseline heuristic emotion predictor using acoustic features.

    Args:
        audio_signal: Audio time-series data
        sample_rate: Sample rate of the audio

    Returns:
        tuple: (emotion_label, confidence, extracted_features)
    """
    feats = extract_features(audio_signal, sample_rate)

    def normalize(x, lo, hi) -> float:
        """Normalize value to [0, 1] range."""
        if hi <= lo:
            return 0.0
        return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))

    # Note: These thresholds assume normalized input audio
    energy = normalize(feats["rms"], 0.005, 0.06)
    brightness = normalize(feats["centroid"], 800.0, 4500.0)
    pitch = normalize(feats["f0_median"], 80.0, 350.0)
    variability = normalize(feats["f0_std"], 5.0, 120.0)

    # Emotion scoring based on acoustic correlations
    anger_score = 0.35 * energy + 0.25 * pitch + 0.25 * variability + 0.15 * brightness
    sadness_score = (
        0.45 * (1 - energy) + 0.25 * (1 - pitch) + 0.2 * (1 - brightness) + 0.1 * (1 - variability)
    )
    joy_score = 0.4 * energy + 0.3 * pitch + 0.2 * brightness + 0.1 * (variability * 0.5)
    neutral_score = (
        0.5 * (1 - abs(energy - 0.5) * 2) + 0.3 * (1 - variability) + 0.2 * (1 - abs(pitch - 0.5))
    )

    scores = {
        "Anger": float(anger_score),
        "Sadness": float(sadness_score),
        "Joy": float(joy_score),
        "Neutral": float(neutral_score),
    }

    label = max(scores, key=scores.get)

    # Softmax-style confidence
    score_vals = np.array(list(scores.values()), dtype=float)
    exp_scores = np.exp(score_vals)
    probs = exp_scores / np.sum(exp_scores)
    confidence = float(probs[list(scores.keys()).index(label)])

    return label, confidence, feats


def plot_spectrogram(
    audio_signal,
    sample_rate,
    n_fft=2048,
    hop_length=None,
    win_length=None,
    cmap="magma",
    show=False,
    save_path=None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Generate and plot a spectrogram of the audio signal.

    Args:
        audio_signal: Audio time-series data
        sample_rate: Sample rate of the audio
        n_fft: FFT window size
        hop_length: Number of samples between frames
        win_length: Window length
        cmap: Matplotlib colormap name
        show: Whether to display the plot
        save_path: Path to save the figure

    Returns:
        tuple: (figure, axes) matplotlib objects
    """
    if hop_length is None:
        hop_length = win_length // 4 if win_length is not None else n_fft // 4

    S = librosa.stft(audio_signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

    if cmap is None:
        cmap = "viridis"

    fig, ax = plt.subplots(figsize=(8, 4), facecolor="white")
    ax.set_facecolor("white")
    img = librosa.display.specshow(
        S_db,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis="time",
        y_axis="hz",
        cmap=cmap,
        ax=ax,
    )
    ax.set_title("Spectrogram (dB)")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax


class AcousticStatisticalClassifier:
    """
    Random Forest classifier for emotion recognition from acoustic features.

    Extracts statistical features from audio and trains
    a Random Forest model for emotion classification.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        random_state: Optional[int] = None,
        use_grid_search: bool = False,
    ):
        self.base_model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced",
        )
        self.model = self.base_model
        self.scaler = StandardScaler()
        self.is_trained = False
        self.labels: Optional[List[str]] = None
        self.use_grid_search = use_grid_search

    def extract_statistical_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract statistical features from audio signal.
        """
        y = np.asarray(y, dtype=float)
        if y.ndim > 1:
            y = librosa.to_mono(y)

        y, _ = librosa.effects.trim(y, top_db=30)

        # Ensure minimum length (1 second) for complex features
        min_length = sr
        if y.size < min_length:
            y = np.pad(y, (0, max(0, min_length - y.size)), mode="constant")

        # Safety pad for very short clips
        if y.size < 2048:
            y = np.pad(y, (0, 2048 - y.size), mode="constant")

        n_fft = min(2048, y.size)

        mfcc_mean = self._extract_mfcc(y, sr, n_fft)
        mel_features = self._extract_mel_spectrogram(y, sr, n_fft)
        chroma_tonnetz = self._extract_chroma_tonnetz(y, sr, n_fft)
        spectral_features = self._extract_spectral_features(y, sr, n_fft)
        prosody_features = self._extract_prosody(y)

        return np.hstack(
            [
                mfcc_mean,
                mel_features,
                chroma_tonnetz,
                spectral_features,
                prosody_features,
            ]
        ).astype(float)

    def _extract_mfcc(self, y: np.ndarray, sr: int, n_fft: int) -> np.ndarray:
        # Keep only MFCC means to limit feature dimensionality (~40)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=n_fft)
        return np.mean(mfcc, axis=1)

    def _extract_mel_spectrogram(self, y: np.ndarray, sr: int, n_fft: int) -> np.ndarray:
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_mean = np.mean(mel_db, axis=1)
        mel_std = np.std(mel_db, axis=1)
        return np.hstack([mel_mean, mel_std])

    def _extract_chroma_tonnetz(self, y: np.ndarray, sr: int, n_fft: int) -> np.ndarray:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft)
        chroma_mean = np.mean(chroma, axis=1)

        try:
            tonnetz = librosa.feature.tonnetz(chroma=chroma)
            tonnetz_mean = np.mean(tonnetz, axis=1)
        except Exception:
            tonnetz_mean = np.zeros(6)

        return np.hstack([chroma_mean, tonnetz_mean])

    def _extract_spectral_features(self, y: np.ndarray, sr: int, n_fft: int) -> np.ndarray:
        cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft)[0]
        cent_mean = float(np.mean(cent))
        cent_std = float(np.std(cent))

        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft)[0]
        rolloff_mean = float(np.mean(rolloff))
        rolloff_std = float(np.std(rolloff))

        n_bands = 6
        nyquist = sr / 2.0
        min_required_fmin = nyquist / (2**n_bands)
        use_fmin = min(200.0, min_required_fmin * 0.95)
        use_fmin = max(10.0, use_fmin)

        try:
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=n_bands, fmin=use_fmin)
        except Exception:
            contrast = np.zeros((n_bands + 1, 1))
        contrast_mean = np.mean(contrast, axis=1)

        return np.hstack([cent_mean, cent_std, rolloff_mean, rolloff_std, contrast_mean])

    def _extract_prosody(self, y: np.ndarray) -> np.ndarray:
        rms = librosa.feature.rms(y=y)[0]
        rms_mean = float(np.mean(rms))

        zcr = librosa.feature.zero_crossing_rate(y)[0]
        zcr_mean = float(np.mean(zcr))

        return np.array([rms_mean, zcr_mean], dtype=float)

    def train(self, audio_paths: List[str], labels: List[str]) -> None:
        """
        Train the classifier from audio file paths.
        """
        if len(audio_paths) != len(labels):
            raise ValueError("Mismatch in paths and labels length")

        X = []
        for p in audio_paths:
            y, sr = librosa.load(p, sr=None, mono=True)
            feat = self.extract_statistical_features(y, sr)
            X.append(feat)

        self._fit(np.vstack(X), labels)

    def train_from_arrays(
        self, audio_arrays: List[np.ndarray], sample_rates: List[int], labels: List[str]
    ) -> None:
        """
        Train the classifier from audio arrays.
        """
        if not (len(audio_arrays) == len(sample_rates) == len(labels)):
            raise ValueError("Mismatch in inputs length")

        X = []
        for y, sr in zip(audio_arrays, sample_rates):
            feat = self.extract_statistical_features(y, sr)
            X.append(feat)

        self._fit(np.vstack(X), labels)

    def _fit(self, X: np.ndarray, labels: List[str]) -> None:
        y_labels = np.array(labels)
        X_scaled = self.scaler.fit_transform(X)

        if self.use_grid_search:
            print("Performing grid search for best hyperparameters...")
            param_grid = {
                "n_estimators": [100, 200, 300],
                "max_depth": [10, 20, None],
                "min_samples_leaf": [1, 2, 4],
                "min_samples_split": [2, 5],
            }

            rf = RandomForestClassifier(
                random_state=self.base_model.random_state, class_weight="balanced"
            )

            grid_search = GridSearchCV(
                estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2
            )
            grid_search.fit(X_scaled, y_labels)

            print(f"Best parameters found: {grid_search.best_params_}")
            self.model = grid_search.best_estimator_
        else:
            self.model.fit(X_scaled, y_labels)

        self.is_trained = True
        self.labels = list(self.model.classes_)

    def predict_from_array(self, y: np.ndarray, sr: int) -> Tuple[str, float]:
        """
        Predict emotion from audio array.
        """
        if not self.is_trained:
            raise NotFittedError("Model not trained.")

        feat = self.extract_statistical_features(y, sr).reshape(1, -1)
        feat_scaled = self.scaler.transform(feat)
        probs = self.model.predict_proba(feat_scaled)[0]
        idx = int(np.argmax(probs))
        return str(self.model.classes_[idx]), float(probs[idx])

    def predict(self, audio_path: str) -> Tuple[str, float]:
        """
        Predict emotion from audio file.
        """
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        return self.predict_from_array(y, sr)

    def save(self, path: str) -> None:
        """Save trained model to disk."""
        joblib.dump({"model": self.model, "scaler": self.scaler, "labels": self.labels}, path)

    def load(self, path: str) -> None:
        """Load trained model from disk."""
        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.labels = data.get("labels")
        self.is_trained = True


class EmotionAudioModifier:
    """
    Modify audio to convey different emotions using prosody manipulation.
    """

    EMOTION_PARAMS = {
        "joy": {"pitch_shift": 3, "tempo": 1.15, "energy": 1.3},
        "happiness": {"pitch_shift": 3, "tempo": 1.15, "energy": 1.3},
        "sadness": {"pitch_shift": -3, "tempo": 0.85, "energy": 0.7},
        "anger": {"pitch_shift": 2, "tempo": 1.2, "energy": 1.5},
        "fear": {"pitch_shift": 4, "tempo": 1.25, "energy": 1.2},
        "surprise": {"pitch_shift": 5, "tempo": 1.1, "energy": 1.4},
        "neutral": {"pitch_shift": 0, "tempo": 1.0, "energy": 1.0},
    }

    def modify_for_emotion(
        self,
        audio: np.ndarray,
        sample_rate: int,
        target_emotion: str,
        preserve_formants: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Modify audio to match target emotion using prosody manipulation.
        """
        target_emotion = target_emotion.lower()
        params = self.EMOTION_PARAMS.get(target_emotion, self.EMOTION_PARAMS["neutral"])

        y = np.asarray(audio, dtype=float).copy()

        if params["pitch_shift"] != 0:
            y = librosa.effects.pitch_shift(y, sr=sample_rate, n_steps=params["pitch_shift"])

        if params["tempo"] != 1.0:
            y = librosa.effects.time_stretch(y, rate=params["tempo"])

        if params["energy"] != 1.0:
            y = y * params["energy"]

        if preserve_formants:
            y = self._preserve_formants(y, sample_rate)

        y = np.clip(y, -1.0, 1.0)

        return y, params

    def _preserve_formants(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Maintain voice quality during modifications using bandpass filtering.
        """
        try:
            nyquist = sample_rate / 2.0
            low_freq = min(80.0, nyquist * 0.8)
            high_freq = min(8000.0, nyquist * 0.95)

            if low_freq < high_freq:
                sos = scipy_signal.butter(
                    4, [low_freq, high_freq], btype="band", fs=sample_rate, output="sos"
                )
                audio_filtered = scipy_signal.sosfilt(sos, audio)
                return audio_filtered
            else:
                return audio
        except Exception:
            return audio

    def modify_with_classifier(
        self,
        audio: np.ndarray,
        sample_rate: int,
        classifier: AcousticStatisticalClassifier,
        target_emotion: str,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Modify audio toward target emotion using Grid Search optimization.
        """
        if not classifier.is_trained:
            raise NotFittedError("Classifier must be trained before using modify_with_classifier")

        pitch_options = [-4, -2, 0, 2, 4]
        speed_options = [0.8, 1.0, 1.2]

        best_audio = audio
        best_prob = -1.0
        best_params = {"pitch_shift": 0, "time_stretch": 1.0}

        orig_label, orig_conf = classifier.predict_from_array(audio, sample_rate)

        target_idx = -1
        for i, cls_name in enumerate(classifier.model.classes_):
            if cls_name.lower() == target_emotion.lower():
                target_idx = i
                break

        if target_idx == -1:
            return audio, {
                "original_emotion": orig_label,
                "original_confidence": orig_conf,
                "modified_emotion": orig_label,
                "modified_confidence": orig_conf,
                "target_emotion": target_emotion,
                "parameters_applied": best_params,
                "error": "Target emotion not supported by classifier",
            }

        for p in pitch_options:
            for s in speed_options:
                # Note: pitch_shift in dsp_utils preserves duration, so we can apply time_stretch after
                try:
                    y_pitch = pitch_shift(audio, n_steps=p)
                    y_mod = time_stretch(y_pitch, rate=s)

                    feat = classifier.extract_statistical_features(y_mod, sample_rate).reshape(
                        1, -1
                    )
                    feat_scaled = classifier.scaler.transform(feat)
                    probs = classifier.model.predict_proba(feat_scaled)[0]

                    current_prob = probs[target_idx]

                    if current_prob > best_prob:
                        best_prob = current_prob
                        best_audio = y_mod
                        best_params = {"pitch_shift": p, "time_stretch": s}
                except Exception:
                    continue

        final_label, final_conf = classifier.predict_from_array(best_audio, sample_rate)

        results = {
            "original_emotion": orig_label,
            "original_confidence": orig_conf,
            "modified_emotion": final_label,
            "modified_confidence": final_conf,
            "target_emotion": target_emotion,
            "parameters_applied": best_params,
            "optimization_score": float(best_prob),
        }

        return best_audio, results

    def get_available_emotions(self) -> List[str]:
        """Return list of supported emotion labels."""
        return list(self.EMOTION_PARAMS.keys())


__all__ = [
    "extract_features",
    "emotion_analysis_heuristic",
    "plot_spectrogram",
    "AcousticStatisticalClassifier",
    "EmotionAudioModifier",
]
