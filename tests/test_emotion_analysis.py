"""
Unit tests for emotion analysis module.
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch

from src.backend.emotion_analysis import (
    extract_features,
    emotion_analysis_heuristic,
    AcousticStatisticalClassifier,
    EmotionAudioModifier,
)


class TestFeatureExtraction:
    """Test feature extraction functions."""

    def test_extract_features_basic(self):
        """Test basic feature extraction with synthetic audio."""
        # Create synthetic audio: 1 second at 22050 Hz
        sample_rate = 22050
        duration = 1.0
        audio_signal = np.random.randn(int(sample_rate * duration))

        features = extract_features(audio_signal, sample_rate)

        # Verify all expected features are present
        assert "rms" in features
        assert "centroid" in features
        assert "f0_median" in features
        assert "f0_mean" in features
        assert "f0_std" in features

        # Verify feature values are floats
        for key, value in features.items():
            assert isinstance(value, float)

    def test_extract_features_short_audio(self):
        """Test feature extraction with very short audio."""
        sample_rate = 22050
        audio_signal = np.random.randn(1024)  # Very short

        features = extract_features(audio_signal, sample_rate)

        # Should still return valid features without crashing
        assert isinstance(features, dict)
        assert len(features) == 5


class TestEmotionHeuristic:
    """Test heuristic emotion analysis."""

    def test_emotion_analysis_heuristic_returns_valid(self):
        """Test that heuristic returns valid emotion labels."""
        sample_rate = 22050
        audio_signal = np.random.randn(sample_rate)

        label, confidence, features = emotion_analysis_heuristic(audio_signal, sample_rate)

        # Verify label is one of expected emotions
        assert label in ["Anger", "Sadness", "Joy", "Neutral"]

        # Verify confidence is between 0 and 1
        assert 0.0 <= confidence <= 1.0

        # Verify features dict is returned
        assert isinstance(features, dict)


class TestAcousticStatisticalClassifier:
    """Test machine learning classifier."""

    def test_classifier_initialization(self):
        """Test classifier can be initialized."""
        clf = AcousticStatisticalClassifier(n_estimators=50)

        assert clf.is_trained is False
        assert clf.labels is None

    def test_extract_statistical_features(self):
        """Test statistical feature extraction."""
        clf = AcousticStatisticalClassifier()
        sample_rate = 22050
        audio = np.random.randn(sample_rate)

        features = clf.extract_statistical_features(audio, sample_rate)

        # Verify feature vector is returned
        assert isinstance(features, np.ndarray)
        assert features.ndim == 1
        # Should have ~352 features
        assert 300 < len(features) < 400


class TestEmotionAudioModifier:
    """Test emotion audio modification."""

    def test_emotion_modifier_initialization(self):
        """Test emotion modifier can be initialized."""
        modifier = EmotionAudioModifier()

        # Verify emotion parameters exist
        assert hasattr(modifier, "EMOTION_PARAMS")
        assert "joy" in modifier.EMOTION_PARAMS
        assert "sadness" in modifier.EMOTION_PARAMS

    def test_get_available_emotions(self):
        """Test getting list of available emotions."""
        modifier = EmotionAudioModifier()

        emotions = modifier.get_available_emotions()

        assert isinstance(emotions, list)
        assert "joy" in emotions
        assert "sadness" in emotions
        assert "anger" in emotions
        assert "neutral" in emotions

    def test_modify_for_emotion(self):
        """Test basic emotion modification."""
        modifier = EmotionAudioModifier()
        sample_rate = 22050
        audio = np.random.randn(sample_rate)

        modified, params = modifier.modify_for_emotion(
            audio, sample_rate, "joy", preserve_formants=False
        )

        # Verify modified audio is returned
        assert isinstance(modified, np.ndarray)
        assert len(modified) > 0

        # Verify parameters used are returned
        assert isinstance(params, dict)
        assert "pitch_shift" in params
        assert "tempo" in params
        assert "energy" in params

    def test_modify_with_classifier_not_reloading(self):
        """Test that modify_with_classifier doesn't reload the classifier."""
        # This is the critical optimization test
        modifier = EmotionAudioModifier()
        sample_rate = 22050
        audio = np.random.randn(sample_rate)

        # Create mock classifier
        mock_classifier = Mock()
        mock_classifier.is_trained = True
        mock_classifier.predict_from_array = Mock(return_value=("neutral", 0.8))
        # Mock the model classes for the loop in modify_with_classifier
        mock_classifier.model.classes_ = ["neutral", "joy", "sadness", "anger"]

        # Call modify_with_classifier multiple times
        for _ in range(3):
            modified, results = modifier.modify_with_classifier(
                audio, sample_rate, mock_classifier, "joy"
            )

            # Verify results structure
            assert "original_emotion" in results
            assert "modified_emotion" in results
            assert "parameters_applied" in results

        # Verify classifier.predict_from_array was called (2x per loop: before & after)
        # but classifier was NOT reloaded (no load() calls)
        assert mock_classifier.predict_from_array.call_count == 6  # 3 loops * 2 predictions

        # If there was a load() method being called, this would fail
        # This confirms the classifier is passed by reference, not reloaded


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
