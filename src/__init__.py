"""
Sentim - Audio Emotion Analysis Package
"""

__version__ = "1.0.0"
__author__ = "Ilias Laoukili"

# Make modules accessible at package level
from src.backend import audio_processor, dsp_utils, emotion_analysis
from src.frontend import app, ui_components, ui_styles

__all__ = [
    "audio_processor",
    "dsp_utils",
    "emotion_analysis",
    "app",
    "ui_components",
    "ui_styles",
]
