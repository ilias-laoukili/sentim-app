# Overview

Sentim is a Streamlit application that combines digital signal processing (DSP) and
machine learning to classify emotions in speech audio and manipulate prosody.

## Quick start

1. Install runtime dependencies: `pip install -r requirements.txt`
2. Launch the app: `streamlit run src/frontend/app.py`
3. Upload an audio file (WAV/MP3/M4A/OGG) and view the detected emotion, confidence, and
   acoustic visualizations.

## Training

Run `python scripts/train_model.py` after placing the RAVDESS dataset under
`data/raw/audio_speech_actors_01-24/` to retrain the Random Forest classifier.

## Building the docs

```bash
pip install -r docs/requirements.txt
make -C docs html
open docs/_build/html/index.html
```

The `conf.py` file is configured to autodoc modules from the `src` package and render
both reStructuredText and MyST Markdown sources.
