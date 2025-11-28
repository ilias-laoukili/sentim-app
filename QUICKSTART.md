# Sentim - Audio Emotion Analysis

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An interactive ML application for analyzing and manipulating emotions in speech audio using DSP and Random Forest classification.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run src/frontend/app.py
```

See [full documentation](README.md) for training, dataset setup, and advanced usage.

## Project Structure

- `src/backend/` - Core audio processing and ML models
- `src/frontend/` - Streamlit UI
- `scripts/` - Training and utility scripts
- `models/` - Trained model artifacts
- `data/` - Audio datasets (RAVDESS)

## Requirements

- Python 3.9-3.12 (Python 3.13 has numba compatibility issues)
- ffmpeg (for MP3/M4A support)
- 2 GB disk space for RAVDESS dataset

## Citation

```bibtex
@misc{sentim2025,
  author = {Laoukili, Ilias},
  title = {Sentim: AI-Powered Audio Emotion Analysis},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/ilias-laoukili/sentim-app}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
