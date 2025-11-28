import streamlit as st
import sys
from pathlib import Path
from typing import Optional

# Ensure project root on sys.path so `src` is importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.frontend.ui_styles import apply_styles
from src.frontend.ui_components import (
    hero_section,
    how_it_works_section,
    analysis_dashboard_section,
    app_footer,
)
from src.backend import emotion_analysis, dsp_utils


@st.cache_resource
def get_classifier(models_dir: Path) -> emotion_analysis.AcousticStatisticalClassifier:
    """Load and cache the trained emotion classifier.

    Strategy:
    1. Try local file `models/ravdess_model.joblib`.
    2. If missing, attempt download from Hugging Face Hub (public or with token).
       - Uses repo id from env var `HF_MODEL_REPO` (default: `ilias-laoukili/ravdess-emotion-model`).
       - Uses file name from env var `HF_MODEL_FILENAME` (default: `ravdess_model.joblib`).
    3. Hydrate classifier object from loaded joblib dict.
    """
    import os
    import joblib
    from pathlib import Path as _Path

    clf = emotion_analysis.AcousticStatisticalClassifier()

    local_path = models_dir / "ravdess_model.joblib"
    if not local_path.exists():
        repo_id = os.getenv("HF_MODEL_REPO", "ilias-laoukili/ravdess-emotion-model")
        filename = os.getenv("HF_MODEL_FILENAME", "ravdess_model.joblib")
        try:
            from huggingface_hub import hf_hub_download

            downloaded = hf_hub_download(
                repo_id=repo_id, filename=filename, token=os.getenv("HF_TOKEN")
            )
            local_path = _Path(downloaded)
        except Exception:
            pass  # Will fallback to not-fitted classifier

    try:
        data = joblib.load(str(local_path))
        if data and all(k in data for k in ("model", "scaler")):
            clf.model = data["model"]
            clf.scaler = data["scaler"]
            clf.labels = data.get("labels")
            clf.is_trained = True
        else:
            clf.is_trained = False
    except Exception:
        clf.is_trained = False
    return clf


def main() -> None:
    """
    Main function to run the Sentim application.
    This function sets up the Streamlit page and renders the UI components.
    """
    # --- Page Configuration ---
    assets_dir = PROJECT_ROOT / "assets" / "images"
    st.set_page_config(
        page_title="Sentim - AI Audio Analysis",
        page_icon=str(assets_dir / "logo.svg"),
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # --- Apply Custom CSS Styles ---
    apply_styles()

    # --- Render UI Sections ---
    st.write("")  # Start with some space at the top
    hero_section()
    st.write("")
    st.write("")
    how_it_works_section()
    st.write("")
    st.write("")

    # --- Backend resources & state ---
    # Cache and store the classifier in session_state for downstream components
    if "clf" not in st.session_state:
        st.session_state.clf = get_classifier(PROJECT_ROOT / "models")

    # Ensure synthesized audio persists even when tweaking original controls
    if "synth_audio_bytes" not in st.session_state:
        st.session_state.synth_audio_bytes = None
    if "synth_audio_array" not in st.session_state:
        st.session_state.synth_audio_array = None

    # --- Advanced Options (Grid Search) ---
    with st.expander("Advanced Options", expanded=False):
        st.caption("Configure classifier search parameters (used during training).")
        n_estimators = st.select_slider("n_estimators", options=[100, 200, 300], value=200)
        max_depth = st.selectbox("max_depth", options=[10, 20, None], index=2)
        min_samples_leaf = st.select_slider("min_samples_leaf", options=[1, 2, 4], value=1)
        min_samples_split = st.select_slider("min_samples_split", options=[2, 5], value=2)
        st.session_state.grid_params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "min_samples_split": min_samples_split,
        }

    analysis_dashboard_section()
    st.write("")
    st.write("")
    app_footer()


if __name__ == "__main__":
    main()
