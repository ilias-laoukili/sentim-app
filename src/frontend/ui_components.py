import sys
import base64
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Ensure project root on sys.path so `src` is importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend import audio_processor
from src.backend import emotion_analysis


def hero_section() -> None:
    """Displays the main hero section of the app."""
    # Using columns for layout
    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.markdown(
            """
            <h1 style="font-size: 3.5rem; font-weight: 800; letter-spacing: -2px;">
                Analyze Audio Sentiment with Unparalleled Precision
            </h1>
            <p style="font-size: 1.2rem; color: #4B5563;">
                Sentim uses AI to analyze the emotional content of your audio files. 
                Upload a file and get a detailed sentiment analysis in seconds.
            </p>
        """,
            unsafe_allow_html=True,
        )

        # Adding some space before the button
        st.write("")
        st.write("")

        # To style the button as primary, we'll need a bit of a hack or use a different approach
        # For now, this creates the button structure. We'll rely on global styles.
        st.button("Get Started Now")

    with col2:
        st.image("assets/images/illustration.jpg")


def how_it_works_section() -> None:
    """Displays the 'How it Works' grid."""
    st.markdown("---")
    st.markdown(
        "<h2 style='text-align: center; font-size: 2.5rem;'>A Simple, Three-Step Process</h2>",
        unsafe_allow_html=True,
    )
    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="custom-container">
                <h3 style="font-weight: 700;">📤 1. Upload & Decode</h3>
                <p>Securely upload your audio. Our ML models instantly decompose the signal into raw emotional features.</p>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="custom-container">
                <h3 style="font-weight: 700;">🎚️ 2. Shape the Emotion</h3>
                <p>Take control. Adjust pitch, speed, and timbre using our DSP engine to manually shift the sentiment of the recording.</p>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="custom-container">
                <h3 style="font-weight: 700;">✨ 3. Resynthesize</h3>
                <p>Generate the modified audio in real-time. Visualize the emotional shift and export your perfectly tuned file.</p>
            </div>
        """,
            unsafe_allow_html=True,
        )


def _plot_amplitude(audio_data: np.ndarray, sr: int, color: str = "#0D9488") -> None:
    """Helper to plot amplitude over time."""
    fig_amp, ax_amp = plt.subplots(figsize=(10, 3))
    fig_amp.patch.set_alpha(0.0)
    ax_amp.set_facecolor("none")

    time_axis = np.linspace(0, len(audio_data) / sr, len(audio_data))

    ax_amp.plot(time_axis, audio_data, color=color, linewidth=0.8)
    ax_amp.fill_between(time_axis, audio_data, alpha=0.15, color=color)

    # Zero-clutter styling: remove all spines except bottom
    ax_amp.spines["top"].set_visible(False)
    ax_amp.spines["right"].set_visible(False)
    ax_amp.spines["left"].set_visible(False)
    ax_amp.spines["bottom"].set_color("#9CA3AF")
    ax_amp.spines["bottom"].set_linewidth(0.5)

    # Remove Y-axis completely
    ax_amp.set_yticks([])
    ax_amp.set_ylabel("")

    # Minimal X-axis styling
    ax_amp.set_xlabel("Time (s)", fontsize=9, color="#9CA3AF", fontfamily="sans-serif")
    ax_amp.tick_params(axis="x", colors="#9CA3AF", labelsize=8)
    ax_amp.grid(False)

    fig_amp.tight_layout(pad=0.5)
    st.pyplot(fig_amp)
    plt.close(fig_amp)


def _plot_spectrogram(audio_data: np.ndarray, sr: int) -> None:
    """Helper to plot spectrogram."""
    fig_spec, ax_spec = plt.subplots(figsize=(10, 3))
    fig_spec.patch.set_alpha(0.0)
    ax_spec.set_facecolor("none")

    # Compute STFT
    D = librosa.stft(audio_data)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # Plot without colorbar
    librosa.display.specshow(
        S_db,
        sr=sr,
        x_axis="time",
        y_axis="hz",
        cmap="magma",
        ax=ax_spec,
    )

    # Zero-clutter styling
    ax_spec.spines["top"].set_visible(False)
    ax_spec.spines["right"].set_visible(False)
    ax_spec.spines["left"].set_visible(False)
    ax_spec.spines["bottom"].set_color("#9CA3AF")
    ax_spec.spines["bottom"].set_linewidth(0.5)

    # Remove Y-axis completely
    ax_spec.set_yticks([])
    ax_spec.set_ylabel("")

    # Minimal X-axis styling
    ax_spec.set_xlabel("Time (s)", fontsize=9, color="#9CA3AF", fontfamily="sans-serif")
    ax_spec.tick_params(axis="x", colors="#9CA3AF", labelsize=8)

    fig_spec.tight_layout(pad=0.5)
    st.pyplot(fig_spec)
    plt.close(fig_spec)


def analysis_dashboard_section() -> None:
    """Handles the file upload and displays the analysis results with interactive refinement."""
    st.markdown("---")
    st.markdown(
        "<h2 style='text-align: center; font-size: 2.5rem;'>Analysis Dashboard</h2>",
        unsafe_allow_html=True,
    )
    st.write("")

    # Initialize session state
    if "original_audio" not in st.session_state:
        st.session_state.original_audio = None
    if "original_sr" not in st.session_state:
        st.session_state.original_sr = None
    if "original_bytes" not in st.session_state:
        st.session_state.original_bytes = None
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None
    if "modified_audio_bytes" not in st.session_state:
        st.session_state.modified_audio_bytes = None
    if "modified_audio_array" not in st.session_state:
        st.session_state.modified_audio_array = None

    uploaded_file = st.file_uploader(
        "Choose an audio file to analyze",
        type=["wav", "mp3", "m4a", "ogg"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")

        # Analyze button
        if st.button("🎯 Analyze Audio", type="primary"):
            with st.spinner("🔍 Analyzing audio... This may take a moment."):
                try:
                    # Use backend modules via src package

                    # Read uploaded file bytes
                    uploaded_bytes = uploaded_file.read()
                    uploaded_file.seek(0)  # Reset for playback

                    # Load and process audio
                    y, sr, audio_bytes, error = audio_processor.load_audio_file(
                        uploaded_bytes, uploaded_file.name
                    )

                    if error:
                        st.error(f"❌ Error loading audio: {error}")
                    else:
                        # Perform emotion analysis
                        label, confidence, features = emotion_analysis.emotion_analysis_heuristic(
                            y, sr
                        )

                        # Store in session state
                        st.session_state.original_audio = y
                        st.session_state.original_sr = sr
                        st.session_state.original_bytes = audio_bytes
                        st.session_state.analysis_results = {
                            "emotion": label,
                            "confidence": confidence,
                            "features": features,
                        }
                        st.session_state.modified_audio_bytes = None  # Reset modified audio

                        st.success("✅ Analysis Complete!")
                        st.rerun()

                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    st.error(traceback.format_exc())

        # Display results if analysis has been performed
        if st.session_state.analysis_results is not None:
            st.write("")
            st.write("")

            # Split layout: Results (left) and Interactive Tweaking (right)
            col1, col2 = st.columns([1, 1])

            with col1:
                # Read-Only Results Section
                st.markdown("### 📊 Original Analysis")

                with st.container(border=True):
                    st.markdown("#### **Before State**")

                    # Display detected emotion
                    emotion = st.session_state.analysis_results["emotion"]
                    confidence = st.session_state.analysis_results["confidence"]

                    st.metric(
                        label="Detected Emotion",
                        value=emotion,
                        delta=f"{confidence*100:.1f}% confidence",
                    )

                    # Display features
                    features = st.session_state.analysis_results["features"]
                    st.write("**Audio Features:**")
                    st.write(f"- RMS Energy: {features['rms']:.4f}")
                    st.write(f"- Spectral Centroid: {features['centroid']:.1f} Hz")
                    st.write(f"- Pitch (F0): {features['f0_median']:.1f} Hz")

                    st.write("")

                    # Plot amplitude over time
                    st.write("**Amplitude Over Time:**")
                    _plot_amplitude(
                        st.session_state.original_audio,
                        st.session_state.original_sr,
                        color="#0D9488",
                    )

                    st.write("")

                    # Plot spectrogram
                    st.write("**Spectrogram:**")
                    _plot_spectrogram(st.session_state.original_audio, st.session_state.original_sr)

                    st.write("")
                    st.write("**Original Audio:**")
                    st.audio(st.session_state.original_bytes, format="audio/wav")

            with col2:
                # Interactive Tweaking Section
                st.markdown("### 🎚️ Shape Your Audio")

                # Tab selection for manual vs automatic
                tab1, tab2 = st.tabs(["⚙️ Manual Controls", "🤖 Auto Transform"])

                with tab1:
                    with st.container(border=True):
                        st.markdown("#### **Manual Modification**")

                        # Visual guide
                        st.caption(
                            "💡 **Tip:** Increasing Pitch and Speed typically correlates with higher Joy/Anxiety metrics."
                        )

                        st.write("")

                        # DSP Controls mapped to Emotional outcomes
                        pitch_shift = st.slider(
                            "🎵 Pitch Shift (Higher = More Happy/Tense)",
                            min_value=-5.0,
                            max_value=5.0,
                            value=0.0,
                            step=0.5,
                            help="Adjust pitch to convey different emotions",
                            key="manual_pitch",
                        )

                        speed = st.slider(
                            "⚡ Speed (Faster = More Urgent)",
                            min_value=0.5,
                            max_value=2.0,
                            value=1.0,
                            step=0.1,
                            help="Change speaking rate to modify emotional intensity",
                            key="manual_speed",
                        )

                        # Robotize effect
                        robotize_freq = st.slider(
                            "🤖 Robotize (Carrier Frequency in Hz)",
                            min_value=0.0,
                            max_value=500.0,
                            value=0.0,
                            step=10.0,
                            help="Apply ring modulation for robotic voice effect (0 = off)",
                            key="robotize_freq",
                        )

                        # Echo effect
                        echo_delay = st.slider(
                            "🔊 Echo Delay (milliseconds)",
                            min_value=0.0,
                            max_value=1000.0,
                            value=0.0,
                            step=50.0,
                            help="Add echo/delay effect to the audio (0 = off)",
                            key="echo_delay",
                        )

                        echo_decay = st.slider(
                            "📉 Echo Decay (feedback strength)",
                            min_value=0.0,
                            max_value=0.9,
                            value=0.5,
                            step=0.1,
                            help="Controls how quickly the echo fades (higher = longer echo)",
                            key="echo_decay",
                        )

                        st.write("")

                        # Apply button with custom styling
                        if st.button(
                            "✨ Apply Manual Shift", use_container_width=True, key="apply_manual"
                        ):
                            with st.spinner("🔧 Applying transformations..."):
                                try:
                                    # Use audio_processor via src package

                                    # Apply effects
                                    y_modified, modified_bytes = audio_processor.process_audio(
                                        st.session_state.original_audio,
                                        st.session_state.original_sr,
                                        speed=speed,
                                        pitch=int(pitch_shift),
                                        carrier_freq=robotize_freq,
                                        echo_delay=echo_delay,
                                        echo_decay=echo_decay,
                                    )

                                    if y_modified is not None and modified_bytes is not None:
                                        st.session_state.modified_audio_bytes = modified_bytes
                                        st.session_state.modified_audio_array = y_modified
                                        st.success("✅ Transformation applied!")
                                        st.rerun()
                                    else:
                                        st.error("❌ Failed to apply transformations")

                                except Exception as e:
                                    st.error(f"❌ Error applying effects: {str(e)}")
                                    with st.expander("See error details"):
                                        st.code(traceback.format_exc())

                with tab2:
                    with st.container(border=True):
                        st.markdown("#### **Automatic Emotion Transform**")

                        st.caption(
                            "🎭 Select a target emotion and let AI automatically adjust the audio parameters."
                        )

                        st.write("")

                        # Target emotion selector
                        target_emotion = st.selectbox(
                            "🎯 Target Emotion",
                            options=[
                                "Joy",
                                "Happiness",
                                "Sadness",
                                "Anger",
                                "Fear",
                                "Surprise",
                                "Neutral",
                            ],
                            help="Choose the emotion you want the audio to convey",
                            key="target_emotion",
                        )

                        st.write("")

                        # Apply automatic transformation
                        if st.button(
                            f"🪄 Transform to {target_emotion}",
                            use_container_width=True,
                            key="apply_auto",
                        ):
                            with st.spinner(f"🔮 Transforming audio to {target_emotion}..."):
                                try:
                                    # Ensure original audio/session aliases exist
                                    if "y" not in st.session_state:
                                        st.session_state["y"] = st.session_state.original_audio
                                    if "sr" not in st.session_state:
                                        st.session_state["sr"] = st.session_state.original_sr

                                    # Instantiate modifier
                                    modifier = emotion_analysis.EmotionAudioModifier()

                                    # Call optimization using cached classifier object
                                    modified_audio, results = modifier.modify_with_classifier(
                                        audio=st.session_state["y"],
                                        sample_rate=st.session_state["sr"],
                                        classifier=st.session_state["clf"],
                                        target_emotion=target_emotion,
                                    )

                                    # Map results to requested outputs
                                    best_audio = modified_audio
                                    best_params = results.get("parameters_applied", {})
                                    best_score = results.get("modified_confidence", None)
                                    orig_score = results.get("original_confidence", None)

                                    # Encode for playback
                                    modified_bytes = audio_processor.encode_audio_for_playback(
                                        best_audio, st.session_state["sr"]
                                    )

                                    if modified_bytes is not None:
                                        # Persist synthesized audio so it doesn't disappear on original control changes
                                        st.session_state.modified_audio_bytes = modified_bytes
                                        st.session_state.modified_audio_array = best_audio
                                        st.session_state.synth_audio_bytes = modified_bytes
                                        st.session_state.synth_audio_array = best_audio

                                        st.success(f"✅ Transformed to {target_emotion}!")

                                        # Display applied parameters and scores
                                        if best_params:
                                            st.info(
                                                f"**Applied Parameters:**\n\n"
                                                f"• Pitch Shift: {best_params.get('pitch_shift', 0):+.1f} semitones\n\n"
                                                f"• Tempo: {best_params.get('tempo', 1.0):.2f}x\n\n"
                                                f"• Energy: {best_params.get('energy', 1.0):.2f}x"
                                            )
                                        if best_score is not None and orig_score is not None:
                                            st.caption(
                                                f"Classifier confidence: original {orig_score:.2f} → modified {best_score:.2f}"
                                            )
                                        st.rerun()
                                    else:
                                        st.error("❌ Failed to encode transformed audio")

                                except Exception as e:
                                    st.error(f"❌ Error during transformation: {str(e)}")
                                    with st.expander("See error details"):
                                        st.code(traceback.format_exc())

                # Display modified audio if available (shared between both tabs)
                if st.session_state.modified_audio_bytes is not None:
                    st.write("")
                    st.markdown("---")
                    with st.container(border=True):
                        st.write("**🎧 Modified Audio Results:**")

                        # Audio player
                        st.audio(st.session_state.modified_audio_bytes, format="audio/wav")

                        st.write("")

                        # Plot modified amplitude
                        if st.session_state.modified_audio_array is not None:
                            st.write("**Modified Amplitude Over Time:**")
                            _plot_amplitude(
                                st.session_state.modified_audio_array,
                                st.session_state.original_sr,
                                color="#6366F1",
                            )

                            st.write("")

                            # Plot modified spectrogram
                            st.write("**Modified Spectrogram:**")
                            _plot_spectrogram(
                                st.session_state.modified_audio_array, st.session_state.original_sr
                            )


def app_footer() -> None:
    """Displays the custom app footer."""
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; padding: 2rem 0; background-color: #000000; color: #FFFFFF; border-radius: 16px;">
            <p style="color: #FFFFFF;">Built with ❤️ by Ilias Laoukili</p>
            <p style="font-size: 0.9rem; color: #E5E7EB;">© 2025 Ilias Laoukili. All Rights Reserved.</p>
        </div>
    """,
        unsafe_allow_html=True,
    )
