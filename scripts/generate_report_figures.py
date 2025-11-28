import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Add project root to path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.backend import dsp_utils

# Configuration
AUDIO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/course_ressources/Diner.wav'))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../notebooks/Signal_Processing___Project/images'))

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def save_plot(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved {path}")

def generate_analysis_diner(y, sr):
    plt.figure(figsize=(10, 12))

    # 1. Waveform
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(y, sr=sr, alpha=0.8)
    plt.title('Waveform of Diner.wav')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # 2. Spectrogram
    plt.subplot(3, 1, 2)
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log-Power Spectrogram')

    # 3. F0 Contour
    plt.subplot(3, 1, 3)
    f0 = librosa.yin(y, fmin=50, fmax=500)
    times = librosa.times_like(f0, sr=sr)
    plt.plot(times, f0, label='F0 (Fundamental Frequency)', color='r', linewidth=2)
    plt.title('F0 Contour (Pitch)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    save_plot('analysis_diner.png')

def generate_effect_speed(y, sr):
    # Time Stretch 1.5x (Faster)
    y_fast = dsp_utils.time_stretch(y, rate=1.5)
    
    plt.figure(figsize=(12, 6))
    
    # Original
    plt.subplot(1, 2, 1)
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
    plt.title(f'Original (Duration: {len(y)/sr:.2f}s)')
    
    # Stretched
    plt.subplot(1, 2, 2)
    D_fast = librosa.stft(y_fast)
    S_db_fast = librosa.amplitude_to_db(np.abs(D_fast), ref=np.max)
    librosa.display.specshow(S_db_fast, sr=sr, x_axis='time', y_axis='log')
    plt.title(f'Time Stretched x1.5 (Duration: {len(y_fast)/sr:.2f}s)')
    
    plt.tight_layout()
    save_plot('effect_speed.png')

def generate_effect_pitch(y, sr):
    # Pitch Shift +4 semitones
    y_shifted = dsp_utils.pitch_shift(y, n_steps=4)
    
    plt.figure(figsize=(12, 6))
    
    # Original
    plt.subplot(1, 2, 1)
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
    plt.title('Original Signal')
    
    # Shifted
    plt.subplot(1, 2, 2)
    D_shifted = librosa.stft(y_shifted)
    S_db_shifted = librosa.amplitude_to_db(np.abs(D_shifted), ref=np.max)
    librosa.display.specshow(S_db_shifted, sr=sr, x_axis='time', y_axis='log')
    plt.title('Pitch Shifted (+4 Semitones)')
    
    plt.tight_layout()
    save_plot('effect_pitch.png')

def generate_effect_robot(y, sr):
    # Robotize
    y_robot = dsp_utils.robotize(y, rate=1.0, carrier_freq=50, sr=sr)
    
    plt.figure(figsize=(12, 6))
    
    # Original
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(y, sr=sr, alpha=0.8)
    plt.title('Original Waveform')
    plt.ylim([-1, 1])
    
    # Robotized
    plt.subplot(2, 1, 2)
    librosa.display.waveshow(y_robot, sr=sr, alpha=0.8, color='g')
    plt.title('Robotized Waveform (50Hz Carrier)')
    plt.ylim([-1, 1])
    
    plt.tight_layout()
    save_plot('effect_robot.png')

def main():
    print(f"Loading audio from {AUDIO_PATH}...")
    try:
        y, sr = librosa.load(AUDIO_PATH, sr=None)
    except FileNotFoundError:
        print(f"Error: File not found at {AUDIO_PATH}")
        return

    print("Generating analysis_diner.png...")
    generate_analysis_diner(y, sr)

    print("Generating effect_speed.png...")
    generate_effect_speed(y, sr)

    print("Generating effect_pitch.png...")
    generate_effect_pitch(y, sr)

    print("Generating effect_robot.png...")
    generate_effect_robot(y, sr)

    print("Done!")

if __name__ == "__main__":
    main()
