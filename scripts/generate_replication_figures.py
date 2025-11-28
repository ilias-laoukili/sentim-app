import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import scipy.fftpack

# Add project root to path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.backend import dsp_utils

# Configuration
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/course_ressources'))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../notebooks/Signal_Processing___Project/images'))

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def save_plot(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved {path}")

def plot_analysis(filename, title_prefix):
    filepath = os.path.join(DATA_DIR, filename)
    y, sr = librosa.load(filepath, sr=None)
    
    plt.figure(figsize=(12, 10))
    
    # 1. Waveform
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(y, sr=sr, alpha=0.8)
    plt.title(f'{title_prefix} - Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    # 2. Spectrum (FFT)
    plt.subplot(3, 1, 2)
    N = len(y)
    yf = scipy.fftpack.fft(y)
    xf = np.linspace(0.0, sr/2.0, N//2)
    plt.plot(xf, 2.0/N * np.abs(yf[:N//2]))
    plt.title(f'{title_prefix} - Frequency Spectrum (FFT)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid()
    
    # 3. Spectrogram
    plt.subplot(3, 1, 3)
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'{title_prefix} - Spectrogram')
    
    plt.tight_layout()
    save_plot(f'analysis_{filename.split(".")[0].lower()}.png')

def plot_speed_comparison():
    filepath = os.path.join(DATA_DIR, 'Diner.wav')
    y, sr = librosa.load(filepath, sr=None)
    
    y_slow = dsp_utils.time_stretch(y, rate=0.5)
    y_fast = dsp_utils.time_stretch(y, rate=1.5)
    
    plt.figure(figsize=(12, 12))
    
    # Original
    plt.subplot(3, 1, 1)
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
    plt.title(f'Original (Duration: {len(y)/sr:.2f}s)')
    plt.colorbar(format='%+2.0f dB')

    # Slow (0.5x)
    plt.subplot(3, 1, 2)
    D_slow = librosa.stft(y_slow)
    S_db_slow = librosa.amplitude_to_db(np.abs(D_slow), ref=np.max)
    librosa.display.specshow(S_db_slow, sr=sr, x_axis='time', y_axis='log')
    plt.title(f'Slow 0.5x (Duration: {len(y_slow)/sr:.2f}s)')
    plt.colorbar(format='%+2.0f dB')

    # Fast (1.5x)
    plt.subplot(3, 1, 3)
    D_fast = librosa.stft(y_fast)
    S_db_fast = librosa.amplitude_to_db(np.abs(D_fast), ref=np.max)
    librosa.display.specshow(S_db_fast, sr=sr, x_axis='time', y_axis='log')
    plt.title(f'Fast 1.5x (Duration: {len(y_fast)/sr:.2f}s)')
    plt.colorbar(format='%+2.0f dB')
    
    plt.tight_layout()
    save_plot('comparison_speed.png')

def plot_pitch_comparison():
    filepath = os.path.join(DATA_DIR, 'Diner.wav')
    y, sr = librosa.load(filepath, sr=None)
    
    y_up = dsp_utils.pitch_shift(y, n_steps=4)
    y_down = dsp_utils.pitch_shift(y, n_steps=-4)
    
    plt.figure(figsize=(12, 12))
    
    # Original
    plt.subplot(3, 1, 1)
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
    plt.title('Original Pitch')
    plt.colorbar(format='%+2.0f dB')

    # Up (+4)
    plt.subplot(3, 1, 2)
    D_up = librosa.stft(y_up)
    S_db_up = librosa.amplitude_to_db(np.abs(D_up), ref=np.max)
    librosa.display.specshow(S_db_up, sr=sr, x_axis='time', y_axis='log')
    plt.title('Pitch Shift +4 Semitones')
    plt.colorbar(format='%+2.0f dB')

    # Down (-4)
    plt.subplot(3, 1, 3)
    D_down = librosa.stft(y_down)
    S_db_down = librosa.amplitude_to_db(np.abs(D_down), ref=np.max)
    librosa.display.specshow(S_db_down, sr=sr, x_axis='time', y_axis='log')
    plt.title('Pitch Shift -4 Semitones')
    plt.colorbar(format='%+2.0f dB')
    
    plt.tight_layout()
    save_plot('comparison_pitch.png')

def plot_robotization_comparison():
    filepath = os.path.join(DATA_DIR, 'Diner.wav')
    y, sr = librosa.load(filepath, sr=None)
    
    freqs = [200, 500, 1000, 2000]
    
    plt.figure(figsize=(15, 10))
    
    for i, fc in enumerate(freqs):
        y_rob = dsp_utils.robotize(y, rate=1.0, carrier_freq=fc, sr=sr)
        
        plt.subplot(2, 2, i+1)
        D = librosa.stft(y_rob)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
        plt.title(f'Robotization (fc={fc}Hz)')
        plt.colorbar(format='%+2.0f dB')
        
    plt.tight_layout()
    save_plot('comparison_robot.png')

def main():
    print("Generating Diner Analysis...")
    plot_analysis('Diner.wav', 'Diner')
    
    print("Generating Halleluia Analysis...")
    plot_analysis('Halleluia.wav', 'Halleluia')
    
    print("Generating Speed Comparison...")
    plot_speed_comparison()
    
    print("Generating Pitch Comparison...")
    plot_pitch_comparison()
    
    print("Generating Robotization Comparison...")
    plot_robotization_comparison()

if __name__ == "__main__":
    main()
