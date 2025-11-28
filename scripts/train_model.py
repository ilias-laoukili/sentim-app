import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from src.backend import emotion_analysis
import numpy as np

# --- Configuration ---
DATA_PATH = "data/raw/audio_speech_actors_01-24"
# Save to PROJECT_ROOT/models/
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_SAVE_PATH = os.path.join(base_dir, 'models', 'ravdess_model.joblib')
TEST_SPLIT_SIZE = 0.2

# --- Emotion Mapping ---
# As per the dataset description
EMOTIONS = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def load_data(data_path):
    """
    Loads audio file paths and their corresponding emotion labels.
    """
    print(f"Loading data from: {data_path}")
    paths = glob.glob(os.path.join(data_path, "Actor_*", "*.wav"))
    if not paths:
        raise ValueError(f"No .wav files found in {data_path}. Check the path.")
        
    labels = []
    valid_paths = []
    for path in paths:
        basename = os.path.basename(path)
        try:
            emotion_code = basename.split('-')[2]
            labels.append(EMOTIONS[emotion_code])
            valid_paths.append(path)
        except (IndexError, KeyError):
            print(f"Warning: Could not parse emotion from filename: {basename}. Skipping.")

    print(f"Found {len(valid_paths)} audio files.")
    return valid_paths, labels

def main():
    """
    Main training and evaluation script.
    """
    # 1. Load data
    audio_paths, labels = load_data(DATA_PATH)

    # 2. Split data into training and testing sets
    print(f"Splitting data: {1-TEST_SPLIT_SIZE:.0%} train, {TEST_SPLIT_SIZE:.0%} test")
    X_train, X_test, y_train, y_test = train_test_split(
        audio_paths,
        labels,
        test_size=TEST_SPLIT_SIZE,
        random_state=None,
        stratify=labels  # Important for balanced splits
    )

    # 3. Initialize and train the classifier
    # Enable grid search for better hyperparameters
    print("Initializing classifier with Grid Search enabled...")
    clf = emotion_analysis.AcousticStatisticalClassifier(use_grid_search=True)
    print("Training model... This may take a few minutes.")
    clf.train(X_train, y_train)
    print("Training complete.")

    # 4. Evaluate the model
    print("Evaluating model on the test set...")
    predictions = []
    
    for i, path in enumerate(X_test):
        pred, _ = clf.predict(path)
        predictions.append(pred)

    # Print evaluation metrics
    unique_labels = np.unique(np.concatenate((y_test, predictions)))
    target_names = [label for label in clf.labels if label in unique_labels]

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, target_names=target_names, zero_division=0)
    cm = confusion_matrix(y_test, predictions, labels=clf.labels)
    
    print("\n--- Evaluation Results ---")
    print(f"Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(report)
    print("--------------------------\n")

    # 4.1 Plot and save confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_title('Confusion Matrix (RAVDESS)')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_xticks(range(len(clf.labels)))
    ax.set_yticks(range(len(clf.labels)))
    ax.set_xticklabels(clf.labels, rotation=45, ha='right')
    ax.set_yticklabels(clf.labels)
    plt.colorbar(im, ax=ax)
    # Add counts on cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='black')

    images_dir = os.path.join(base_dir, 'report', 'Signal_Processing___Project', 'images')
    os.makedirs(images_dir, exist_ok=True)
    cm_path = os.path.join(images_dir, 'confusion_matrix_ravdess.png')
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix figure to: {cm_path}")

    # 5. Save the trained model
    print(f"Saving trained model to: {MODEL_SAVE_PATH}")
    clf.save(MODEL_SAVE_PATH)
    print("Model saved successfully.")
    print(f"\nYou can now use '{MODEL_SAVE_PATH}' for predictions in your application.")

if __name__ == "__main__":
    main()
