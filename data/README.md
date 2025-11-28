# Data Directory

This directory contains audio datasets for emotion recognition training.

## Structure

```
data/
├── raw/                        # Raw audio files (RAVDESS dataset)
│   └── audio_speech_actors_01-24/
├── processed/                  # Preprocessed features (optional)
└── raw.zip                     # Archived raw dataset
```

## Setup

### Option 1: Download RAVDESS Dataset

1. Visit [RAVDESS on Zenodo](https://zenodo.org/record/1188976)
2. Download "Audio-only files" (approximately 1.5 GB)
3. Extract to `data/raw/audio_speech_actors_01-24/`

### Option 2: Use Provided Archive

If you have `raw.zip`:
```bash
cd data
unzip raw.zip -d raw/
```

## Important Notes

- **Large Files:** The `data/raw/` directory is ignored by Git (see `.gitignore`)
- **Git LFS Alternative:** For version control of datasets, consider using [Git LFS](https://git-lfs.github.com/)
- **DVC Alternative:** For ML data pipelines, consider using [DVC](https://dvc.org/)

## Dataset Citation

> Livingstone SR, Russo FA (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). PLoS ONE 13(5): e0196391. https://doi.org/10.1371/journal.pone.0196391

## Disk Space Requirements

- Raw RAVDESS: ~1.5 GB
- Processed features (if cached): ~500 MB
- **Total:** ~2 GB

## IDE Performance

If your IDE (VS Code, PyCharm) is slow:

1. **Exclude from indexing:**
   - VS Code: Add `data/` to `files.watcherExclude` in settings
   - PyCharm: Right-click `data/` → Mark Directory as → Excluded

2. **Use .gitignore:** Already configured to ignore `data/raw/`

3. **Consider symlinks:** Store data on external drive:
   ```bash
   mv data /Volumes/ExternalDrive/sentim-data
   ln -s /Volumes/ExternalDrive/sentim-data data
   ```
