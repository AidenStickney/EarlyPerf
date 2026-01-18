# Project Artifacts

This project includes large artifacts (trained models and parsed datasets) that are not stored in the git repo. They can be downloaded from the GitHub Releases.

## Trained Models (v1)
- **Tag**: `models-v1`
- **File**: `earlyperf-models-v1.zip`
- **SHA256**: `7dd09a2caccf11ae397f8d972dcc25b38792c15c3ceed28335f29fe6fcdb5d7c`
- **Contents**: 72 Trained ExtraTreesRegressors.
- **Unpack Location**: `artifacts/models/` (or `models/` for immediate use)

## Parsed Data (v1)
- **Tag**: `data-v1`
- **File**: `earlyperf-parsed-data-v1.zip`
- **SHA256**: `7926ad47623ff8d0a992b98a1769e83513f68f94757376c509a13ef9a13658a0`
- **Contents**: Parsed simulation traces and feature dictionaries.
- **Unpack Location**: `artifacts/data/` (or `data/` for immediate use)

## helper Scripts
Run the following scripts to automatically download and verify these artifacts:
```bash
python3 scripts/download_models.py
python3 scripts/download_data.py
```
