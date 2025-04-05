# Android Static Analysis

This project provides a comprehensive pipeline for static analysis of Android APKs to detect malware. It extracts features from APK files, preprocesses the data, and trains machine-learning models to classify applications as benign or malicious. To perform this analysis, I used 800 malware and 800 benign APK files, which I collected from https://m4lware.org.

## Project Structure

```
android-static-analysis/
├── android_malware_preprocessing.py  : Preprocesses cleaned feature data
├── apk_features_updated.csv          : Output of feature extraction
├── benignSample/                     : Directory for benign APK samples
│   └── [benign APKs]
├── cleaned_features.csv              : Output of feature dropping
├── drop_irrelevant_features.py       : Removes irrelevant features
├── extract_apk_features.py           : Extracts features from APKs
├── malwareSample/                    : Directory for malware APK samples
│   └── [malware APKs]
├── model_comparison.py               : Trains and evaluates ML models
├── preprocessed_data_[timestamp]/    : Preprocessed data output directory
├── trainModel/                       : Trained model output directory
├── requirements.txt                  : Python dependencies
└── run_pipeline.py                   : Orchestrates the full pipeline
```

## Prerequisites

- Python: Version 3.8 or higher
- Virtual Environment: Recommended (e.g., `venv`)
- APK Samples: Place benign APKs in `benignSample/` and malware APKs in `malwareSample/`

## Installation

1. Clone the Repository:
   ```bash
   git clone <repository_url>
   cd android-static-analysis
   ```

2. Set Up a Virtual Environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install Dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Full Pipeline
To execute the entire pipeline from feature extraction to model training:
```bash
python3 run_pipeline.py
```

### Options
- `--malware-dir`: Directory with malware APKs (default: `malwareSample`)
- `--workers`: Number of worker processes for feature extraction (default: 5)
- `--save-interval`: Save interval for feature extraction (default: 50)
- `--resume`: Resume from the last successful step
- `--clean`: Clean output directories and files (e.g., `python3 run_pipeline.py --clean 1`)

### Running Individual Steps

### Resuming a Failed Run
If the pipeline is interrupted, resume the last successful step:
```bash
python3 run_pipeline.py --resume
```

## Pipeline Overview

1. **Feature Extraction** (`extract_apk_features.py`):
   - Extract static features (permissions, API calls, etc.) from APKs using Androguard.
   - Outputs: `apk_features_updated.csv`

2. **Feature Dropping** (`drop_irrelevant_features.py`):
   - Removes irrelevant features (e.g., `file_name`, `package_name`).
   - Outputs: `cleaned_features.csv`

3. **Preprocessing** (`android_malware_preprocessing.py`):
   - Handles missing values, outliers, and creates derived features.
   - Performs feature selection and standardization.
   - Outputs: `preprocessed_data_[timestamp]/` with train/test splits and visualizations.

4. **Model Training** (`model_comparison.py`):
   - Trains and evaluates multiple models (Random Forest, SVM, etc.).
   - Saves trained models and evaluation metrics.
   - Outputs: `trainModel/` with models (e.g., `best_model_random_forest.pkl`) and plots.

## Dependencies

See `requirements.txt` for a full list. Key packages include:
- `numpy`, `pandas`: Data processing
- `scikit-learn`: Machine learning
- `matplotlib`, `seaborn`: Visualization
- `androguard`: APK analysis

## Troubleshooting

- **Missing APKs**: Ensure `benignSample/` and `malwareSample/` contain `.apk` files.
- **Dependency Errors**: Verify all packages are installed (`pip install -r requirements.txt`).
- **Permission Issues**: Run with appropriate permissions if accessing restricted directories.
- **Model Not Saved**: Check `pipeline_run_*.log` for errors in the "Model Comparison" step.

## Contributing

Feel free to submit issues or pull requests to enhance the pipeline.

## License

This project is unlicensed unless specified otherwise by the repository owner.
