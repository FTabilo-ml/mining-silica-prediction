# Silica Concentration Prediction

This repository hosts a reproducible pipeline for predicting the percentage of silica in a mining concentrate stream. It contains feature engineering utilities, several model training scripts (baseline, LSTM and Transformer) and pre–computed reports.
## Inspirations

This project is inspired by the work done in [Quality_Prediction_ML by npatel221](https://github.com/npatel221/Quality_Prediction_ML), which applied machine learning techniques to predict the percentage of silica (an impurity) in iron ore concentrate using flotation plant data :contentReference[oaicite:2]{index=2}.

Building on that foundation, our project extends the approach by:
- Introducing advanced architectures such as LSTM and Transformer
- Structuring a modular and reproducible pipeline (via DVC)
- Including a visual dashboard for exploratory analysis
- Integrating CI/CD, badges, and metadata enhancements to improve usability and maintainability

## Project structure

```
├── data/                # Dataset pointers managed with DVC
├── models/              # Model checkpoints (tracked via DVC)
├── notebooks/           # Exploratory notebooks
├── reports/             # CSV with final results
├── results/             # Placeholder for generated outputs (ignored)
├── src/
│   ├── features/        # Feature engineering scripts
│   └── models/          # Model training & evaluation scripts
```

## Dataset

The dataset used in this project comes from the ([Quality Prediction in a Mining Process](https://www.kaggle.com/datasets/edumagalhaes/quality-prediction-in-a-mining-process)) on Kaggle.  
It contains time-series measurements of silica concentration and other process variables from a mining operation.

To download it manually:

1. Create a Kaggle account and generate an API token from your profile.  
2. Install the Kaggle CLI:
   ```bash
   pip install kaggle
   ```
3. Place your `kaggle.json` in `~/.kaggle/` (Linux/Mac) or `%USERPROFILE%\.kaggle\` (Windows).  
4. Download the dataset:
   ```bash
   kaggle datasets download -d imoore/mining-core-sampling-dataset -p data/raw
   unzip data/raw/mining-core-sampling-dataset.zip -d data/raw
   ```

## Requirements
Install Python 3.10+ and the packages listed in [`requirements.txt`](requirements.txt).

```
pip install -r requirements.txt
```

The dataset and model artifacts are versioned with [DVC](https://dvc.org/). Fetch them using:

```
dvc pull
```

## Usage
1. Generate additional features (optional):
   ```bash
   python src/features/make_features.py
   ```
2. Train baseline models (Random Forest and XGBoost):
   ```bash
   python src/models/train_baselines.py
   ```
3. Train the LSTM model:
   ```bash
   python src/models/train_lstm.py
   ```
4. Train the Transformer model:
   ```bash
   python src/models/train_transformer.py
   ```
5. Validate time‑series splits and compare models:
   ```bash
   python src/models/validate_timeseries.py
   ```

Generated metrics are stored in the `reports/` folder. Cross‑validation results show a mean RMSE of **0.075**, MAE of **0.039** and R² of **0.993** on the test sets.

## License
This project is released under the [MIT License](LICENSE).
