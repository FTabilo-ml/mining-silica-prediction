# Silica Concentration Prediction

This repository hosts a reproducible pipeline for predicting the percentage of silica in a mining concentrate stream. It contains feature engineering utilities, several model training scripts (baseline, LSTM and Transformer) and pre–computed reports.

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
