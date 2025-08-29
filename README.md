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

The dataset used in this project comes from the [Quality Prediction in a Mining Process](https://www.kaggle.com/datasets/edumagalhaes/quality-prediction-in-a-mining-process) dataset on Kaggle.  
It contains hourly time-series measurements of silica concentration and other process variables from an iron ore flotation plant, including feed and concentrate compositions, pH, density, air flow, and reagent dosages.

To download it manually:

1. Create a Kaggle account and generate an API token from your profile.  
2. Install the Kaggle CLI:
   ```bash
   pip install kaggle
   ```
3. Place your `kaggle.json` in `~/.kaggle/` (Linux/Mac) or `%USERPROFILE%\.kaggle\` (Windows).  
4. Download the dataset:
   ```bash
   kaggle datasets download -d edumagalhaes/quality-prediction-in-a-mining-process -p data/raw
   unzip data/raw/quality-prediction-in-a-mining-process.zip -d data/raw
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
# ☁️ Cloud Architecture (Azure)

To extend the project beyond local experiments, we evolved the pipeline into a **cloud-native architecture on Azure**, making it scalable, queryable, and ready for monitoring.

## Architecture Components

### Data Lake (ADLS Gen2)
A hierarchical storage account organizes data into the classical **bronze / silver / gold** layers:

- **Bronze**: Raw CSVs directly from the source (e.g., flotation process logs)
- **Silver**: Cleaned and preprocessed features stored as Parquet
- **Gold**: Curated datasets and aggregated KPIs for reporting

### Synapse Serverless SQL
A lightweight data warehouse layer implemented using Synapse Serverless. It queries CSV and Parquet directly from the Data Lake without duplicating data.

**Features:**
- **External Data Sources** connected to the lake
- **External Tables & Views** to expose metrics (pH, reagent flows, % silica concentrate)
- Daily aggregated KPIs to track process stability

### Monitoring & Alerts
The API and models were deployed in **Azure Container Apps**, connected to **Log Analytics**.

- Real-time log collection for debugging and audit trails
- Action Groups + Alerts configured for email notifications when:
  - Service becomes unavailable
  - Anomalies are detected

### BI & Visualization
Synapse queries connected to **Power BI** for interactive dashboards and process monitoring.

**Dashboard Features:**
- Time series of critical variables (pH, starch/amine flows, concentrate silica)
- Daily KPI aggregates
- Distribution plots to track flotation process variability

### Mini "Redshift" in Azure
This setup behaves like a serverless warehouse: data is queried on demand directly from the lake, enabling both ML model consumption and business reporting.

## Status

> **Note**: The cloud architecture is still a work in progress as we continue refining ingestion, permissions, and model integration. However, the foundation already enables queries, monitoring, and visualization end-to-end.

## 🔧 Architecture Diagram

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Sources  │───▶│   ADLS Gen2      │───▶│  Synapse SQL    │
│                 │    │                  │    │                 │
│ • Flotation     │    │ • Bronze (Raw)   │    │ • External      │
│   Process Logs  │    │ • Silver (Clean) │    │   Tables/Views  │
│ • Sensor Data   │    │ • Gold (KPIs)    │    │ • Aggregations  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                 │                        │
                                 │                        │
┌─────────────────┐              │               ┌─────────────────┐
│  Container Apps │              │               │    Power BI     │
│                 │              │               │                 │
│ • ML Models     │◀─────────────┘               │ • Dashboards    │
│ • API Services  │                              │ • Time Series   │
│ • Log Analytics │                              │ • KPI Reports   │
└─────────────────┘                              └─────────────────┘
```
## License
This project is released under the [MIT License](LICENSE).
