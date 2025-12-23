# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the iHARP ML Challenge 2 repository focused on "Predicting Coastal Flooding Events" as part of the "Modelling Out Of Distribution" theme.

### Background and Motivation
The US East Coast is vulnerable to storm surge-driven extreme flooding. The National Data Buoy Center (NDBC) provides high-resolution data from offshore and coastal buoys, offering 70+ years of tidal variability and sea level measurements. The challenge is to leverage AI/ML methods to predict coastal flooding events across different spatial locations, accounting for the complex interplay between large-scale atmospheric/ocean dynamics and local conditions.

### Challenge Structure
Participants develop models to predict coastal flooding using historical sea level data from 12 coastal stations (1950-2020). The focus is on out-of-distribution (OOD) modeling across both spatial (different stations) and temporal (different time periods) dimensions.

## Data Architecture

### Dataset Structure
- **Main data**: `NEUSTG_19502020_12stations.mat` (MATLAB format, 13.7MB)
  - Contains hourly sea level measurements (1950-2020) for 12 coastal stations
  - Variables: `lattg`, `lontg`, `sltg` (sea level time series), `sname` (station names), `t` (time in MATLAB datenum)
  - Each station has at least 80% hourly records from 1950-2020 (number of records may vary slightly)

### Dataset Characteristics
**Important**: The dataset exhibits:
- **Seasonality**: Periodic patterns in sea level variation
- **Noise**: Random fluctuations in measurements
- **Auto-correlation**: Values correlated with previous time steps
- **Non-stationarity**: Statistical properties change over time

Models should account for these properties when developing prediction algorithms.

### Station Split (Fixed for OOD Evaluation)
- **Training stations (9)**: Annapolis, Atlantic_City, Charleston, Washington, Wilmington, Eastport, Portland, Sewells_Point, Sandy_Hook
- **Testing stations (3)**: Lewes, Fernandina_Beach, The_Battery
- This split is hardcoded in `Ingestion_Program/Ingestion Program/ingestion.py:23-24`

### Seed Files
- `Seed_Coastal_Stations.txt`: List of all 12 station names
- `Seed_Coastal_Stations_Thresholds.mat`: Official flooding thresholds per station
- `Seed_Historical_Time_Intervals.txt`: 15 pre-selected 7-day historical time intervals for evaluation

### Global Model Expectations
- **Development dataset**: 2520 total prediction entries
  - Calculation: 12 stations × 15 historical windows × 14 prediction days
  - 9 training stations, 3 testing stations (for OOD evaluation)
- **Hidden test dataset**: 840 entries (used in final phase)
  - Calculation: 4 hidden stations × 15 historical windows × 14 prediction days
  - Hidden stations are completely separate from the 12 development stations

## Model Submission Structure

Submissions are executed via `model.py` with the following interface:

```bash
python -m model --train_hourly <csv> --test_hourly <csv> --test_index <csv> --predictions_out <csv>
```

### Required Arguments
- `--train_hourly`: CSV with hourly sea level data for training stations
- `--test_hourly`: CSV with hourly sea level data for testing stations
- `--test_index`: CSV defining evaluation windows with columns: id, station_name, hist_start, hist_end, future_start, future_end
- `--predictions_out`: Output path for predictions CSV

### Submission Formats (Both Supported)
1. **Minimal**: Only `model.py` (baseline uses XGBoost or sklearn fallback)
2. **Full**: `model.py`, `model.pkl` (pre-trained weights), `requirements.txt`, `README.md`

### Prediction Format
Output CSV must contain:
- `id`: Matches test_index ids
- `y_prob`: Float probability [0,1] OR `label`: Binary {0,1}

## Core ML Pipeline

### Time Window Structure
- **Historical window**: 7 days (`HIST_DAYS=7`)
- **Forecast horizon**: 14 days (`FUTURE_DAYS=14`)
- Windows slide through data with 1-day steps

### Feature Engineering (Baseline)
1. **Hourly to Daily Aggregation**:
   - `daily_aggregate()` converts hourly measurements to daily mean/max

2. **Computed Features** (per station, per day):
   - `sea_level`: Daily mean
   - `sea_level_3d_mean`: 3-day rolling mean
   - `sea_level_7d_mean`: 7-day rolling mean

3. **Flood Labeling** (Binary: 1=Flooding, 0=Non-flooding):
   - Threshold = mean(sea_level) + 1.5 * std(sea_level) per station (baseline uses this; official thresholds in .mat file)
   - **Daily flood determination**: A day is marked as flooding if ANY hour (out of 24) has a flooding event
   - Hourly flood event: sea_level measurement exceeds the flooding threshold
   - Binary classification target: Any flood in 14-day forecast window → label=1

### Baseline Model Architecture
- **Model**: XGBoost binary classifier (falls back to sklearn GradientBoostingClassifier if XGBoost unavailable)
- **Input**: 21 features (7 days × 3 features per day)
- **Output**: Probability that ANY flood occurs in next 14 days
- **Hyperparameters** (XGBoost):
  - n_estimators=400, max_depth=4, learning_rate=0.05
  - subsample=0.8, colsample_bytree=0.8
  - Uses class imbalance weighting via `scale_pos_weight`

## Evaluation Workflow

### Evaluation Phases

#### Development Phase
- **Purpose**: Model development and refinement
- **Dataset**: 12 stations (9 training, 3 testing) with 15 pre-selected historical windows
- **Submission limits**: Up to 5 submissions per day allowed
- **Leaderboard**: Participants can submit one score, which can be replaced with better scores
- **Feedback**: Provided continuously throughout the development phase

#### Final Phase
- **Activation**: Starts automatically at the end of the challenge
- **Dataset**: Hidden test dataset with 4 undisclosed coastal stations (840 entries)
- **Submission**: Participant's last/preferred submission is evaluated on hidden data
- **Scoring**: Conducted secretly by organizers, final scores posted to leaderboard

### Local Testing
The baseline code (`baseline_model_xgboost_v3.py`) demonstrates:
1. Loading MAT file using scipy
2. Converting MATLAB datenum to Python datetime
3. Station selection and feature engineering
4. Building sliding windows for training
5. Training and evaluation

### Codabench Execution
1. **Ingestion** (`Ingestion_Program/Ingestion Program/ingestion.py`):
   - Reads MAT file and splits into train/test CSVs
   - Generates `test_index.csv` with evaluation windows
   - Executes participant's `model.py`
   - Validates output format

2. **Scoring** (`Ingestion_Program/Scoring Program/scoring.py`):
   - Loads ground truth from `y_test.csv`
   - Computes metrics: AUC, Accuracy, F1, MCC
   - Outputs `scores.json`

### Metrics
Models are evaluated based on:
1. **F1 Score**: Harmonic mean of precision/recall (primary metric per README)
2. **Accuracy**: Correct predictions / total predictions
3. **MCC**: Matthews correlation coefficient (handles class imbalance well)
4. **AUC**: Area under ROC curve (also computed by scoring program)

**Expected Outputs**: For each 14-day prediction window following each 7-day historical interval:
- Binary predictions or probabilities
- Confusion matrix (TP, FP, TN, FN)
- All evaluation metrics listed above

## Key Implementation Notes

### Data Loading
Use `scipy.io.loadmat()` for MAT files. Convert MATLAB datenum:
```python
def matlab2datetime(matlab_datenum):
    return datetime.fromordinal(int(matlab_datenum)) + timedelta(days=matlab_datenum % 1) - timedelta(days=366)
```

### Threshold Computation
Always compute thresholds from TRAINING data only to avoid data leakage:
```python
thr = (train.groupby("station_name")["sea_level"]
            .agg(["mean","std"])
            .assign(flood_threshold=lambda x: x["mean"] + 1.5*x["std"]))
```

### Window Building
The `build_windows()` function in baseline model:
- Handles station grouping and temporal sorting
- Skips windows with NaN values
- Creates unique keys: `station|hist_start|future_start` for alignment

### Important Caveats
1. **Threshold source**: Baseline uses alternative thresholds (mean + 1.5*std). Official thresholds are in `Seed_Coastal_Stations_Thresholds.mat`
2. **Temporal granularity**: Input data is hourly, but models should predict on daily intervals (convert hourly → daily)
3. **Global model requirement**: Model should predict across ALL stations (not station-specific models)
4. **Evaluation variance**: Local evaluation results may differ from Codabench due to fixed train/test split
5. **Hidden dataset**: Final phase uses 4 completely hidden stations with same OOD evaluation methodology
6. **Time window flexibility**: Participants can use the 15 seed intervals OR define custom 7-day historical windows for additional validation

## Python Dependencies

Core packages (from `model_submission-2/requirements.txt`):
- numpy, pandas, scipy (data handling, MAT file loading)
- xgboost (baseline classifier)
- scikit-learn (metrics, fallback classifier)

Optional whitelisted packages include: pytorch, tensorflow, keras, lightgbm, catboost, transformers, opencv-python-headless, albumentations, timm, etc.

## File Structure Reference

```
iHARP-ML-Challenge-3/
├── NEUSTG_19502020_12stations.mat          # Main dataset
├── Seed_Coastal_Stations.txt               # Station names
├── Seed_Coastal_Stations_Thresholds.mat    # Official flood thresholds
├── Seed_Historical_Time_Intervals.txt      # Example time windows
├── baseline_model_xgboost_v3.py        # Reference code
├── model_submission/                        # Minimal submission example
│   └── model.py
├── model_submission-2/                      # Full submission example
│   ├── model.py
│   ├── requirements.txt
│   └── README.md
└── Ingestion_Program/
    ├── Ingestion Program/
    │   ├── ingestion.py                     # Codabench ingestion
    │   └── metadata.yaml
    └── Scoring Program/
        ├── scoring.py                       # Evaluation metrics
        └── metadata.yaml
```
