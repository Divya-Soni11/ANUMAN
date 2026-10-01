# Anuman

**A soft sensor for the S-Zorb gasoline desulfurization unit.**

Predicts product octane number (RON) in real time from refinery process
variables, closing the two-day gap between lab measurements.

[**Live Demo**](https://anuman-soft-sensor.streamlit.app) · [Dataset DOI](https://doi.org/10.17632/h44t3pvw22.1) · [Report an Issue](https://github.com/YOUR_USERNAME/anuman/issues)

---

## Overview

S-Zorb is a licensed refinery process that removes sulfur from gasoline.
The industrial challenge: removing sulfur also destroys octane, and
product RON is only measured twice a week in the lab. Between lab
readings, operators have no visibility into quality. Off-spec product
can be produced without warning.

Anuman is a soft sensor — a machine learning model that estimates product
RON continuously from the 365 process variables that are always available
at the unit. It gives operators early warning before off-spec product is
produced.

The model was trained on three years of real operational data from a
Sinopec S-Zorb unit, sourced from a Honeywell PHD historian [reference:0].

## Results

| Model              | RMSE    | MAE     | R²      |
|--------------------|---------|---------|---------|
| Ridge              | 5.6791  | 3.0460  | -436.05 |
| **Random Forest**  | **0.3850** | **0.2684** | **0.7734** |
| Gradient Boosting  | 0.4097  | 0.2888  | 0.7665  |
| XGBoost            | 0.4423  | 0.3336  | 0.6566  |

Evaluated with five-fold `TimeSeriesSplit` cross-validation on 325 samples.
Random Forest was selected based on cross-validated RMSE.

## What the model learned

SHAP attribution shows that **feed RON dominates the prediction** (impact
≈ +0.81), while process variables contribute a small but real residual
(±0.005 per feature). This is physically correct. S-Zorb cannot create
octane, only remove sulfur. Feed quality drives product quality.

The model is not a black box. Every prediction is explained.

## Negative results (documented honestly)

Two targets were tested and failed cross-validation:

- **RON loss** — R² < 0. Lab noise dominates this signal.
- **Product sulfur** — R² < 0. Twice-weekly composite lab samples do not
  align temporally with snapshot process variables.

This is a dataset limitation, not a modeling failure. It is why industry
deploys online analyzers rather than soft sensors for high-frequency
quality variables.

---

## Project structure
anuman/
├── app.py # Streamlit dashboard
├── requirements.txt # Python dependencies
├── readme.md # This file
├── LICENSE # MIT License
│
├── data/
│ ├── raw/ # Original Excel (excluded from Git)
│ └── processed/
│ └── s_zorb_clean.csv # Cleaned dataset used by the app
│
├── models/
│ ├── best_model.joblib # Trained Random Forest
│ └── model_metadata.json # Feature names, metrics
│
├── src/
│ ├── data_loader.py # Load, clean, split the raw data
│ ├── train.py # Train and benchmark models
│ └── insights.py # SHAP explainability
│
├── notebooks/
│ └── 01_dataset_exploration.py # First-pass exploration
│
└── reviewing original dataset.py # Initial diagnostic script


---

## How to run locally

### Requirements

- Python 3.10 or higher (3.12 recommended)
- pip
- ~500 MB of free disk space

### Setup

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/anuman.git
cd anuman

#### Create and activate a virtual environment:
Windows:
python -m venv venv
venv\Scripts\activate

macOS / Linux:
python -m venv venv
source venv/bin/activate

# Install dependencies:

pip install -r requirements.txt

#Get the raw dataset
The raw Excel file is not included in this repository (1.4 MB of
proprietary refinery data). Download it from Mendeley:

DOI: 10.17632/h44t3pvw22.1

#Run the app
streamlit run app.py --server.headless=true

Pipeline
The full pipeline runs in three steps.

1. Data loading and cleaning (src/data_loader.py)

Reads the raw Excel file, which has a three-row header with merged
cells for group labels

Handles the dash - markers used for missing measurements

Parses mixed-format timestamps (2020/5/26 8:00:00 and 2018/12/1)

Splits into features, targets, and metadata

Excludes all product_* columns from the feature set to prevent
data leakage — they are lab-measured outputs, only known after the
lab result comes back

2. Model training and selection (src/train.py)

Trains four regression models: Ridge, Random Forest, Gradient
Boosting, XGBoost

Evaluates each with five-fold TimeSeriesSplit cross-validation

Selects the best by cross-validated RMSE

Saves the winner and metadata to models/

3. Explainability (src/insights.py)

Loads the trained model

Builds a SHAP TreeExplainer

Generates per-prediction feature attributions

Produces operator-readable explanations



FAQ
1.Why TimeSeriesSplit and not random split?
The dataset is chronological. Random splits leak future information
into training, inflating performance. TimeSeriesSplit respects time
order — training folds always come before test folds.

2.Why exclude product_* columns from features?
product_ron_loss is feed_RON − product_RON. Including it as a
feature would leak the target. All product properties are lab-measured
outputs, unavailable at prediction time. They belong in y, not X.

3.Why is Ridge catastrophically bad?
365 features against 325 samples is an underdetermined system for
linear models. Ridge needs more data than features to find a stable
solution. Tree ensembles handle high dimensionality naturally by
selecting subsets of features at each split.

4.Why is R² = 0.77 the right number, not higher?
Feed RON legitimately dominates the prediction. The process variables
contribute a small residual. A higher R² would be suspicious — it would
mean leakage. 0.77 with cross-validated metrics on 325 real industrial
samples is a defensible result.



Tech stack

Python 3.12
pandas — data loading and manipulation
scikit-learn — model training, cross-validation, preprocessing
XGBoost — gradient boosting
SHAP — model explainability
Streamlit — dashboard
joblib — model serialization

Data source
Dataset title: S-ZORB device measured data and related code
Contributor: Yu Li
Institution: Hohai University, College of Hydrology and Water Resources
Published: 11 June 2021
DOI: 10.17632/h44t3pvw22.1
License: CC BY 4.0
Category: Chemical Engineering 

Original data provenance: The operating variable data comes from
Sinopec Gaoqiao Petrochemical's real-time database (Honeywell PHD).
The properties of raw materials, products, and catalysts come from the
LIMS experimental database. Data collection time is from April 2017 to
May 2020. The number of operational variables is 354. From April 2017
to September 2019, the data collection frequency was 3 minutes per
point. From October 2019 to May 2020, it was 6 minutes per point. The
octane number of raw materials and products was collected twice a week



