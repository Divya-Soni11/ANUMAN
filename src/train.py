"""
train.py

Trains soft-sensor models to predict product RON and sulfur content
from S-Zorb process variables.

We try a few common regression models, use TimeSeriesSplit to keep
the train/test split honest (no leaking future data into training),
and pick the best one by cross-validated RMSE.

Saves the winning model and its metadata to models/ so the Streamlit
app can load it later.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import xgboost as xgb
from data_loader import load_and_prepare, PROJECT_ROOT

# Where to save the trained model and its metadata.
# Anchored to the project root so it works from any folder.
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
MODEL_FILE = os.path.join(MODEL_DIR, 'best_model.joblib')
METADATA_FILE = os.path.join(MODEL_DIR, 'model_metadata.json')



# We'll predict one target at a time (RON first, then sulfur).
# Multi-output regression adds complexity we don't need tonight.
PRIMARY_TARGET = 'product_octane_number_ron'

# TimeSeriesSplit parameters — 5 folds is standard for small datasets
N_SPLITS = 5
RANDOM_STATE = 42


def get_candidate_models():
    """
    Return a dictionary of models to try.
    
    We include one linear baseline (Ridge) and three tree-based models.
    Ridge is there to check whether a simple linear fit is good enough —
    if it is, we don't need complexity.
    """
    return {
        'Ridge': Pipeline([
            ('scale', StandardScaler()),
            ('model', Ridge(alpha=100.0, random_state=RANDOM_STATE)),
        ]),
        'Random Forest': RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def evaluate_models(X, y):
    """
    Run cross-validation for each model and print a comparison table.
    
    Uses TimeSeriesSplit so training folds always come before test folds —
    important for industrial data where drift and fouling matter over time.
    """
    models = get_candidate_models()
    splitter = TimeSeriesSplit(n_splits=N_SPLITS)

    results = {}

    print(f"\nEvaluating {len(models)} models with {N_SPLITS}-fold TimeSeriesSplit")
    print(f"{'Model':<20} {'RMSE':>8} {'MAE':>8} {'R2':>8}")
    print("-" * 50)

    for name, model in models.items():
        # RMSE and MAE — negative sign flips sklearn's "higher is better"
        rmse_scores = -cross_val_score(
            model, X, y, cv=splitter,
            scoring='neg_root_mean_squared_error', n_jobs=-1
        )
        mae_scores = -cross_val_score(
            model, X, y, cv=splitter,
            scoring='neg_mean_absolute_error', n_jobs=-1
        )
        r2_scores = cross_val_score(
            model, X, y, cv=splitter,
            scoring='r2', n_jobs=-1
        )

        results[name] = {
            'rmse_mean': rmse_scores.mean(),
            'rmse_std':  rmse_scores.std(),
            'mae_mean':  mae_scores.mean(),
            'r2_mean':   r2_scores.mean(),
            'r2_std':    r2_scores.std(),
        }

        print(f"{name:<20} {rmse_scores.mean():>8.4f} "
              f"{mae_scores.mean():>8.4f} {r2_scores.mean():>8.4f}")

    return results, models


def pick_best_model(results, models):
    """Return the model with the lowest mean RMSE."""
    best_name = min(results, key=lambda n: results[n]['rmse_mean'])
    print(f"\nBest model: {best_name} "
          f"(RMSE {results[best_name]['rmse_mean']:.4f})")
    return best_name, models[best_name]


def save_model(model, name, results, feature_names, target_name):
    """
    Save the trained model and its metadata (feature names, performance,
    training info) so the Streamlit app knows what it's looking at.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, MODEL_FILE)

    metadata = {
        'model_type':     name,
        'target':         target_name,
        'feature_names':  list(feature_names),
        'performance':    results[name],
        'n_features':     len(feature_names),
    }
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved model to {MODEL_FILE}")
    print(f"Saved metadata to {METADATA_FILE}")


def main():
    # --- Load the cleaned data ---
    df, X, y, meta = load_and_prepare()

    # --- Sort by time, ascending (important for TimeSeriesSplit) ---
    # The raw file lists samples newest-first; flip that.
    order = meta['Time'].argsort().values
    X = X.iloc[order].reset_index(drop=True)
    y = y.iloc[order].reset_index(drop=True)
    meta = meta.iloc[order].reset_index(drop=True)

    print(f"\nSorted chronologically: {meta['Time'].iloc[0]} "
          f"to {meta['Time'].iloc[-1]}")

    # --- Pick the target column ---
    if PRIMARY_TARGET not in y.columns:
        raise ValueError(f"Target '{PRIMARY_TARGET}' not found in {y.columns.tolist()}")
    y_target = y[PRIMARY_TARGET]

    print(f"Predicting: {PRIMARY_TARGET}")
    print(f"  Range: {y_target.min():.2f} to {y_target.max():.2f}, "
          f"mean {y_target.mean():.2f}")

    # --- Evaluate all candidates ---
    results, models = evaluate_models(X, y_target)

    # --- Pick the winner and refit on the full dataset ---
    best_name, best_model = pick_best_model(results, models)

    print(f"\nRefitting {best_name} on the full dataset...")
    best_model.fit(X, y_target)

    # --- Save ---
    save_model(best_model, best_name, results, X.columns, PRIMARY_TARGET)

    print("\nDone.")


if __name__ == '__main__':
    main()