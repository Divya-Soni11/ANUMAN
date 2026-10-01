"""
insights.py

Explains model predictions using SHAP (SHapley Additive exPlanations).

The trained model is a black box — it takes 366 process variables and
outputs a predicted RON. Operators can't act on a number alone; they
need to know WHICH variables drove the prediction and by how much.

SHAP gives each feature a contribution score for a specific prediction:
    +0.42  reactor temperature is low → pushes RON down
    -0.18  H2-to-oil ratio is high    → pushes RON up
    ...
The sum of all contributions equals the prediction minus the baseline.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import shap

from data_loader import PROJECT_ROOT


MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
MODEL_FILE = os.path.join(MODEL_DIR, 'best_model.joblib')
METADATA_FILE = os.path.join(MODEL_DIR, 'model_metadata.json')


def load_model_and_metadata():
    """Load the trained model and its saved metadata (feature names, etc.)."""
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            f"No trained model at {MODEL_FILE}. Run train.py first."
        )

    model = joblib.load(MODEL_FILE)

    with open(METADATA_FILE) as f:
        metadata = json.load(f)

    return model, metadata


def build_explainer(model):
    """
    Build a SHAP explainer for the model.

    Random Forest is tree-based, so TreeExplainer is fast and exact.
    If a non-tree model somehow wins later, this falls back to a
    slower but universal KernelExplainer.
    """
    try:
        return shap.TreeExplainer(model)
    except Exception:
        # Universal fallback — works on any sklearn-compatible model
        # but is much slower.
        return shap.KernelExplainer(model.predict, np.zeros((1, model.n_features_in_)))


def explain_one(explainer, feature_row, feature_names, top_n=8):
    """
    Explain a single prediction.

    Parameters
    ----------
    explainer : shap explainer
    feature_row : pd.Series or 1-row DataFrame
        One sample of process data.
    feature_names : list of str
        Column names, in the same order as the model expects.
    top_n : int
        How many top contributions to return.

    Returns
    -------
    contributions : pd.DataFrame
        Columns: feature, value, shap_value, direction
        Sorted by absolute SHAP value, largest first, top_n rows.
    """
    # Convert to a 2D array the way sklearn/SHAP expect
    X_row = pd.DataFrame([feature_row], columns=feature_names)

    shap_values = explainer.shap_values(X_row)

    # TreeExplainer sometimes returns a list (one array per output).
    # For regression we want the first (and only) output.
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.asarray(shap_values).flatten()

    contributions = pd.DataFrame({
        'feature':    feature_names,
        'value':      X_row.iloc[0].values,
        'shap_value': shap_values,
    })

    # Direction is a human-readable hint
    contributions['direction'] = np.where(
        contributions['shap_value'] > 0,
        'pushes prediction up',
        'pushes prediction down',
    )

    # Sort by absolute impact and take the top N
    contributions['abs_shap'] = contributions['shap_value'].abs()
    contributions = contributions.sort_values('abs_shap', ascending=False)
    contributions = contributions.head(top_n).reset_index(drop=True)

    return contributions


def format_operator_message(prediction, contributions, target_name,
                             baseline=None):
    """
    Turn SHAP contributions into a plain-language message that a
    control-room operator could read.
    """
    lines = []
    lines.append(f"Predicted {target_name}: {prediction:.2f}")

    if baseline is not None:
        delta = prediction - baseline
        direction = "above" if delta > 0 else "below"
        lines.append(f"  ({abs(delta):.2f} {direction} the training average "
                     f"of {baseline:.2f})")

    lines.append("")
    lines.append("Top contributing factors:")

    for _, row in contributions.iterrows():
        sign = "+" if row['shap_value'] > 0 else ""
        lines.append(
            f"  {row['feature']:<45s} "
            f"value={row['value']:>10.3f}  "
            f"impact={sign}{row['shap_value']:.3f}"
        )

    return "\n".join(lines)


def main():
    """Demo: load the model, pick a sample, show the explanation."""
    from data_loader import load_and_prepare

    print("Loading model and data...")
    model, metadata = load_model_and_metadata()
    print(f"  Model:   {metadata['model_type']}")
    print(f"  Target:  {metadata['target']}")
    print(f"  Trained RMSE: {metadata['performance']['rmse_mean']:.4f}")
    print(f"  Trained R2:   {metadata['performance']['r2_mean']:.4f}")

    df, X, y, meta = load_and_prepare()
    feature_names = metadata['feature_names']

    # Align features with the order used during training
    X = X[feature_names]

    # Build the explainer once — reuse for many predictions
    print("\nBuilding SHAP explainer...")
    explainer = build_explainer(model)

    # Explain the very first sample as a demo
    print("\n" + "=" * 70)
    print("EXAMPLE EXPLANATION — sample 0")
    print("=" * 70)

    row = X.iloc[0]
    prediction = model.predict(pd.DataFrame([row], columns=feature_names))[0]
    baseline = y[metadata['target']].mean()

    contributions = explain_one(explainer, row, feature_names, top_n=8)

    print()
    print(format_operator_message(
        prediction=prediction,
        contributions=contributions,
        target_name=metadata['target'],
        baseline=baseline,
    ))

    # Show first two samples side by side to prove it works per-sample
    print("\n" + "=" * 70)
    print("EXAMPLE EXPLANATION — sample 1")
    print("=" * 70)

    row = X.iloc[1]
    prediction = model.predict(pd.DataFrame([row], columns=feature_names))[0]
    contributions = explain_one(explainer, row, feature_names, top_n=5)

    print()
    print(format_operator_message(
        prediction=prediction,
        contributions=contributions,
        target_name=metadata['target'],
        baseline=baseline,
    ))


if __name__ == '__main__':
    main()