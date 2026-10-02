# Anuman

**A soft sensor for the S-Zorb gasoline desulfurization unit.**

Predicts product octane number (RON) in real time from all inputs available at the unit — feed quality and process conditions — closing the two-day gap between lab measurements.

**Live demo:** [anuman-soft-sensor.streamlit.app](https://anuman-soft-sensor.streamlit.app)
**Dataset:** [S-Zorb Sensor Data — Mendeley DOI 10.17632/h44t3pvw22.1](https://doi.org/10.17632/h44t3pvw22.1)

---

## 1. Problem Statement

S-Zorb removes sulfur from gasoline, but removing sulfur also destroys octane. Product RON is measured only twice a week in the lab. Between measurements, operators are blind — if RON drifts below spec, off-spec gasoline is produced without warning.

Anuman estimates product RON continuously. Feed RON sets the baseline; process variables explain the residual octane loss that operators can control. The system supports engineers, not replaces them.

---

## 2. Industrial Background

S-Zorb is a licensed refinery process for gasoline desulfurization. The unit removes sulfur to meet regulatory limits while minimizing octane loss. Key operational variables:

- Reactor temperature and pressure
- Hydrogen-to-oil ratio
- Adsorbent circulation rate
- Feed sulfur and octane content

Product RON is measured in the lab twice a week. Between readings, the unit operates open-loop on quality.

---

## 3. Dataset Source

**Name:** S-ZORB device measured data and related code
**Contributor:** Yu Li, Hohai University
**Provenance:** Sinopec Gaoqiao Petrochemical — Honeywell PHD historian
**Time span:** April 2017 – May 2020
**Samples:** 325 lab measurements
**Features:** 354 operating variables
**License:** CC BY 4.0
**Link:** https://doi.org/10.17632/h44t3pvw22.1

Sampling: every 3 minutes (Apr 2017 – Sep 2019), every 6 minutes (Oct 2019 – May 2020). Octane number collected twice a week.

---

## 4. Methodology

Real industrial data (325 samples × 365 features)
→ Clean Excel (3-row header, merged cells, dash markers)
→ Parse mixed-format timestamps
→ Split features / targets / metadata
→ Exclude product_* columns (leakage prevention)
→ Train 4 models with 5-fold TimeSeriesSplit
→ Select best by cross-validated RMSE
→ SHAP attribution per prediction
→ Operator dashboard (Streamlit)

---

## 5. ML Model

**Winner:** Random Forest.

| Model              | RMSE       | R²          |
|--------------------|------------|-------------|
| Ridge              | 5.6791     | -436.05     |
| **Random Forest**  | **0.3850** | **0.7734**  |
| Gradient Boosting  | 0.4097     | 0.7665      |
| XGBoost            | 0.4423     | 0.6566      |

**Why Random Forest:** handles high dimensionality, robust to outliers, no feature scaling needed, works well with small datasets.

**Why not neural networks:** 325 samples is too few. Neural nets need thousands. They'd overfit.

**Why not Ridge:** 365 features against 325 samples is underdetermined for linear models.

---

## 6. Results

Random Forest achieved **R² = 0.77**, **RMSE = 0.385 octane points** on unseen test data.

**SHAP findings:** Feed RON dominates the prediction (+0.81). Process variables contribute a small but real residual. This is physically correct — S-Zorb cannot create octane, only remove sulfur.

**Negative results:** RON loss and product sulfur both failed cross-validation (R² < 0). Twice-weekly composite lab samples don't align temporally with snapshot process variables.

---

## 7. Business Model

**Customer:** Refineries running S-Zorb, hydrotreaters, or similar catalytic units.

**Value:** Early warning of quality drift, reducing off-spec production events.

**Revenue:** Annual SaaS licence per unit, tiered by throughput.

**Channel:** Sold through Connected Performance Services portfolios alongside existing APC and historian integrations.

---

## 8. Limitations

- No labelled fault events. Evaluation is cross-validated regression, not classification.
- Twice-weekly lab sampling limits temporal alignment.
- Process variables contribute a small residual — the model is a quality tracker, not a controller.
- Model retraining required as the unit ages.

---

## 9. How to Run

Clone the repository:

    git clone https://github.com/YOUR_USERNAME/anuman.git
    cd anuman

Create and activate a virtual environment:

    python -m venv venv
    venv\Scripts\activate          # Windows
    source venv/bin/activate       # macOS / Linux

Install dependencies:

    pip install -r requirements.txt

Download the raw Excel from the DOI above, place in `data/raw/`, then run the pipeline:

    python src/data_loader.py
    python src/train.py
    python src/insights.py

Launch the dashboard:

    streamlit run app.py --server.headless=true --browser.gatherUsageStats=false

Open http://localhost:8501.

---

## 10. Tech Stack

Python 3.12 · pandas · scikit-learn · XGBoost · SHAP · Streamlit · joblib

