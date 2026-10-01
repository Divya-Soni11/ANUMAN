"""
app.py — Anuman
Soft sensor dashboard for the S-Zorb gasoline desulfurization unit.

Run:  streamlit run app.py --server.headless=true --browser.gatherUsageStats=false
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import json
from data_loader import load_and_prepare, PROJECT_ROOT

st.set_page_config(page_title="Anuman · S-Zorb Soft Sensor",
                   layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,300..900&family=Manrope:wght@300..800&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Manrope', sans-serif;
        color: #202020;
        -webkit-font-smoothing: antialiased;
    }
    .stApp { background: #faf9f6; }

    .block-container {
        padding-top: 0;
        padding-bottom: 4rem;
        max-width: 1280px;
    }
    header, footer, #MainMenu { visibility: hidden; }

    .nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1.4rem 0;
        border-bottom: 1px solid #e5e0d4;
        margin-bottom: 4.5rem;
    }
    .nav-mark {
        font-family: 'Fraunces', serif;
        font-size: 1.05rem;
        font-weight: 500;
        letter-spacing: -0.01em;
        color: #202020;
    }
    .nav-meta {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #8a8578;
    }
    .nav-live {
        display: flex; align-items: center; gap: 0.55rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #6b7a4e;
    }
    .live-dot {
        width: 6px; height: 6px;
        border-radius: 50%;
        background: #6b7a4e;
        box-shadow: 0 0 0 4px rgba(107,122,78,0.14);
    }

    .hero {
        text-align: center;
        padding: 1rem 0 5rem 0;
    }
    .hero-eyebrow {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        color: #9e1b23;
        margin-bottom: 1.9rem;
    }
    .hero-name {
        font-family: 'Fraunces', serif;
        font-weight: 500;
        font-size: 8rem;
        line-height: 0.86;
        letter-spacing: -0.045em;
        color: #9e1b23;
        margin: 0;
        font-variation-settings: "opsz" 144;
    }
    .hero-italic {
        font-family: 'Fraunces', serif;
        font-style: italic;
        font-weight: 300;
        font-size: 1.3rem;
        color: #4a4438;
        margin-top: 1.6rem;
        letter-spacing: -0.005em;
    }
    .hero-rule {
        width: 44px;
        height: 1px;
        background: #d9a72c;
        margin: 2.2rem auto 2.2rem auto;
    }
    .hero-body {
        max-width: 640px;
        margin: 0 auto;
        font-size: 1rem;
        line-height: 1.75;
        color: #5c5648;
    }

    .sect {
        display: flex;
        align-items: baseline;
        gap: 1.4rem;
        margin: 0 0 2rem 0;
        padding-bottom: 1rem;
        border-bottom: 1px solid #e5e0d4;
    }
    .sect-num {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.14em;
        color: #9e1b23;
    }
    .sect-title {
        font-family: 'Fraunces', serif;
        font-weight: 400;
        font-size: 1.6rem;
        letter-spacing: -0.015em;
        color: #202020;
        margin: 0;
    }

    .card {
        background: #ffffff;
        border: 1px solid #e8e3d7;
        border-radius: 3px;
        padding: 1.6rem 1.5rem 1.5rem 1.5rem;
        height: 100%;
        position: relative;
    }
    .card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: #d9a72c;
    }
    .card-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.66rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: #9e1b23;
        margin-bottom: 1.1rem;
    }
    .card-value {
        font-family: 'Fraunces', serif;
        font-weight: 400;
        font-size: 2.6rem;
        line-height: 0.95;
        letter-spacing: -0.025em;
        color: #202020;
        font-variant-numeric: tabular-nums lining-nums;
    }
    .card-value.small {
        font-size: 1.5rem;
        line-height: 1.2;
    }
    .card-note {
        font-family: 'Manrope', sans-serif;
        font-size: 0.78rem;
        color: #8a8578;
        margin-top: 0.95rem;
        font-weight: 500;
    }

    .status {
        padding: 1.3rem 1.5rem;
        font-size: 0.94rem;
        line-height: 1.7;
        border-left: 2px solid;
        background: #ffffff;
    }
    .status b {
        font-family: 'Fraunces', serif;
        font-weight: 500;
        font-style: italic;
        letter-spacing: -0.005em;
    }
    .status-ok    { border-color: #6b7a4e; color: #354026; }
    .status-warn  { border-color: #9e1b23; color: #6b1218; }
    .status-info  { border-color: #d9a72c; color: #5c4a1e; }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 1px solid #e5e0d4;
        margin-bottom: 2.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        padding: 15px 28px 15px 0;
        margin-right: 28px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.74rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #a39a86;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        color: #9e1b23 !important;
        border-bottom: 2px solid #9e1b23 !important;
        font-weight: 600;
    }

    .stSlider [data-baseweb="slider"] div[role="slider"] {
        background-color: #9e1b23 !important;
    }

    .stCaption, [data-testid="stCaptionContainer"] {
        color: #8a8578 !important;
        font-size: 0.78rem !important;
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 0.03em;
    }

    .foot {
        margin-top: 5rem;
        padding-top: 2rem;
        border-top: 1px solid #e5e0d4;
        display: flex;
        justify-content: space-between;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #a39a86;
    }

    .stDataFrame { font-size: 0.88rem; }
    .stLineChart, .stBarChart { border: 1px solid #e8e3d7; border-radius: 3px; padding: 0.5rem; background: #ffffff; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_assets():
    model = joblib.load(os.path.join(PROJECT_ROOT, 'models', 'best_model.joblib'))
    meta = json.load(open(os.path.join(PROJECT_ROOT, 'models', 'model_metadata.json')))
    return model, meta


@st.cache_data
def load_samples():
    df, X, y, meta = load_and_prepare()
    order = meta['Time'].argsort().values
    return (X.iloc[order].reset_index(drop=True),
            y.iloc[order].reset_index(drop=True),
            meta.iloc[order].reset_index(drop=True))


model, metadata = load_assets()
X, y, meta_df = load_samples()


# --- nav ---
st.markdown('''
<div class="nav">
    <div class="nav-mark">Anuman</div>
    <div class="nav-meta">S-Zorb · Sinopec · 2017–2020</div>
    <div class="nav-live"><span class="live-dot"></span>Model live</div>
</div>
''', unsafe_allow_html=True)


# --- hero ---
st.markdown('''
<div class="hero">
    <div class="hero-eyebrow">Industrial Soft Sensing · S-Zorb Unit</div>
    <h1 class="hero-name">ANUMAN</h1>
    <div class="hero-italic">product quality, between lab measurements.</div>
    <div class="hero-rule"></div>
    <p class="hero-body">
        Predicts gasoline octane number in real time from refinery process
        variables. Trained on three years of historian data from a Sinopec
        S-Zorb unit, closing the two-day gap between lab reports.
    </p>
</div>
''', unsafe_allow_html=True)


# --- headline strip ---
st.markdown('''
<div class="sect">
    <div class="sect-num">01</div>
    <div class="sect-title">Model at a glance</div>
</div>
''', unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(f'''
        <div class="card">
            <div class="card-label">Algorithm</div>
            <div class="card-value small">{metadata['model_type']}</div>
            <div class="card-note">Time-series validated</div>
        </div>
    ''', unsafe_allow_html=True)

with m2:
    st.markdown(f'''
        <div class="card">
            <div class="card-label">R² Score</div>
            <div class="card-value">{metadata['performance']['r2_mean']:.3f}</div>
            <div class="card-note">Unseen test data</div>
        </div>
    ''', unsafe_allow_html=True)

with m3:
    st.markdown(f'''
        <div class="card">
            <div class="card-label">RMSE</div>
            <div class="card-value">{metadata['performance']['rmse_mean']:.3f}</div>
            <div class="card-note">Octane points</div>
        </div>
    ''', unsafe_allow_html=True)

with m4:
    st.markdown(f'''
        <div class="card">
            <div class="card-label">Input Features</div>
            <div class="card-value">{metadata['n_features']}</div>
            <div class="card-note">Process variables</div>
        </div>
    ''', unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)


# --- tabs ---
tab_pred, tab_shap, tab_perf, tab_recs, tab_limits = st.tabs(
    ["Prediction", "Insights", "Model Comparison", "Recommendations", "Limitations"]
)


# ---------- prediction ----------
with tab_pred:
    st.markdown('''
        <div class="sect">
            <div class="sect-num">02</div>
            <div class="sect-title">Select a sample and view the prediction</div>
        </div>
    ''', unsafe_allow_html=True)

    idx = st.slider("Sample index", 0, len(X) - 1, 0, label_visibility="collapsed")
    st.caption(f"Timestamp — {meta_df.iloc[idx]['Time']}")

    row = X.iloc[[idx]]
    pred = model.predict(row)[0]
    actual = y.iloc[idx]['product_octane_number_ron']
    err = pred - actual

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'''
            <div class="card">
                <div class="card-label">Predicted RON</div>
                <div class="card-value">{pred:.2f}</div>
                <div class="card-note">From process variables</div>
            </div>
        ''', unsafe_allow_html=True)
    with c2:
        st.markdown(f'''
            <div class="card">
                <div class="card-label">Lab-Measured RON</div>
                <div class="card-value">{actual:.2f}</div>
                <div class="card-note">Ground truth</div>
            </div>
        ''', unsafe_allow_html=True)
    with c3:
        st.markdown(f'''
            <div class="card">
                <div class="card-label">Residual</div>
                <div class="card-value">{err:+.2f}</div>
                <div class="card-note">Prediction error</div>
            </div>
        ''', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('''
        <div class="status status-info">
        The lab reports RON twice a week. Between measurements, this model
        estimates product quality continuously — allowing operators to react
        before off-spec product is produced.
        </div>
    ''', unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('''
        <div class="sect">
            <div class="sect-num">03</div>
            <div class="sect-title">Product RON · recent samples</div>
        </div>
    ''', unsafe_allow_html=True)

    start = max(0, idx - 60)
    trend = pd.DataFrame({
        "RON": y['product_octane_number_ron'].iloc[start:idx + 1].values,
        "Sample": list(range(start, idx + 1)),
    }).set_index("Sample")
    st.line_chart(trend, color="#9e1b23", height=300)


# ---------- insights (SHAP) ----------
with tab_shap:
    st.markdown('''
        <div class="sect">
            <div class="sect-num">04</div>
            <div class="sect-title">Why the model made this prediction</div>
        </div>
    ''', unsafe_allow_html=True)

    st.write("SHAP attributes each prediction to its input features. "
             "Positive values push the prediction up; negative values pull it down.")

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(row)
    if isinstance(sv, list):
        sv = sv[0]
    sv = np.array(sv).flatten()

    contrib = pd.DataFrame({
        "Feature": metadata['feature_names'],
        "Value": row.values.flatten(),
        "Impact": sv,
    })
    contrib['abs'] = contrib['Impact'].abs()
    contrib = contrib.sort_values('abs', ascending=False).head(10).reset_index(drop=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.bar_chart(contrib.set_index("Feature")[["Impact"]], color="#d9a72c", height=400)

    st.markdown("<br>", unsafe_allow_html=True)
    st.dataframe(contrib[["Feature", "Value", "Impact"]],
                 use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('''
        <div class="status status-info">
        Feature importance is not causal proof. It indicates which variables
        the model relies on, not which variables physically cause the outcome.
        </div>
    ''', unsafe_allow_html=True)


# ---------- model comparison ----------
with tab_perf:
    st.markdown('''
        <div class="sect">
            <div class="sect-num">05</div>
            <div class="sect-title">Model comparison</div>
        </div>
    ''', unsafe_allow_html=True)

    st.write("Five-fold TimeSeriesSplit cross-validation on 325 refinery samples.")

    perf = pd.DataFrame({
        "Model": ["Ridge", "Random Forest", "Gradient Boosting", "XGBoost"],
        "RMSE": [5.6791, 0.3850, 0.4097, 0.4423],
        "MAE": [3.0460, 0.2684, 0.2888, 0.3336],
        "R²": [-436.05, 0.7734, 0.7665, 0.6566],
    })
    perf["Selected"] = ["", "Yes", "", ""]
    st.dataframe(
        perf.style.format({"RMSE": "{:.4f}", "MAE": "{:.4f}", "R²": "{:.4f}"}),
        use_container_width=True, hide_index=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('''
        <div class="status status-info">
        <b>Recommended model.</b> Random Forest, selected based on
        cross-validated RMSE. Ridge regression failed catastrophically —
        365 features against 325 samples is an underdetermined system for
        linear models. Tree ensembles handle high dimensionality naturally.
        </div>
    ''', unsafe_allow_html=True)


# ---------- recommendations ----------
with tab_recs:
    st.markdown('''
        <div class="sect">
            <div class="sect-num">06</div>
            <div class="sect-title">Operator Recommendations</div>
        </div>
    ''', unsafe_allow_html=True)

    SPEC_LOW = 88.5
    DRIFT_WARN = 0.3

    window = 20
    start = max(0, idx - window)
    recent_preds = [model.predict(X.iloc[[i]])[0] for i in range(start, idx + 1)]
    recent_avg = np.mean(recent_preds) if len(recent_preds) > 1 else pred
    drift = pred - recent_avg

    if pred < SPEC_LOW:
        st.markdown(f'''
            <div class="status status-warn">
            <b>Alert.</b> Predicted RON is {pred:.2f}, below the product
            specification limit of {SPEC_LOW:.2f}. Schedule a lab sample
            and review recent operating conditions.
            </div>
        ''', unsafe_allow_html=True)
    elif drift < -DRIFT_WARN:
        st.markdown(f'''
            <div class="status status-warn">
            <b>Watch.</b> Predicted RON is {drift:+.2f} points below the
            recent {window}-sample average. Product quality is drifting.
            Monitor the next lab result.
            </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
            <div class="status status-ok">
            <b>Within spec.</b> Predicted RON is {pred:.2f}, above the
            specification limit of {SPEC_LOW:.2f}, and stable relative to
            recent history.
            </div>
        ''', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="sect" style="border:none; padding:0; margin-bottom:1rem;">'
                '<div class="sect-title" style="font-size:1.1rem;">Suggested actions</div>'
                '</div>', unsafe_allow_html=True)

    if pred < SPEC_LOW:
        actions = [
            "Collect an immediate lab sample to verify the prediction.",
            "Review feed quality — if feed RON has dropped recently, product RON will follow.",
            "Check whether recent changes to reactor conditions coincide with the drift.",
        ]
    elif drift < -DRIFT_WARN:
        actions = [
            "Continue monitoring. If the next lab result confirms the drift, review unit severity settings.",
            "Confirm no feed change or upset occurred in the last 8 hours.",
        ]
    else:
        actions = [
            "No operator action required at this time.",
            "Continue routine monitoring.",
        ]

    for a in actions:
        st.markdown(f"- {a}")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="sect" style="border:none; padding:0; margin-bottom:1rem;">'
                '<div class="sect-title" style="font-size:1.1rem;">Supporting context</div>'
                '</div>', unsafe_allow_html=True)

    ctx1, ctx2 = st.columns(2)
    with ctx1:
        feed_ron = row["raw_material_octane_number_ron"].values[0]
        st.markdown(f'''
            <div class="card">
                <div class="card-label">Feed RON · current sample</div>
                <div class="card-value small">{feed_ron:.2f}</div>
                <div class="card-note">Dominant driver of prediction</div>
            </div>
        ''', unsafe_allow_html=True)
    with ctx2:
        st.markdown(f'''
            <div class="card">
                <div class="card-label">Drift vs. {window}-sample average</div>
                <div class="card-value small">{drift:+.3f}</div>
                <div class="card-note">Octane points</div>
            </div>
        ''', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('''
        <div class="status status-info">
        These recommendations are based on prediction, drift, and threshold
        logic. They are advisory only — not a substitute for the plant DCS
        or APC system. A real deployment would integrate these outputs into
        the operator console and would require sign-off from process engineering.
        </div>
    ''', unsafe_allow_html=True)


# ---------- limitations ----------
with tab_limits:
    st.markdown('''
        <div class="sect">
            <div class="sect-num">07</div>
            <div class="sect-title">What this model can and cannot do</div>
        </div>
    ''', unsafe_allow_html=True)

    st.markdown('''
        <div class="status status-ok">
        <b>Works.</b> Predicts product RON with R² = 0.77 on unseen test data
        using time-series cross-validation.
        </div>
    ''', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('''
        <div class="status status-info">
        <b>Key insight.</b> SHAP shows feed RON dominates the prediction.
        This is physically correct: S-Zorb cannot create octane, only remove
        sulfur. Feed quality is the primary driver of product quality.
        </div>
    ''', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('''
        <div class="status status-warn">
        <b>Negative results.</b> Two targets failed cross-validation. RON
        loss (R² &lt; 0) is dominated by lab noise. Product sulfur (R² &lt; 0)
        suffers from temporal misalignment between twice-weekly composite
        lab samples and snapshot process variables.
        </div>
    ''', unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('''
        <div class="sect">
            <div class="sect-num">08</div>
            <div class="sect-title">Why sulfur prediction fails</div>
        </div>
    ''', unsafe_allow_html=True)

    st.write("With 325 lab samples over 3 years, and each lab value being a "
             "composite of product collected over hours, the process variable "
             "snapshots at the lab timestamp do not align temporally with the "
             "lab measurement itself.")
    st.write("This is a dataset limitation, not a modeling failure. It is the "
             "reason industry deploys online analyzers rather than soft sensors "
             "for high-frequency quality variables.")


st.markdown('''
<div class="foot">
    <div>Anuman · Industrial Soft Sensing</div>
    <div>Honeywell PHD · Random Forest · SHAP</div>
</div>
''', unsafe_allow_html=True)
