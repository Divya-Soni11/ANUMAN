"""
data_loader.py

Loads and cleans the S-Zorb raw dataset.

The raw Excel file has a messy structure: three header rows, merged cells
for group names, and dashes used as missing-value markers. This script
parses it properly and produces a cleaned CSV plus separate feature and
target dataframes.

S-Zorb is a UOP-licensed gasoline desulfurization process. The product
RON (octane) and sulfur content are only measured twice a week in the
lab, which is why we want a soft sensor for them.
"""

import os
import pandas as pd
import numpy as np


# ---------------------------------------------------------------
# File paths
# ---------------------------------------------------------------
# File paths
# ---------------------------------------------------------------
# The script might be run from the project root (ANUMAN/) or from
# inside src/. To make paths work in both cases, we anchor them to
# the project root, not the current working directory.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_FILE = os.path.join(PROJECT_ROOT, 'data', 'raw', 'Raw data .xlsx')
CLEAN_FILE = os.path.join(PROJECT_ROOT, 'data', 'processed', 's_zorb_clean.csv')

# Column names we'll use downstream
TIME_COL = 'Time'
SAMPLE_ID_COL = 'Sample numbur'

# What we're trying to predict (product quality variables from the lab)
TARGETS = ['product_octane_number_ron', 'product_sulfur_content_ug_per_g', 'product_ron_loss',]

# The raw file uses '-' to mean "no measurement here"
MISSING_MARKERS = ['-', '', 'NA', 'N/A']


# ---------------------------------------------------------------
# The Excel file has merged cells for group headers, so pandas reads
# them as "Unnamed: X" except for the first cell of each group.
# These are the manual renames based on what I saw in Excel row 3.
# ---------------------------------------------------------------
RENAMES = {
    # Raw material properties (cols 2-8)
    'Raw material properties':          'raw_material_sulfur_content_ug_per_g',
    'Unnamed: 3':                       'raw_material_octane_number_ron',
    'Unnamed: 4':                       'raw_material_saturated_hydrocarbon_vol_pct',
    'Unnamed: 5':                       'raw_material_olefin_vol_pct',
    'Unnamed: 6':                       'raw_material_aromatics_vol_pct',
    'Unnamed: 7':                       'raw_material_bromine_number_gbr_per_100g',
    'Unnamed: 8':                       'raw_material_density_at_20c_kg_per_m3',

    # Product properties (cols 9-11) — these are our targets
    'Product properties':               'product_sulfur_content_ug_per_g',
    'Unnamed: 10':                      'product_octane_number_ron',
    'Unnamed: 11':                      'product_ron_loss',

    # Pending adsorbent (cols 12-13)
    'pending adsorbent properties':     'pending_adsorbent_coke_wt_pct',
    'Unnamed: 13':                      'pending_adsorbent_sulfur_wt_pct',

    # Regenerated adsorbent (cols 14-15)
    'regenerated adsorbent properties': 'regenerated_adsorbent_coke_wt_pct',
    'Unnamed: 15':                      'regenerated_adsorbent_sulfur_wt_pct',
}


def load_raw(path=RAW_FILE):
    """
    Read the raw Excel file and fix up the column names.

    The file has the real variable names on Excel row 2 (which becomes
    the header), but rows 2-15 are group headers with merged cells.
    The specific names for those columns are on Excel row 3.
    """
    print(f"Reading {path}")

    # header=1 means use Excel row 2 (0-indexed) as the column names.
    df = pd.read_excel(path, header=1)

    # Row 0 is now the sub-header row (Excel row 3). Save it for reference
    # before dropping it — we need those names for the merged-cell columns.
    sub_headers = df.iloc[0].tolist()

    # Rename columns: apply our manual map first, then fall back to
    # the sub-header names for any remaining "Unnamed: X" columns.
    new_names = []
    for i, old_name in enumerate(df.columns):
        if old_name in RENAMES:
            new_names.append(RENAMES[old_name])
        elif str(old_name).startswith('Unnamed:') and i < len(sub_headers):
            sub = str(sub_headers[i]).split('\n')[0].strip()
            if sub and sub.lower() != 'nan':
                # Clean it up so it's usable in code
                safe = sub.lower().replace(',', '_').replace(' ', '_')
                safe = safe.replace('(', '').replace(')', '')
                safe = safe.replace('/', '_per_').replace('%', 'pct')
                safe = safe.replace('μ', 'u').replace('.', '_')
                new_names.append(f'col_{i}_{safe}')
            else:
                new_names.append(f'col_{i}')
        else:
            new_names.append(old_name)

    df.columns = new_names

    # The first row was the sub-header — drop it now that we've used it
    df = df.iloc[1:].reset_index(drop=True)

    print(f"  Loaded shape: {df.shape}")
    return df


def clean(df):
    """
    Replace dash markers with NaN, parse the timestamp column, and
    force every other column to numeric.
    """
    df = df.copy()

        # Dashes mean "no measurement" — convert them to real NaN
    df = df.replace(MISSING_MARKERS, np.nan)

    # Parse the timestamp column.
    # format='mixed' parses each value on its own rather than forcing a
    # single format across the column. Needed here because some rows have
    # only a date ('2018/12/1') and others have date + time ('2020/5/26 8:00:00').
    if TIME_COL in df.columns:
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], format='mixed', errors='coerce')
        valid_times = df[TIME_COL].dropna()
        n_bad = df[TIME_COL].isnull().sum()
        print(f"  Time range: {valid_times.min()} to {valid_times.max()}")
        if n_bad > 0:
            print(f"  Note: {n_bad} row(s) with unparseable timestamp")

    # Force everything except Time and Sample ID to numeric.
    # errors='coerce' turns any non-numeric leftovers into NaN.
    for col in df.columns:
        if col in (TIME_COL, SAMPLE_ID_COL):
            continue
        df[col] = pd.to_numeric(df[col], errors='coerce')



    # Quick report on missing values so we know what we're dealing with
    n_missing = df.isnull().sum()
    cols_with_missing = (n_missing > 0).sum()
    print(f"  Columns with missing values: {cols_with_missing}")

    return df


def split(df):

    """
    Split into X (process variables), y (targets), and metadata.
    Features = process variables + raw material properties only.
    Product properties (RON, sulfur, RON loss) are EXCLUDED because
    they are only known after the lab result comes back — using them
    as inputs would be data leakage. They belong in y, not in X.
    """
    targets = [c for c in TARGETS if c in df.columns]
    if not targets:
        raise ValueError(f"No target columns found. Expected one of: {TARGETS}")

    # Product properties are lab-measured outputs of the unit. They are
    # not available at prediction time, so they must not be features.
    product_property_cols = [
        c for c in df.columns
        if c.startswith('product_')
    ]

    metadata_cols = [c for c in [SAMPLE_ID_COL, TIME_COL] if c in df.columns]

    # Everything not a target, product property, or metadata is a feature
    excluded = targets + product_property_cols + metadata_cols
    feature_cols = [c for c in df.columns if c not in excluded]

    X = df[feature_cols].copy()
    y = df[targets].copy()
    meta = df[metadata_cols].copy()

    print(f"  Features: {X.shape[0]} rows x {X.shape[1]} cols")
    print(f"  Targets:  {y.shape[0]} rows x {y.shape[1]} cols -> {targets}")
    print(f"  Excluded from features (would be leakage): {product_property_cols}")
    print(f"  Metadata: {meta.shape[0]} rows x {meta.shape[1]} cols")

    return X, y, meta


def save(df, path=CLEAN_FILE):
    """Write the cleaned dataframe to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    size_kb = os.path.getsize(path) / 1024
    print(f"  Saved to {path} ({size_kb:.0f} KB)")


def load_and_prepare(path=RAW_FILE):
    """
    Runs the full pipeline and returns the cleaned df plus X, y, meta.

    On Streamlit Cloud, the raw Excel file is not available. If the cleaned
    CSV already exists, load it directly instead of re-parsing the Excel.
    """
    if os.path.exists(CLEAN_FILE):
        print(f"  Loading cleaned CSV: {CLEAN_FILE}")
        df = pd.read_csv(CLEAN_FILE, parse_dates=[TIME_COL])
    else:
        df = load_raw(path)
        df = clean(df)
        save(df)

    X, y, meta = split(df)
    return df, X, y, meta


if __name__ == '__main__':
    df, X, y, meta = load_and_prepare()

    # Quick sanity check so we can eyeball the output
    print("\n--- first 3 rows of targets ---")
    print(y.head(3).to_string())

    print("\n--- first 3 rows of metadata ---")
    print(meta.head(3).to_string())

    print("\n--- first 3 rows of features (first 6 cols) ---")
    print(X.iloc[:3, :6].to_string())