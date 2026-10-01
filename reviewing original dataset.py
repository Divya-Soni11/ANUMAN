# import os


# CHANGE THIS to your downloaded folder path
# FOLDER = r"C:\Users\hp\Downloads\h44t3pvw22-1"

# def print_tree(startpath):
#     for root, dirs, files in os.walk(startpath):
#         level = root.replace(startpath, '').count(os.sep)
#         indent = '  ' * level
#         print(f"{indent}{os.path.basename(root)}/")
#         for file in files:
#             filepath = os.path.join(root, file)
#             size_mb = os.path.getsize(filepath) / (1024 * 1024)
#             print(f"{indent} {file}  ({size_mb:.2f} MB)")

# print_tree(FOLDER)


import pandas as pd
import numpy as np

# ============================================================
# STEP 1: Look at the raw structure (no header processing)
# ============================================================
print("=" * 70)
print("STEP 1: RAW STRUCTURE")
print("=" * 70)

raw = pd.read_excel('data/raw/Raw data .xlsx', header=None, nrows=6)
print(f"Raw shape (6 rows): {raw.shape}")
print()

for i in range(6):
    print(f"--- Excel Row {i+1} (first 20 cells) ---")
    for j in range(min(20, raw.shape[1])):
        val = raw.iloc[i, j]
        val_str = str(val)[:55] if pd.notna(val) else "NaN"
        print(f"  Col {j}: {val_str}")
    print()

# ============================================================
# STEP 2: Read with proper header handling
# ============================================================
print("=" * 70)
print("STEP 2: PROPER READ (header on row 2)")
print("=" * 70)

df = pd.read_excel('data/raw/Raw data .xlsx', header=1)
df = df.iloc[1:].reset_index(drop=True)  # drop note row

# Replace "-" with NaN across the whole dataframe
df = df.replace('-', np.nan)

# Try converting every column to numeric (non-convertible become NaN)
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print(f"Shape: {df.shape}")
print(f"Numeric columns: {df.select_dtypes(include=[np.number]).shape[1]}")
print(f"Non-numeric columns: {df.select_dtypes(exclude=[np.number]).shape[1]}")

# ============================================================
# STEP 3: Identify candidate target columns
# ============================================================
print()
print("=" * 70)
print("STEP 3: CANDIDATE TARGET COLUMNS (Product properties area)")
print("=" * 70)

# Show columns 9-15 (where "Product properties" group lives)
for i in range(9, 16):
    col = df.columns[i]
    col_data = df.iloc[:, i]
    n_values = col_data.notna().sum()
    print(f"  [{i}] '{col}' — {n_values} non-null values")
    if n_values > 0:
        print(f"       Sample values: {col_data.dropna().head(5).tolist()}")

# ============================================================
# STEP 4: Save cleaned CSV (optional, doesn't overwrite Excel)
# ============================================================
print()
print("=" * 70)
print("STEP 4: SAVING CLEANED CSV")
print("=" * 70)

# Save to data/processed/ — make sure the folder exists
import os
os.makedirs('data/processed', exist_ok=True)
df.to_csv('data/processed/raw_cleaned.csv', index=False)
print("Saved: data/processed/raw_cleaned.csv")
print(f"File size: {os.path.getsize('data/processed/raw_cleaned.csv') / 1024:.1f} KB")


