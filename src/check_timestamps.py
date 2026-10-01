"""
check_timestamps.py

Find exactly which Time values fail to parse as datetimes,
and what they look like in their raw form.
"""

import pandas as pd
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_FILE = os.path.join(PROJECT_ROOT, 'data', 'raw', 'Raw data .xlsx')

# Read raw, keep everything as strings — don't let pandas infer types
df = pd.read_excel(RAW_FILE, header=1, dtype=str)
df = df.iloc[1:].reset_index(drop=True)

time_col = df.columns[1]
raw_times = df[time_col]

# Try parsing each value individually and record failures
results = []
for idx, val in raw_times.items():
    try:
        parsed = pd.to_datetime(val)
        results.append((idx, val, 'OK'))
    except Exception as e:
        results.append((idx, val, f'FAIL: {type(e).__name__}'))

results_df = pd.DataFrame(results, columns=['row', 'raw_value', 'status'])

# Summary
failures = results_df[results_df['status'].str.startswith('FAIL')]
print(f"Total rows: {len(results_df)}")
print(f"Failures:   {len(failures)}")
print()

if len(failures) > 0:
    print("=" * 70)
    print("FAILED ROWS — full raw value (showing first 30)")
    print("=" * 70)
    for _, row in failures.head(30).iterrows():
        # repr() shows hidden characters like \t, \xa0
        print(f"  Row {row['row']:3d}: {repr(row['raw_value'])}")
    
    print()
    print("=" * 70)
    print("CHARACTER ANALYSIS of first 5 failed values")
    print("=" * 70)
    for _, row in failures.head(5).iterrows():
        val = row['raw_value']
        print(f"\nRow {row['row']}:")
        print(f"  String:    {repr(val)}")
        print(f"  Length:    {len(val) if isinstance(val, str) else 'N/A'}")
        if isinstance(val, str):
            print(f"  Codepoints: {[hex(ord(c)) for c in val]}")
else:
    print("No failures — all timestamps parse individually.")
    print("But pandas batch conversion still failed? That's unusual.")
    print("Re-running batch conversion to confirm...")
    batch = pd.to_datetime(raw_times, errors='coerce')
    print(f"Batch failures: {batch.isnull().sum()}")