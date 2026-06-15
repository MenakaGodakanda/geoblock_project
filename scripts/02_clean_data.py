#!/usr/bin/env python3
"""
Script 02 — Data Cleaning and Preprocessing
============================================
GeoBlock-DRS Research Pipeline | SIET 2026

Cleans the raw OpenFEMA dataset:
  - Standardises column names and data types
  - Parses all date fields
  - Computes response_latency_days (declarationDate − incidentBeginDate)
  - Filters to major disaster declarations (DR type)
  - Restricts analysis window to 2000-2024
  - Removes anomalous latency records (> 90 days)
  - Derives year / cohort / decade columns
  - Outputs clean CSV and a data-quality report

Inputs:
  data/raw_disaster_declarations.csv

Outputs:
  data/clean_declarations.csv
  outputs/reports/data_quality_report.txt
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# ── Configuration ────────────────────────────────────────────────────
RAW_FILE    = "data/raw_disaster_declarations.csv"
CLEAN_FILE  = "data/clean_declarations.csv"
REPORT_FILE = "outputs/reports/data_quality_report.txt"

START_YEAR  = 2000
END_YEAR    = 2024
MAX_LATENCY = 90        # cap latency at 90 days to remove anomalies

DATE_COLS = [
    "declarationDate",
    "incidentBeginDate",
    "incidentEndDate",
    "closeoutDate",
]

INCIDENT_TYPE_MAP = {
    "Severe Storm(s)": "Severe Storm(s)",
    "Severe Storm":    "Severe Storm(s)",
    "Flood":           "Flood",
    "Hurricane":       "Hurricane",
    "Tornado":         "Tornado",
    "Fire":            "Fire/Wildfire",
    "Wildfire":        "Fire/Wildfire",
}

# ── Helpers ──────────────────────────────────────────────────────────
def log(msg):
    print(f"  {msg}", flush=True)

def cohort_label(year):
    """Return five-year cohort label for a given year."""
    base = (year - 2000) // 5 * 5 + 2000
    return f"{base}–{base+4}"

# ── Main ─────────────────────────────────────────────────────────────
def main():
    os.makedirs("outputs/reports", exist_ok=True)

    print("\n=== SCRIPT 02 — Data Cleaning & Preprocessing ===\n")

    # ── 1. Load ──────────────────────────────────────────────────────
    log(f"Loading: {RAW_FILE}")
    df = pd.read_csv(RAW_FILE, low_memory=False)
    raw_shape = df.shape
    log(f"Raw shape: {raw_shape[0]:,} rows × {raw_shape[1]} columns")

    report = []
    report.append("=" * 60)
    report.append("DATA QUALITY REPORT — GeoBlock-DRS Pipeline")
    report.append(f"Generated: {datetime.now().isoformat()}")
    report.append("=" * 60)
    report.append(f"\n[1] RAW DATA")
    report.append(f"    Rows    : {raw_shape[0]:,}")
    report.append(f"    Columns : {raw_shape[1]}")

    # ── 2. Normalise column names ─────────────────────────────────────
    df.columns = [c.strip() for c in df.columns]

    # ── 3. Parse dates ────────────────────────────────────────────────
    log("Parsing date columns ...")
    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
            df[col] = df[col].dt.tz_localize(None)  # strip tz for arithmetic
    date_nulls = {c: int(df[c].isna().sum()) for c in DATE_COLS if c in df.columns}
    report.append(f"\n[2] DATE PARSING — null counts after parse:")
    for k, v in date_nulls.items():
        report.append(f"    {k:25s} : {v:,} nulls")

    # ── 4. Compute response latency ───────────────────────────────────
    log("Computing response latency ...")
    df["response_latency_days"] = (
        df["declarationDate"] - df["incidentBeginDate"]
    ).dt.days
    # remove negative and extreme outliers
    before = len(df)
    df = df[df["response_latency_days"].between(0, MAX_LATENCY)]
    after = len(df)
    report.append(f"\n[3] LATENCY — rows removed (outside 0–{MAX_LATENCY} days): {before - after:,}")
    report.append(f"    Remaining rows: {after:,}")

    # ── 5. Extract year ───────────────────────────────────────────────
    df["declaration_year"] = df["declarationDate"].dt.year

    # ── 6. Filter to analysis window and DR type ──────────────────────
    log(f"Filtering to major disaster declarations (DR), {START_YEAR}–{END_YEAR} ...")
    if "declarationType" in df.columns:
        df = df[df["declarationType"] == "DR"]
    df = df[df["declaration_year"].between(START_YEAR, END_YEAR)]
    log(f"After filter: {len(df):,} rows")
    report.append(f"\n[4] FILTER")
    report.append(f"    Declaration type  : DR (major disasters only)")
    report.append(f"    Year range        : {START_YEAR}–{END_YEAR}")
    report.append(f"    Rows after filter : {len(df):,}")

    # ── 7. Standardise incident type ──────────────────────────────────
    if "incidentType" in df.columns:
        df["incidentType_clean"] = (
            df["incidentType"]
            .str.strip()
            .map(INCIDENT_TYPE_MAP)
            .fillna(df["incidentType"].str.strip())
        )

    # ── 8. Add cohort and decade columns ──────────────────────────────
    df["cohort_5yr"]  = df["declaration_year"].apply(cohort_label)
    df["decade"]      = (df["declaration_year"] // 10 * 10).astype(str) + "s"

    # ── 9. Boolean assistance flags ──────────────────────────────────
    assist_cols = {
        "ihProgramDeclared":  "ia_flag",
        "iaProgramDeclared":  "ia_flag",
        "paProgramDeclared":  "pa_flag",
        "hmProgramDeclared":  "hm_flag",
    }
    for src_col, new_col in assist_cols.items():
        if src_col in df.columns and new_col not in df.columns:
            df[new_col] = df[src_col].astype(str).str.upper().isin(["1", "TRUE", "YES"])
    # fallback if columns have different API names
    for flag, candidates in {
        "ia_flag": ["ihProgramDeclared", "iaProgramDeclared", "individualAssistance"],
        "pa_flag": ["paProgramDeclared",  "publicAssistance"],
        "hm_flag": ["hmProgramDeclared",  "hazardMitigation"],
    }.items():
        if flag not in df.columns:
            for c in candidates:
                if c in df.columns:
                    df[flag] = df[c].astype(str).str.upper().isin(["1", "TRUE", "YES"])
                    break
            if flag not in df.columns:
                df[flag] = False

    # ── 10. Missing value summary ─────────────────────────────────────
    report.append(f"\n[5] MISSING VALUES (after cleaning)")
    keep_cols_candidates = [
        "disasterNumber", "state", "designatedArea",
        "incidentType", "incidentType_clean",
        "declarationDate", "incidentBeginDate", "incidentEndDate",
        "declaration_year", "cohort_5yr", "decade",
        "response_latency_days",
        "ia_flag", "pa_flag", "hm_flag",
        "fipsStateCode", "fipsCountyCode",
        "placeCode",
    ]
    # deduplicate while preserving order
    seen = set()
    keep_cols = []
    for c in keep_cols_candidates:
        if c in df.columns and c not in seen:
            keep_cols.append(c)
            seen.add(c)
    df_clean = df[keep_cols].copy()

    for col in df_clean.columns:
        n_null = int(df_clean[col].isna().sum())
        pct    = n_null / len(df_clean) * 100
        if n_null > 0:
            report.append(f"    {col:35s}: {n_null:>6,}  ({pct:.1f}%)")
    report.append(f"\n[6] FINAL CLEAN DATASET")
    report.append(f"    Rows    : {len(df_clean):,}")
    report.append(f"    Columns : {len(df_clean.columns)}")
    report.append(f"    Columns : {', '.join(df_clean.columns)}")

    # ── 11. Save clean data ───────────────────────────────────────────
    df_clean.to_csv(CLEAN_FILE, index=False)
    log(f"Saved clean CSV → {CLEAN_FILE}")

    # ── 12. Save report ───────────────────────────────────────────────
    with open(REPORT_FILE, "w") as f:
        f.write("\n".join(report))
    log(f"Saved quality report → {REPORT_FILE}")

    # ── 13. Console preview ───────────────────────────────────────────
    print("\n── Column dtypes ──")
    print(df_clean.dtypes.to_string())
    print(f"\n── First 5 rows ──")
    print(df_clean.head().to_string())
    print(f"\n── Latency summary ──")
    print(df_clean["response_latency_days"].describe().to_string())
    print(f"\n── Top incident types ──")
    print(df_clean["incidentType_clean"].value_counts().head(10).to_string())

    print(f"\n✓ Cleaning complete. {len(df_clean):,} clean records ready for analysis.")

if __name__ == "__main__":
    main()
