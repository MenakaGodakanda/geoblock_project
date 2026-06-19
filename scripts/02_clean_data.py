#!/usr/bin/env python3
"""
Script 02 — Data Cleaning and Preprocessing
============================================
GeoBlock-DRS Research Pipeline | SIET 2026

Cleans the raw OpenFEMA dataset and prepares it for analysis:
  1. Parses all date fields to datetime
  2. Computes response_latency_days (declarationDate − incidentBeginDate)
  3. Filters to major disaster declarations (type DR), 2000–2025
  4. Removes records outside the 0–90 day latency range (<0.1% of data)
  5. Standardises incident type labels (merges "Fire" and "Wildfire")
  6. Derives five-year cohort labels (final cohort: 2020–2025)
  7. Creates Boolean assistance flags (ia_flag, pa_flag, hm_flag)
  8. Saves clean CSV and a data quality report

Inputs
------
  data/raw_disaster_declarations.csv

Outputs
-------
  data/clean_declarations.csv
  outputs/reports/data_quality_report.txt
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

# ── Configuration ────────────────────────────────────────────────────
RAW_FILE    = "data/raw_disaster_declarations.csv"
CLEAN_FILE  = "data/clean_declarations.csv"
REPORT_FILE = "outputs/reports/data_quality_report.txt"

START_YEAR  = 2000
END_YEAR    = 2025        # inclusive — 2020–2025 forms the final six-year cohort
MAX_LATENCY = 90          # days; records above this are anomalous historical entries

DATE_COLS   = ["declarationDate", "incidentBeginDate", "incidentEndDate", "closeoutDate"]

# Normalise incident type labels to consistent canonical names
INCIDENT_TYPE_MAP = {
    "Severe Storm(s)": "Severe Storm(s)",
    "Severe Storm":    "Severe Storm(s)",
    "Flood":           "Flood",
    "Hurricane":       "Hurricane",
    "Tornado":         "Tornado",
    "Fire":            "Fire/Wildfire",
    "Wildfire":        "Fire/Wildfire",
    "Snow":            "Snow",
    "Typhoon":         "Typhoon",
    "Earthquake":      "Earthquake",
    "Biological":      "Biological",
    "Dam/Levee Break": "Dam/Levee Break",
    "Mud/Landslide":   "Mud/Landslide",
    "Drought":         "Drought",
}


def cohort_label(year: int) -> str:
    """
    Return the five-year cohort label for a given declaration year.
    Years 2020–2025 are grouped into a single six-year final cohort to
    include the most recent complete calendar year (2025) while preserving
    a clean analytical boundary. The asymmetric period is noted in the
    data quality report.
    """
    if year >= 2020:
        return "2020–2025"
    base = (year - 2000) // 5 * 5 + 2000
    return f"{base}–{base + 4}"


def log(msg: str) -> None:
    print(f"  {msg}", flush=True)


def main() -> None:
    os.makedirs("outputs/reports", exist_ok=True)
    print("\n=== Script 02 — Data Cleaning and Preprocessing ===\n")

    # ── 1. Load ───────────────────────────────────────────────────────
    log(f"Loading: {RAW_FILE}")
    df       = pd.read_csv(RAW_FILE, low_memory=False)
    raw_rows = len(df)
    log(f"Raw shape: {raw_rows:,} rows × {len(df.columns)} columns")

    report = [
        "=" * 64,
        "DATA QUALITY REPORT — GeoBlock-DRS Pipeline",
        f"Generated: {datetime.now().isoformat()}",
        "=" * 64,
        f"\n[1] RAW DATA",
        f"    Rows    : {raw_rows:,}",
        f"    Columns : {len(df.columns)}",
    ]

    # ── 2. Strip whitespace from column names ─────────────────────────
    df.columns = [c.strip() for c in df.columns]

    # ── 3. Parse date columns ─────────────────────────────────────────
    log("Parsing date columns ...")
    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
            df[col] = df[col].dt.tz_localize(None)   # strip timezone for arithmetic

    date_nulls = {c: int(df[c].isna().sum()) for c in DATE_COLS if c in df.columns}
    report.append("\n[2] DATE PARSING — null counts after parse:")
    for k, v in date_nulls.items():
        report.append(f"    {k:30s}: {v:,} nulls")

    # ── 4. Compute response latency ───────────────────────────────────
    log("Computing response latency ...")
    df["response_latency_days"] = (df["declarationDate"] - df["incidentBeginDate"]).dt.days

    before = len(df)
    df     = df[df["response_latency_days"].between(0, MAX_LATENCY)]
    report.append(
        f"\n[3] LATENCY — rows removed (outside 0–{MAX_LATENCY} days): {before - len(df):,}"
    )
    report.append(f"    Remaining rows: {len(df):,}")

    # ── 5. Extract declaration year ───────────────────────────────────
    df["declaration_year"] = df["declarationDate"].dt.year

    # ── 6. Filter to DR type and analysis window ─────────────────────
    log(f"Filtering to DR declarations, {START_YEAR}–{END_YEAR} ...")
    if "declarationType" in df.columns:
        df = df[df["declarationType"] == "DR"]
    df = df[df["declaration_year"].between(START_YEAR, END_YEAR)]
    log(f"After filter: {len(df):,} rows")
    report.append(
        f"\n[4] FILTER"
        f"\n    Declaration type  : DR (major disasters only)"
        f"\n    Year range        : {START_YEAR}–{END_YEAR}"
        f"\n    Rows after filter : {len(df):,}"
        f"\n    Note: 2025 = most recent complete calendar year available in"
        f"\n          the OpenFEMA API at time of analysis (June 2026)."
        f"\n          Records with declarationDate in Jan 2026 (arising from"
        f"\n          Dec 2025 incidents) are correctly excluded."
    )

    # ── 7. Standardise incident type ──────────────────────────────────
    it_col = "incidentType" if "incidentType" in df.columns else None
    if it_col:
        df["incidentType_clean"] = (
            df[it_col].str.strip()
            .map(INCIDENT_TYPE_MAP)
            .fillna(df[it_col].str.strip())
        )

    # ── 8. Cohort and decade labels ───────────────────────────────────
    df["cohort_5yr"] = df["declaration_year"].apply(cohort_label)
    df["decade"]     = (df["declaration_year"] // 10 * 10).astype(str) + "s"

    # ── 9. Boolean assistance flags ───────────────────────────────────
    # Map multiple possible column names from different API versions
    flag_candidates = {
        "ia_flag": ["ihProgramDeclared", "iaProgramDeclared", "individualAssistance"],
        "pa_flag": ["paProgramDeclared", "publicAssistance"],
        "hm_flag": ["hmProgramDeclared", "hazardMitigation"],
    }
    for flag, candidates in flag_candidates.items():
        df[flag] = False
        for c in candidates:
            if c in df.columns:
                df[flag] = df[c].astype(str).str.upper().isin(["1", "TRUE", "YES"])
                break

    # ── 10. Select output columns (deduplicated) ──────────────────────
    desired = [
        "disasterNumber", "state", "designatedArea",
        "incidentType", "incidentType_clean",
        "declarationDate", "incidentBeginDate", "incidentEndDate",
        "declaration_year", "cohort_5yr", "decade",
        "response_latency_days",
        "ia_flag", "pa_flag", "hm_flag",
        "fipsStateCode", "fipsCountyCode", "placeCode",
    ]
    keep = list(dict.fromkeys(c for c in desired if c in df.columns))
    df_clean = df[keep].copy()

    # ── 11. Missing value summary ─────────────────────────────────────
    report.append("\n[5] MISSING VALUES (final clean dataset):")
    for col in df_clean.columns:
        n = int(df_clean[col].isna().sum())
        if n > 0:
            report.append(f"    {col:35s}: {n:>6,}  ({n / len(df_clean) * 100:.1f}%)")

    report.append(
        f"\n[6] FINAL CLEAN DATASET"
        f"\n    Rows    : {len(df_clean):,}"
        f"\n    Columns : {len(df_clean.columns)}"
        f"\n    Cohorts : {sorted(df_clean['cohort_5yr'].unique())}"
        f"\n    Note: The 2020–2025 cohort spans six years (2020–2024 = five"
        f"\n          complete years + 2025 = sixth year) to include the most"
        f"\n          recent complete calendar year in the analysis window."
    )

    # ── 12. Save ──────────────────────────────────────────────────────
    df_clean.to_csv(CLEAN_FILE, index=False)
    log(f"Saved clean CSV → {CLEAN_FILE}")

    with open(REPORT_FILE, "w") as fh:
        fh.write("\n".join(report))
    log(f"Saved quality report → {REPORT_FILE}")

    # ── 13. Console summary ───────────────────────────────────────────
    print("\n── Column dtypes ──")
    print(df_clean.dtypes.to_string())
    print(f"\n── Latency summary (days) ──")
    print(df_clean["response_latency_days"].describe().round(2).to_string())
    print(f"\n── Top 5 incident types ──")
    it = "incidentType_clean" if "incidentType_clean" in df_clean.columns else "incidentType"
    print(df_clean[it].value_counts().head(5).to_string())
    print(f"\n── Cohort distribution ──")
    print(df_clean["cohort_5yr"].value_counts().sort_index().to_string())
    print(f"\n✓ Cleaning complete. {len(df_clean):,} records ready for analysis.")


if __name__ == "__main__":
    main()
