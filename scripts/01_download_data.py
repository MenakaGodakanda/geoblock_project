#!/usr/bin/env python3
"""
Script 01 — Download OpenFEMA Disaster Declarations Summaries v2
================================================================
GeoBlock-DRS Research Pipeline | SIET 2026

Downloads the OpenFEMA Disaster Declarations Summaries dataset (v2)
from the official FEMA OpenData API. The dataset is freely available
with no authentication required.

API endpoint:
  https://www.fema.gov/api/open/v2/disasterDeclarationsSummaries

Output:
  data/raw_disaster_declarations.csv   (~68,000+ records)
  data/download_metadata.txt
"""

import requests
import pandas as pd
import os
import sys
import time
import json
from datetime import datetime

# ── Configuration ──────────────────────────────────────────────────
BASE_URL   = "https://www.fema.gov/api/open/v2/disasterDeclarationsSummaries"
PAGE_SIZE  = 1000          # FEMA API max per page
OUT_DIR    = "data"
OUT_FILE   = os.path.join(OUT_DIR, "raw_disaster_declarations.csv")
META_FILE  = os.path.join(OUT_DIR, "download_metadata.txt")
MAX_PAGES  = 200           # safety cap (~200,000 records max)
PAUSE_SEC  = 0.4           # polite pause between API calls

# ── Helpers ─────────────────────────────────────────────────────────
def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def fetch_page(skip: int, top: int) -> dict:
    """Fetch one page from the FEMA OpenData API."""
    params = {
        "$top":    top,
        "$skip":   skip,
        "$format": "json",
        "$orderby": "declarationDate asc",
    }
    resp = requests.get(BASE_URL, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()

# ── Main ─────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    log("Starting OpenFEMA Disaster Declarations download ...")

    # --- probe total record count --------------------------------
    probe = fetch_page(skip=0, top=1)
    total_count = probe.get("metadata", {}).get("count", None)
    if total_count:
        log(f"Total records reported by API: {total_count:,}")
    else:
        log("Could not read total count; will paginate until empty.")

    # --- paginate ------------------------------------------------
    all_records = []
    skip = 0
    page_num = 0

    while page_num < MAX_PAGES:
        page_num += 1
        log(f"  Fetching page {page_num:>3}  (skip={skip:>6}) ...")

        try:
            data = fetch_page(skip=skip, top=PAGE_SIZE)
        except requests.RequestException as exc:
            log(f"  ERROR on page {page_num}: {exc}. Retrying in 5s ...")
            time.sleep(5)
            data = fetch_page(skip=skip, top=PAGE_SIZE)

        records = data.get("DisasterDeclarationsSummaries", [])
        if not records:
            log("  No more records. Download complete.")
            break

        all_records.extend(records)
        log(f"  Collected {len(all_records):,} records so far ...")

        if len(records) < PAGE_SIZE:
            log("  Last page reached.")
            break

        skip += PAGE_SIZE
        time.sleep(PAUSE_SEC)

    # --- build DataFrame ----------------------------------------
    log(f"\nBuilding DataFrame from {len(all_records):,} records ...")
    df = pd.DataFrame(all_records)
    log(f"DataFrame shape: {df.shape}")
    log(f"Columns: {list(df.columns)}")

    # --- save ----------------------------------------------------
    df.to_csv(OUT_FILE, index=False)
    log(f"Saved raw CSV → {OUT_FILE}")

    # --- metadata -----------------------------------------------
    meta_lines = [
        f"Download timestamp : {datetime.now().isoformat()}",
        f"Source URL         : {BASE_URL}",
        f"Total records      : {len(all_records):,}",
        f"Total columns      : {len(df.columns)}",
        f"Output file        : {OUT_FILE}",
        f"Columns            : {', '.join(df.columns.tolist())}",
    ]
    with open(META_FILE, "w") as f:
        f.write("\n".join(meta_lines))
    log(f"Metadata saved  → {META_FILE}")
    log("\n✓ Download complete.")

if __name__ == "__main__":
    main()
