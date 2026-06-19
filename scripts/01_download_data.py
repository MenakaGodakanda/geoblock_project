#!/usr/bin/env python3
"""
Script 01 — Download Real OpenFEMA Dataset
==========================================
GeoBlock-DRS Research Pipeline | SIET 2026

Downloads the OpenFEMA Disaster Declarations Summaries v2 dataset
from the official FEMA OpenData API. Free to access — no API key,
no registration required.

API endpoint:
  https://www.fema.gov/api/open/v2/disasterDeclarationsSummaries

The script paginates automatically through all records (typically
68,000–75,000 rows) at 1,000 records per page with a polite delay
between requests.

Outputs
-------
  data/raw_disaster_declarations.csv   (all records, all years)
  data/download_metadata.txt           (timestamp, record count, columns)

Usage
-----
  python3 scripts/01_download_data.py

Note: Run Script 00 instead if this machine cannot reach the FEMA API.
"""

import requests
import pandas as pd
import os
import time
import json
from datetime import datetime

# ── Configuration ────────────────────────────────────────────────────
BASE_URL   = "https://www.fema.gov/api/open/v2/disasterDeclarationsSummaries"
PAGE_SIZE  = 1000       # FEMA API maximum records per page
OUT_DIR    = "data"
OUT_FILE   = os.path.join(OUT_DIR, "raw_disaster_declarations.csv")
META_FILE  = os.path.join(OUT_DIR, "download_metadata.txt")
MAX_PAGES  = 200        # safety cap (~200,000 records maximum)
PAUSE_SEC  = 0.4        # polite pause between API calls


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def fetch_page(skip: int, top: int) -> dict:
    """Fetch one page of records from the FEMA OpenData API."""
    params = {
        "$top":    top,
        "$skip":   skip,
        "$format": "json",
        "$orderby": "declarationDate asc",
    }
    response = requests.get(BASE_URL, params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    log("Starting OpenFEMA Disaster Declarations Summaries download ...")

    # Probe total record count
    probe       = fetch_page(skip=0, top=1)
    total_count = probe.get("metadata", {}).get("count", None)
    if total_count:
        log(f"Total records available: {total_count:,}")
    else:
        log("Could not read total count; will paginate until empty.")

    all_records = []
    skip        = 0
    page_num    = 0

    while page_num < MAX_PAGES:
        page_num += 1
        log(f"  Page {page_num:>3}  (skip={skip:>6}) ...")

        try:
            data = fetch_page(skip=skip, top=PAGE_SIZE)
        except requests.RequestException as exc:
            log(f"  ERROR on page {page_num}: {exc}. Retrying in 5 s ...")
            time.sleep(5)
            data = fetch_page(skip=skip, top=PAGE_SIZE)

        records = data.get("DisasterDeclarationsSummaries", [])
        if not records:
            log("  No more records. Download complete.")
            break

        all_records.extend(records)
        log(f"  Cumulative records: {len(all_records):,}")

        if len(records) < PAGE_SIZE:
            log("  Last page reached.")
            break

        skip += PAGE_SIZE
        time.sleep(PAUSE_SEC)

    # Build DataFrame and save
    log(f"\nBuilding DataFrame from {len(all_records):,} records ...")
    df = pd.DataFrame(all_records)
    df.to_csv(OUT_FILE, index=False)
    log(f"Saved → {OUT_FILE}  ({df.shape[0]:,} rows × {df.shape[1]} columns)")

    # Save metadata
    meta = [
        f"Download timestamp : {datetime.now().isoformat()}",
        f"Source             : {BASE_URL}",
        f"Total records      : {len(all_records):,}",
        f"Total columns      : {len(df.columns)}",
        f"Output file        : {OUT_FILE}",
        f"Columns            : {', '.join(df.columns.tolist())}",
    ]
    with open(META_FILE, "w") as fh:
        fh.write("\n".join(meta))
    log(f"Metadata saved → {META_FILE}")
    log("\n✓ Download complete.")


if __name__ == "__main__":
    main()
