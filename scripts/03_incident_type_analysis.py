#!/usr/bin/env python3
"""
Script 03 — Incident Type Frequency Analysis (Table 1)
=======================================================
GeoBlock-DRS Research Pipeline | SIET 2026

Produces Table 1 from the paper:
  "Top Incident Types in FEMA Major Disaster Declarations (2000–2025)"

Computes declaration counts, percentage of total, cumulative percentage,
and mean / median response latency per incident type.

Outputs
-------
  outputs/tables/table1_incident_types.csv
  outputs/tables/table1_incident_types.txt  (formatted)
  outputs/figures/fig_incident_frequency.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# ── Configuration ────────────────────────────────────────────────────
CLEAN_FILE = "data/clean_declarations.csv"
FIG_DIR    = "outputs/figures"
TBL_DIR    = "outputs/tables"

PALETTE = {
    "Severe Storm(s)": "#185FA5",
    "Flood":           "#1D9E75",
    "Hurricane":       "#BA7517",
    "Tornado":         "#534AB7",
    "Fire/Wildfire":   "#D85A30",
    "Other":           "#888780",
}

CRITICAL_DAYS = 3.0   # 72-hour survivability threshold in days


def log(msg: str) -> None:
    print(f"  {msg}", flush=True)


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TBL_DIR, exist_ok=True)
    print("\n=== Script 03 — Incident Type Frequency Analysis ===\n")

    df    = pd.read_csv(CLEAN_FILE, low_memory=False)
    total = len(df)
    log(f"Loaded {total:,} records")

    it_col = "incidentType_clean" if "incidentType_clean" in df.columns else "incidentType"

    # ── Aggregate by incident type ────────────────────────────────────
    grp = (
        df.groupby(it_col)
        .agg(
            Declarations   = ("disasterNumber", "count"),
            mean_latency   = ("response_latency_days", "mean"),
            median_latency = ("response_latency_days", "median"),
        )
        .reset_index()
        .rename(columns={it_col: "Incident Type"})
        .sort_values("Declarations", ascending=False)
        .reset_index(drop=True)
    )

    grp["% of Total"]     = (grp["Declarations"] / total * 100).round(1)
    grp["Cumulative %"]   = grp["% of Total"].cumsum().round(1)
    grp["Mean Latency (d)"]   = grp["mean_latency"].round(1)
    grp["Median Latency (d)"] = grp["median_latency"].round(1)
    grp = grp.drop(columns=["mean_latency", "median_latency"])

    # Top 5 + Other row
    top5  = grp.head(5).copy()
    other = grp.iloc[5:]
    other_row = pd.DataFrame([{
        "Incident Type":      "Other (combined)",
        "Declarations":        other["Declarations"].sum(),
        "% of Total":          round(other["% of Total"].sum(), 1),
        "Cumulative %":        100.0,
        "Mean Latency (d)":    round(other["Mean Latency (d)"].mean(), 1),
        "Median Latency (d)":  round(other["Median Latency (d)"].mean(), 1),
    }])
    total_row = pd.DataFrame([{
        "Incident Type":      "TOTAL",
        "Declarations":        total,
        "% of Total":          100.0,
        "Cumulative %":        100.0,
        "Mean Latency (d)":    round(df["response_latency_days"].mean(), 1),
        "Median Latency (d)":  round(df["response_latency_days"].median(), 1),
    }])
    table = pd.concat([top5, other_row, total_row], ignore_index=True)
    print(table.to_string(index=False))

    # ── Save CSV and TXT ──────────────────────────────────────────────
    table.to_csv(f"{TBL_DIR}/table1_incident_types.csv", index=False)
    with open(f"{TBL_DIR}/table1_incident_types.txt", "w") as fh:
        fh.write("Table 1. Top Incident Types in FEMA Major Disaster Declarations "
                 "(2000–2025)\n")
        fh.write("-" * 88 + "\n")
        fh.write(table.to_string(index=False))
        fh.write("\n" + "-" * 88)
        fh.write("\nSource: Authors' analysis of OpenFEMA Disaster Declarations "
                 "Summaries v2.\n")
        fh.write("Latency = declarationDate − incidentBeginDate (capped at 90 days).\n")
    log(f"Saved table → {TBL_DIR}/table1_incident_types.csv")

    # ── Figure ────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Table 1 — FEMA Major Disaster Declarations by Incident Type (2000–2025)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    types   = top5["Incident Type"].tolist()
    counts  = top5["Declarations"].tolist()
    lats    = top5["Mean Latency (d)"].tolist()
    colours = [PALETTE.get(t, "#888") for t in types]

    # Panel A — declaration counts
    bars_a = ax1.barh(
        types[::-1], counts[::-1], color=colours[::-1],
        edgecolor="white", linewidth=0.5, height=0.62,
    )
    for bar, val in zip(bars_a, counts[::-1]):
        ax1.text(bar.get_width() + total * 0.002, bar.get_y() + bar.get_height() / 2,
                 f"{val:,}", va="center", ha="left", fontsize=9, color="#333")
    ax1.set_xlabel("Number of Declarations", fontsize=10)
    ax1.set_title("A — Declaration Frequency (Top 5)", fontsize=11, fontweight="bold")
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.tick_params(axis="y", labelsize=9)
    ax1.set_xlim(0, max(counts) * 1.18)
    ax1.grid(axis="x", linestyle="--", alpha=0.35)

    # Panel B — mean latency
    bars_b = ax2.barh(
        types[::-1], lats[::-1],
        color=[c + "bb" for c in colours[::-1]],
        edgecolor="white", linewidth=0.5, height=0.62,
    )
    for bar, val in zip(bars_b, lats[::-1]):
        ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f} d", va="center", ha="left", fontsize=9, color="#333")
    ax2.axvline(
        CRITICAL_DAYS, color="#D85A30", linestyle="--", linewidth=1.4,
        label=f"72-hr critical window ({CRITICAL_DAYS} days)",
    )
    ax2.set_xlabel("Mean Response Latency (days)", fontsize=10)
    ax2.set_title("B — Mean Declaration Latency", fontsize=11, fontweight="bold")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.tick_params(axis="y", labelsize=9)
    ax2.set_xlim(0, max(lats) * 1.22)
    ax2.legend(fontsize=8.5, loc="lower right")
    ax2.grid(axis="x", linestyle="--", alpha=0.35)

    plt.tight_layout()
    out_path = f"{FIG_DIR}/fig_incident_frequency.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    log(f"Saved figure → {out_path}")
    log(f"\n✓ Table 1 complete. Total records: {total:,}.")


if __name__ == "__main__":
    main()
