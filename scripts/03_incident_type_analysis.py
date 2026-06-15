#!/usr/bin/env python3
"""
Script 03 — Incident Type Frequency Analysis (Table 1)
=======================================================
GeoBlock-DRS Research Pipeline | SIET 2026

Produces Table 1 from the paper:
  "Top Incident Types in FEMA Major Disaster Declarations (2000–2024)"

Computes:
  - Declaration counts per incident type
  - Percentage of total
  - Mean response latency per type
  - Cumulative percentage

Outputs:
  outputs/tables/table1_incident_types.csv
  outputs/tables/table1_incident_types.txt   (formatted)
  outputs/figures/fig_incident_frequency.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ── Configuration ────────────────────────────────────────────────────
CLEAN_FILE = "data/clean_declarations.csv"
FIG_DIR    = "outputs/figures"
TBL_DIR    = "outputs/tables"

COLORS = {
    "Severe Storm(s)": "#185FA5",
    "Flood":           "#1D9E75",
    "Hurricane":       "#BA7517",
    "Tornado":         "#534AB7",
    "Fire/Wildfire":   "#D85A30",
    "Other":           "#888780",
}

# ── Helpers ──────────────────────────────────────────────────────────
def log(msg): print(f"  {msg}", flush=True)

# ── Main ─────────────────────────────────────────────────────────────
def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TBL_DIR, exist_ok=True)

    print("\n=== SCRIPT 03 — Incident Type Frequency Analysis ===\n")

    df = pd.read_csv(CLEAN_FILE, low_memory=False)
    df["declarationDate"] = pd.to_datetime(df["declarationDate"], errors="coerce")
    total_records = len(df)
    log(f"Loaded {total_records:,} clean records")

    # ── Group by incident type ────────────────────────────────────────
    it_col = "incidentType_clean" if "incidentType_clean" in df.columns else "incidentType"
    grp = df.groupby(it_col).agg(
        declarations   = ("disasterNumber",       "count"),
        mean_latency   = ("response_latency_days", "mean"),
        median_latency = ("response_latency_days", "median"),
    ).reset_index()
    grp.columns = ["Incident Type", "Declarations", "Mean Latency (days)", "Median Latency (days)"]
    grp = grp.sort_values("Declarations", ascending=False).reset_index(drop=True)

    # ── Compute derived columns ───────────────────────────────────────
    grp["% of Total"]    = (grp["Declarations"] / total_records * 100).round(1)
    grp["Cumulative %"]  = grp["% of Total"].cumsum().round(1)
    grp["Mean Latency (days)"]   = grp["Mean Latency (days)"].round(1)
    grp["Median Latency (days)"] = grp["Median Latency (days)"].round(1)

    # ── Top-5 + Other row ─────────────────────────────────────────────
    top5 = grp.head(5).copy()
    other_rows = grp.iloc[5:]
    other_row = pd.DataFrame([{
        "Incident Type":          "Other (combined)",
        "Declarations":           other_rows["Declarations"].sum(),
        "Mean Latency (days)":    round(other_rows["Mean Latency (days)"].mean(), 1),
        "Median Latency (days)":  round(other_rows["Median Latency (days)"].mean(), 1),
        "% of Total":             round(other_rows["% of Total"].sum(), 1),
        "Cumulative %":           100.0,
    }])
    table = pd.concat([top5, other_row], ignore_index=True)

    # Total row
    total_row = pd.DataFrame([{
        "Incident Type":          "TOTAL",
        "Declarations":           total_records,
        "Mean Latency (days)":    round(df["response_latency_days"].mean(), 1),
        "Median Latency (days)":  round(df["response_latency_days"].median(), 1),
        "% of Total":             100.0,
        "Cumulative %":           100.0,
    }])
    table_full = pd.concat([table, total_row], ignore_index=True)

    # ── Print to console ──────────────────────────────────────────────
    print(table_full.to_string(index=False))

    # ── Save CSV ──────────────────────────────────────────────────────
    table_full.to_csv(f"{TBL_DIR}/table1_incident_types.csv", index=False)
    log(f"Saved → {TBL_DIR}/table1_incident_types.csv")

    # ── Save formatted TXT ────────────────────────────────────────────
    with open(f"{TBL_DIR}/table1_incident_types.txt", "w") as f:
        f.write("Table 1. Top Incident Types in FEMA Major Disaster Declarations (2000–2024)\n")
        f.write("-" * 90 + "\n")
        f.write(table_full.to_string(index=False))
        f.write("\n" + "-" * 90)
        f.write("\nSource: Authors' analysis of OpenFEMA Disaster Declarations Summaries v2.\n")
        f.write("Latency = declarationDate − incidentBeginDate (capped at 90 days).\n")
    log(f"Saved → {TBL_DIR}/table1_incident_types.txt")

    # ── Figure: Horizontal bar chart ─────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Table 1 Visualisation — FEMA Major Disaster Declarations (2000–2024)",
                 fontsize=13, fontweight="bold", y=1.01)

    # Left: declaration counts
    ax1 = axes[0]
    types   = top5["Incident Type"].tolist()
    counts  = top5["Declarations"].tolist()
    bar_colors = [COLORS.get(t, "#888780") for t in types]
    bars = ax1.barh(types[::-1], counts[::-1], color=bar_colors[::-1],
                    edgecolor="white", linewidth=0.5, height=0.6)
    for bar, val in zip(bars, counts[::-1]):
        ax1.text(bar.get_width() + 150, bar.get_y() + bar.get_height()/2,
                 f"{val:,}", va="center", ha="left", fontsize=9, color="#333")
    ax1.set_xlabel("Number of Declarations", fontsize=10)
    ax1.set_title("A — Declaration Frequency (Top 5 Types)", fontsize=11, fontweight="bold")
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.tick_params(axis="y", labelsize=9)
    ax1.set_xlim(0, max(counts) * 1.18)

    # Right: mean latency comparison
    ax2 = axes[1]
    latencies = top5["Mean Latency (days)"].tolist()
    bars2 = ax2.barh(types[::-1], latencies[::-1],
                     color=[c + "bb" for c in bar_colors[::-1]],
                     edgecolor="white", linewidth=0.5, height=0.6)
    for bar, val in zip(bars2, latencies[::-1]):
        ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                 f"{val:.1f}d", va="center", ha="left", fontsize=9, color="#333")
    # 72-hour critical window line
    ax2.axvline(3, color="#D85A30", linestyle="--", linewidth=1.2, label="72-hr critical window")
    ax2.set_xlabel("Mean Response Latency (days)", fontsize=10)
    ax2.set_title("B — Mean Declaration Response Latency", fontsize=11, fontweight="bold")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.tick_params(axis="y", labelsize=9)
    ax2.set_xlim(0, max(latencies) * 1.22)
    ax2.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    out_path = f"{FIG_DIR}/fig_incident_frequency.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    log(f"Saved figure → {out_path}")

    print(f"\n✓ Table 1 analysis complete. {len(top5)} incident types analysed.")

if __name__ == "__main__":
    main()
