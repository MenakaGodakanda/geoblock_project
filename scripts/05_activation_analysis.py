#!/usr/bin/env python3
"""
Script 05 — Assistance Activation Rate Analysis (Table 2 + Figure 4B)
======================================================================
GeoBlock-DRS Research Pipeline | SIET 2026

Produces Table 2 from the paper:
  "Assistance Programme Activation Rates by Incident Type (2000–2024)"

Computes IA, PA, and HM activation rates per incident type and overall,
and exposes the systematic IA–PA gap documented in the paper.

Outputs:
  outputs/tables/table2_activation_rates.csv
  outputs/tables/table2_activation_rates.txt
  outputs/figures/fig_activation_gap.png
  outputs/figures/fig_ia_pa_gap_trend.png
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

TOP_TYPES = ["Hurricane", "Flood", "Severe Storm(s)", "Tornado", "Fire/Wildfire"]

COLORS = {
    "ia":  "#185FA5",
    "pa":  "#EF9F27",
    "hm":  "#1D9E75",
    "gap": "#D85A30",
}

# ── Helpers ──────────────────────────────────────────────────────────
def log(msg): print(f"  {msg}", flush=True)

def activation_rate(series):
    """Compute activation rate as percentage of True values."""
    return series.sum() / len(series) * 100

# ── Main ─────────────────────────────────────────────────────────────
def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TBL_DIR, exist_ok=True)

    print("\n=== SCRIPT 05 — Assistance Activation Rate Analysis ===\n")

    df = pd.read_csv(CLEAN_FILE, low_memory=False)
    log(f"Loaded {len(df):,} records")

    it_col = "incidentType_clean" if "incidentType_clean" in df.columns else "incidentType"

    # ── Compute rates per incident type ───────────────────────────────
    rows = []
    for t in TOP_TYPES:
        sub = df[df[it_col] == t]
        if len(sub) == 0:
            continue
        row = {
            "Incident Type":    t,
            "n":                len(sub),
            "IA Rate (%)":      round(activation_rate(sub["ia_flag"]), 1),
            "PA Rate (%)":      round(activation_rate(sub["pa_flag"]), 1),
            "HM Rate (%)":      round(activation_rate(sub["hm_flag"]), 1),
        }
        row["IA–PA Gap (pp)"] = round(row["PA Rate (%)"] - row["IA Rate (%)"], 1)
        rows.append(row)

    # All major declarations row
    all_row = {
        "Incident Type":    "All Major Declarations",
        "n":                len(df),
        "IA Rate (%)":      round(activation_rate(df["ia_flag"]), 1),
        "PA Rate (%)":      round(activation_rate(df["pa_flag"]), 1),
        "HM Rate (%)":      round(activation_rate(df["hm_flag"]), 1),
    }
    all_row["IA–PA Gap (pp)"] = round(all_row["PA Rate (%)"] - all_row["IA Rate (%)"], 1)
    rows.append(all_row)

    table = pd.DataFrame(rows)

    print(table.to_string(index=False))

    # ── Save ─────────────────────────────────────────────────────────
    table.to_csv(f"{TBL_DIR}/table2_activation_rates.csv", index=False)
    with open(f"{TBL_DIR}/table2_activation_rates.txt", "w") as f:
        f.write("Table 2. Assistance Programme Activation Rates by Incident Type (Major Disasters 2000–2024)\n")
        f.write("-" * 80 + "\n")
        f.write(table.to_string(index=False))
        f.write("\n" + "-" * 80)
        f.write("\nIA = Individual Assistance; PA = Public Assistance; HM = Hazard Mitigation.\n")
        f.write("pp = percentage points gap between PA and IA rates.\n")
        f.write("Source: Authors' analysis of OpenFEMA Disaster Declarations Summaries v2.\n")
    log(f"Saved Table 2 → {TBL_DIR}/table2_activation_rates.csv")

    # ── Figure A: Grouped bar chart ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    types_plot = table[table["Incident Type"] != "All Major Declarations"]["Incident Type"].tolist()
    n = len(types_plot)
    x = np.arange(n)
    w = 0.26

    ia_vals = table[table["Incident Type"].isin(types_plot)]["IA Rate (%)"].values
    pa_vals = table[table["Incident Type"].isin(types_plot)]["PA Rate (%)"].values
    hm_vals = table[table["Incident Type"].isin(types_plot)]["HM Rate (%)"].values

    b1 = ax.bar(x - w, ia_vals, width=w, color=COLORS["ia"], label="Individual Assistance (IA)",
                edgecolor="white", linewidth=0.5)
    b2 = ax.bar(x,     pa_vals, width=w, color=COLORS["pa"], label="Public Assistance (PA)",
                edgecolor="white", linewidth=0.5)
    b3 = ax.bar(x + w, hm_vals, width=w, color=COLORS["hm"], label="Hazard Mitigation (HM)",
                edgecolor="white", linewidth=0.5)

    # Value labels
    for bar, val in zip(list(b1) + list(b2) + list(b3),
                        list(ia_vals) + list(pa_vals) + list(hm_vals)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=8, color="#333")

    # Annotate IA–PA gap for Flood (most illustrative)
    flood_idx = types_plot.index("Flood") if "Flood" in types_plot else 1
    flood_ia  = ia_vals[flood_idx]
    flood_pa  = pa_vals[flood_idx]
    ax.annotate("", xy=(x[flood_idx], flood_pa - 1),
                xytext=(x[flood_idx] - w, flood_ia + 1),
                arrowprops=dict(arrowstyle="<->", color=COLORS["gap"], lw=1.5))
    ax.text(x[flood_idx] - w*1.5, (flood_ia + flood_pa)/2,
            f"Gap\n{flood_pa - flood_ia:.0f}pp", ha="center", va="center",
            fontsize=8, color=COLORS["gap"], fontweight="bold")

    # Mean reference lines
    ax.axhline(all_row["IA Rate (%)"], color=COLORS["ia"], linestyle="--",
               linewidth=1.2, alpha=0.6, label=f'Mean IA = {all_row["IA Rate (%)"]:.1f}%')
    ax.axhline(all_row["PA Rate (%)"], color=COLORS["pa"], linestyle="--",
               linewidth=1.2, alpha=0.6, label=f'Mean PA = {all_row["PA Rate (%)"]:.1f}%')

    ax.set_xticks(x)
    ax.set_xticklabels(types_plot, fontsize=10)
    ax.set_ylabel("Activation Rate (%)", fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_title("Figure 4B — Assistance Programme Activation Rates by Incident Type (2000–2024)\n"
                 "Source: OpenFEMA Disaster Declarations Summaries v2",
                 fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)

    plt.tight_layout()
    out_path = f"{FIG_DIR}/fig_activation_gap.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    log(f"Saved figure → {out_path}")

    # ── Figure B: IA-PA gap trend over cohorts ────────────────────────
    if "cohort_5yr" in df.columns:
        cohort_order = ["2000–2004", "2005–2009", "2010–2014", "2015–2019", "2020–2024"]
        cgrp = df.groupby("cohort_5yr").agg(
            ia_rate = ("ia_flag", lambda x: x.sum() / len(x) * 100),
            pa_rate = ("pa_flag", lambda x: x.sum() / len(x) * 100),
            hm_rate = ("hm_flag", lambda x: x.sum() / len(x) * 100),
        ).reset_index()
        cgrp["_sort"] = cgrp["cohort_5yr"].map({c: i for i, c in enumerate(cohort_order)})
        cgrp = cgrp.sort_values("_sort").drop("_sort", axis=1)

        fig2, ax3 = plt.subplots(figsize=(10, 5))
        c_x = np.arange(len(cgrp))
        ax3.plot(c_x, cgrp["pa_rate"].values, "o-", color=COLORS["pa"],
                 linewidth=2.5, markersize=7, label="PA activation rate (%)",
                 markerfacecolor="white", markeredgewidth=2)
        ax3.plot(c_x, cgrp["ia_rate"].values, "s--", color=COLORS["ia"],
                 linewidth=2.5, markersize=7, label="IA activation rate (%)",
                 markerfacecolor="white", markeredgewidth=2)
        ax3.plot(c_x, cgrp["hm_rate"].values, "^:", color=COLORS["hm"],
                 linewidth=1.8, markersize=6, label="HM activation rate (%)",
                 markerfacecolor="white", markeredgewidth=1.5)

        # Fill gap
        ax3.fill_between(c_x, cgrp["ia_rate"].values, cgrp["pa_rate"].values,
                         alpha=0.10, color=COLORS["gap"], label="IA–PA gap (pp)")

        for i, row in cgrp.iterrows():
            idx = list(cgrp.index).index(i)
            ax3.annotate(f"{row['pa_rate']:.0f}%", (idx, row["pa_rate"]),
                         textcoords="offset points", xytext=(0, 7),
                         ha="center", fontsize=8, color=COLORS["pa"])
            ax3.annotate(f"{row['ia_rate']:.0f}%", (idx, row["ia_rate"]),
                         textcoords="offset points", xytext=(0, -14),
                         ha="center", fontsize=8, color=COLORS["ia"])

        ax3.set_xticks(c_x)
        ax3.set_xticklabels(cgrp["cohort_5yr"].values, fontsize=10)
        ax3.set_ylabel("Activation Rate (%)", fontsize=11)
        ax3.set_title("IA vs PA Activation Rate Trend by Cohort (2000–2024)\n"
                      "Shaded region = IA–PA gap",
                      fontsize=12, fontweight="bold")
        ax3.set_ylim(0, 115)
        ax3.spines[["top", "right"]].set_visible(False)
        ax3.grid(axis="y", linestyle="--", alpha=0.35)
        ax3.legend(fontsize=9, loc="lower right")

        plt.tight_layout()
        out_path2 = f"{FIG_DIR}/fig_ia_pa_gap_trend.png"
        plt.savefig(out_path2, dpi=180, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        plt.close()
        log(f"Saved figure → {out_path2}")

    print(f"\n✓ Activation analysis complete.")
    print(f"  Overall IA rate: {all_row['IA Rate (%)']}%")
    print(f"  Overall PA rate: {all_row['PA Rate (%)']}%")
    print(f"  IA–PA gap      : {all_row['IA–PA Gap (pp)']} percentage points")

if __name__ == "__main__":
    main()
