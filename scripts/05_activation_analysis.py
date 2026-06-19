#!/usr/bin/env python3
"""
Script 05 — Assistance Activation Rate Analysis (Table 2)
==========================================================
GeoBlock-DRS Research Pipeline | SIET 2026

Produces Table 2 from the paper:
  "Assistance Programme Activation Rates by Incident Type (2000–2025)"

Computes IA, PA, and HM activation rates per incident type and overall,
and quantifies the IA–PA gap that motivates the ResourceAllocation
smart contract's equity weight correction.

Outputs
-------
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
CLEAN_FILE   = "data/clean_declarations.csv"
FIG_DIR      = "outputs/figures"
TBL_DIR      = "outputs/tables"
COHORT_ORDER = ["2000–2004", "2005–2009", "2010–2014", "2015–2019", "2020–2025"]
TOP_TYPES    = ["Hurricane", "Flood", "Severe Storm(s)", "Tornado", "Fire/Wildfire"]

COLOURS = {
    "ia":  "#185FA5",
    "pa":  "#EF9F27",
    "hm":  "#1D9E75",
    "gap": "#D85A30",
}


def log(msg: str) -> None:
    print(f"  {msg}", flush=True)


def activation_rate(series: pd.Series) -> float:
    """Return activation rate as a percentage."""
    return round(series.sum() / max(len(series), 1) * 100, 1)


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TBL_DIR, exist_ok=True)
    print("\n=== Script 05 — Assistance Activation Rate Analysis ===\n")

    df    = pd.read_csv(CLEAN_FILE, low_memory=False)
    it_col = "incidentType_clean" if "incidentType_clean" in df.columns else "incidentType"
    log(f"Loaded {len(df):,} records")

    # ── Table 2 — rates per incident type ────────────────────────────
    rows = []
    for t in TOP_TYPES:
        sub = df[df[it_col] == t]
        if len(sub) == 0:
            continue
        row = {
            "Incident Type": t,
            "n":             len(sub),
            "IA Rate (%)":   activation_rate(sub["ia_flag"]),
            "PA Rate (%)":   activation_rate(sub["pa_flag"]),
            "HM Rate (%)":   activation_rate(sub["hm_flag"]),
        }
        row["IA–PA Gap (pp)"] = round(row["PA Rate (%)"] - row["IA Rate (%)"], 1)
        rows.append(row)

    all_row = {
        "Incident Type":  "All Major Declarations",
        "n":              len(df),
        "IA Rate (%)":    activation_rate(df["ia_flag"]),
        "PA Rate (%)":    activation_rate(df["pa_flag"]),
        "HM Rate (%)":    activation_rate(df["hm_flag"]),
    }
    all_row["IA–PA Gap (pp)"] = round(all_row["PA Rate (%)"] - all_row["IA Rate (%)"], 1)
    rows.append(all_row)

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))

    # ── Save ─────────────────────────────────────────────────────────
    table.to_csv(f"{TBL_DIR}/table2_activation_rates.csv", index=False)
    with open(f"{TBL_DIR}/table2_activation_rates.txt", "w") as fh:
        fh.write(
            "Table 2. Assistance Programme Activation Rates by Incident Type "
            "(Major Disasters 2000–2025)\n" + "-" * 84 + "\n"
        )
        fh.write(table.to_string(index=False))
        fh.write(
            "\n" + "-" * 84
            + "\nIA = Individual Assistance; PA = Public Assistance; "
              "HM = Hazard Mitigation.\n"
            + "pp = percentage-point gap between PA and IA rates.\n"
            + "Source: Authors' analysis of OpenFEMA Disaster Declarations "
              "Summaries v2.\n"
        )
    log(f"Saved → {TBL_DIR}/table2_activation_rates.csv")

    # ── Figure A — grouped bar chart ──────────────────────────────────
    types_plot = [r["Incident Type"] for r in rows
                  if r["Incident Type"] != "All Major Declarations"]
    n  = len(types_plot)
    x  = np.arange(n)
    w  = 0.26

    ia_vals = [r["IA Rate (%)"]  for r in rows if r["Incident Type"] != "All Major Declarations"]
    pa_vals = [r["PA Rate (%)"]  for r in rows if r["Incident Type"] != "All Major Declarations"]
    hm_vals = [r["HM Rate (%)"]  for r in rows if r["Incident Type"] != "All Major Declarations"]

    fig, ax = plt.subplots(figsize=(12, 6))

    b1 = ax.bar(x - w, ia_vals, width=w, color=COLOURS["ia"],
                label="Individual Assistance (IA)", edgecolor="white", lw=0.5)
    b2 = ax.bar(x,     pa_vals, width=w, color=COLOURS["pa"],
                label="Public Assistance (PA)",     edgecolor="white", lw=0.5)
    b3 = ax.bar(x + w, hm_vals, width=w, color=COLOURS["hm"],
                label="Hazard Mitigation (HM)",     edgecolor="white", lw=0.5)

    for bar, val in zip(list(b1) + list(b2) + list(b3),
                        ia_vals + pa_vals + hm_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{val:.0f}%", ha="center", va="bottom",
                fontsize=8, color="#333")

    # Annotate IA–PA gap on the Flood bar (most representative)
    flood_i = types_plot.index("Flood") if "Flood" in types_plot else 1
    gap_val = round(pa_vals[flood_i] - ia_vals[flood_i], 1)
    ax.annotate("", xy=(x[flood_i], pa_vals[flood_i] - 1),
                xytext=(x[flood_i] - w, ia_vals[flood_i] + 1),
                arrowprops=dict(arrowstyle="<->", color=COLOURS["gap"], lw=1.5))
    ax.text(x[flood_i] - w * 1.6, (ia_vals[flood_i] + pa_vals[flood_i]) / 2,
            f"Gap\n{gap_val:.0f} pp", ha="center", va="center",
            fontsize=8, color=COLOURS["gap"], fontweight="bold")

    # Overall mean reference lines
    ax.axhline(all_row["IA Rate (%)"], color=COLOURS["ia"], linestyle="--",
               lw=1.2, alpha=0.6, label=f'Mean IA = {all_row["IA Rate (%)"]}%')
    ax.axhline(all_row["PA Rate (%)"], color=COLOURS["pa"], linestyle="--",
               lw=1.2, alpha=0.6, label=f'Mean PA = {all_row["PA Rate (%)"]}%')

    ax.set_xticks(x)
    ax.set_xticklabels(types_plot, fontsize=10)
    ax.set_ylabel("Activation Rate (%)", fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_title(
        "Figure 2 — Assistance Programme Activation Rates by Incident Type (2000–2025)\n"
        "Source: OpenFEMA Disaster Declarations Summaries v2",
        fontsize=12, fontweight="bold",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)

    plt.tight_layout()
    out1 = f"{FIG_DIR}/fig_activation_gap.png"
    plt.savefig(out1, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    log(f"Saved → {out1}")

    # ── Figure B — IA/PA gap trend across cohorts ─────────────────────
    if "cohort_5yr" in df.columns:
        cgrp = (
            df.groupby("cohort_5yr")
            .agg(
                ia_rate=("ia_flag", lambda x: x.sum() / len(x) * 100),
                pa_rate=("pa_flag", lambda x: x.sum() / len(x) * 100),
                hm_rate=("hm_flag", lambda x: x.sum() / len(x) * 100),
            )
            .reset_index()
        )
        cgrp["_sort"] = cgrp["cohort_5yr"].map(
            {c: i for i, c in enumerate(COHORT_ORDER)}
        )
        cgrp = (
            cgrp[cgrp["_sort"].notna()]
            .sort_values("_sort")
            .drop(columns=["_sort"])
            .reset_index(drop=True)
        )

        c_x = np.arange(len(cgrp))
        fig2, ax3 = plt.subplots(figsize=(10, 5))

        ax3.plot(c_x, cgrp["pa_rate"], "o-", color=COLOURS["pa"],
                 lw=2.5, ms=7, mfc="white", mew=2, label="PA activation rate (%)")
        ax3.plot(c_x, cgrp["ia_rate"], "s--", color=COLOURS["ia"],
                 lw=2.5, ms=7, mfc="white", mew=2, label="IA activation rate (%)")
        ax3.plot(c_x, cgrp["hm_rate"], "^:", color=COLOURS["hm"],
                 lw=1.8, ms=6, mfc="white", mew=1.5, label="HM activation rate (%)")

        ax3.fill_between(c_x, cgrp["ia_rate"], cgrp["pa_rate"],
                         alpha=0.10, color=COLOURS["gap"], label="IA–PA gap")

        for i, row in cgrp.iterrows():
            ax3.annotate(f"{row['pa_rate']:.0f}%", (i, row["pa_rate"]),
                         textcoords="offset points", xytext=(0, 7),
                         ha="center", fontsize=8, color=COLOURS["pa"])
            ax3.annotate(f"{row['ia_rate']:.0f}%", (i, row["ia_rate"]),
                         textcoords="offset points", xytext=(0, -14),
                         ha="center", fontsize=8, color=COLOURS["ia"])

        ax3.set_xticks(c_x)
        ax3.set_xticklabels(cgrp["cohort_5yr"].values, fontsize=10)
        ax3.set_ylabel("Activation Rate (%)", fontsize=11)
        ax3.set_title(
            "IA vs PA Activation Rate Trend by Cohort (2000–2025)\n"
            "Shaded region = IA–PA gap",
            fontsize=12, fontweight="bold",
        )
        ax3.set_ylim(0, 115)
        ax3.spines[["top", "right"]].set_visible(False)
        ax3.grid(axis="y", linestyle="--", alpha=0.35)
        ax3.legend(fontsize=9, loc="lower right")

        plt.tight_layout()
        out2 = f"{FIG_DIR}/fig_ia_pa_gap_trend.png"
        plt.savefig(out2, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close()
        log(f"Saved → {out2}")

    print(
        f"\n✓ Activation analysis complete."
        f"\n  Overall IA rate   : {all_row['IA Rate (%)']}%"
        f"\n  Overall PA rate   : {all_row['PA Rate (%)']}%"
        f"\n  IA–PA gap overall : {all_row['IA–PA Gap (pp)']} pp"
    )


if __name__ == "__main__":
    main()
