#!/usr/bin/env python3
"""
Script 04 — Response Latency Cohort Analysis (Table 3)
=======================================================
GeoBlock-DRS Research Pipeline | SIET 2026

Produces Table 3 from the paper:
  "Declaration Response Latency by Five-Year Cohort (2000–2025)"

The final cohort (2020–2025) spans six years to include the most recent
complete calendar year available in the OpenFEMA dataset at the time
of analysis. All other cohorts span five years.

Outputs
-------
  outputs/tables/table3_latency_cohorts.csv
  outputs/tables/table3_latency_cohorts.txt
  outputs/figures/fig_latency_trend.png
  outputs/figures/fig_latency_heatmap.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from scipy import stats

# ── Configuration ────────────────────────────────────────────────────
CLEAN_FILE    = "data/clean_declarations.csv"
FIG_DIR       = "outputs/figures"
TBL_DIR       = "outputs/tables"

# Final cohort is 2020–2025 (six years) — all others are five years
COHORT_ORDER  = ["2000–2004", "2005–2009", "2010–2014", "2015–2019", "2020–2025"]
CRITICAL_DAYS = 3.0    # 72-hour survivability threshold in days

COLOURS = {
    "mean":   "#185FA5",
    "median": "#5DCAA5",
    "72hr":   "#D85A30",
    "target": "#1D9E75",
}

TOP_TYPES = ["Severe Storm(s)", "Flood", "Hurricane", "Tornado", "Fire/Wildfire"]


def log(msg: str) -> None:
    print(f"  {msg}", flush=True)


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TBL_DIR, exist_ok=True)
    print("\n=== Script 04 — Response Latency Cohort Analysis ===\n")

    df = pd.read_csv(CLEAN_FILE, low_memory=False)
    log(f"Loaded {len(df):,} records")

    # ── Build cohort aggregation ──────────────────────────────────────
    grp = (
        df.groupby("cohort_5yr")["response_latency_days"]
        .agg(
            mean_latency   = "mean",
            median_latency = "median",
            std_latency    = "std",
            n_declarations = "count",
        )
        .reset_index()
    )
    grp.columns = ["Cohort", "Mean (d)", "Median (d)", "Std Dev (d)", "n"]

    # Sort by COHORT_ORDER — only keep cohorts in the expected list
    sort_map = {c: i for i, c in enumerate(COHORT_ORDER)}
    grp["_sort"] = grp["Cohort"].map(sort_map)
    grp = (
        grp[grp["_sort"].notna()]
        .sort_values("_sort")
        .drop(columns=["_sort"])
        .reset_index(drop=True)
    )
    for col in ["Mean (d)", "Median (d)", "Std Dev (d)"]:
        grp[col] = grp[col].round(2)

    print(grp.to_string(index=False))

    # ── Save table ────────────────────────────────────────────────────
    grp.to_csv(f"{TBL_DIR}/table3_latency_cohorts.csv", index=False)
    with open(f"{TBL_DIR}/table3_latency_cohorts.txt", "w") as fh:
        fh.write(
            "Table 3. Declaration Response Latency by Cohort (2000–2025)\n"
            + "-" * 72 + "\n"
        )
        fh.write(grp.to_string(index=False))
        fh.write(
            "\n" + "-" * 72
            + "\nSource: Authors' analysis of OpenFEMA Disaster Declarations "
              "Summaries v2.\n"
            + "Note: The 2020–2025 cohort spans six years (2020–2024 plus 2025)\n"
            + "to include the most recent complete calendar year available.\n"
        )
    log(f"Saved table → {TBL_DIR}/table3_latency_cohorts.csv")

    # ── Linear trend test ─────────────────────────────────────────────
    mean_vals = grp["Mean (d)"].values
    x_trend   = np.arange(len(mean_vals))
    slope, intercept, r_val, p_val, _ = stats.linregress(x_trend, mean_vals)
    log(f"Trend: slope={slope:.2f} d/cohort  R²={r_val**2:.3f}  p={p_val:.4f}")

    # Projection: how many more cohorts to reach 3-day threshold?
    if slope < 0:
        cohorts_to_threshold = (CRITICAL_DAYS - intercept) / slope
        proj_year = 2000 + cohorts_to_threshold * 5
        log(f"Projected year to reach {CRITICAL_DAYS}-day threshold: ~{proj_year:.0f}")

    # ── Figure A — latency trend line ─────────────────────────────────
    x     = np.arange(len(grp))
    mean  = grp["Mean (d)"].values
    med   = grp["Median (d)"].values
    std   = grp["Std Dev (d)"].values

    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.fill_between(x, mean - std * 0.5, mean + std * 0.5,
                    color=COLOURS["mean"], alpha=0.10, label="±0.5 SD (mean)")
    ax.plot(x, mean, "o-", color=COLOURS["mean"], lw=2.5, ms=8,
            mfc="white", mew=2.0, label="Mean latency (days)", zorder=5)
    ax.plot(x, med, "s--", color=COLOURS["median"], lw=1.8, ms=6,
            mfc="white", mew=1.5, label="Median latency (days)", zorder=4)

    for i, (m, md) in enumerate(zip(mean, med)):
        ax.annotate(f"{m:.1f}", (x[i], m), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9.5,
                    fontweight="bold", color=COLOURS["mean"])
        ax.annotate(f"{md:.1f}", (x[i], md), textcoords="offset points",
                    xytext=(0, -15), ha="center", fontsize=8.5,
                    color=COLOURS["median"])

    # 72-hour critical window (3 days)
    ax.axhline(CRITICAL_DAYS, color=COLOURS["72hr"], linestyle=":",
               lw=1.6, label=f"72-hr critical window ({CRITICAL_DAYS} days)", zorder=3)

    # GeoBlock-DRS target (~6.3 s = near-zero on day scale)
    sc_days = 6.3 / 86400
    ax.axhline(sc_days, color=COLOURS["target"], linestyle="-.", lw=1.4,
               label="GeoBlock-DRS target (6.3 s ≈ 0.000073 d)", zorder=3)

    # Trend line
    trend_y = intercept + slope * x_trend
    ax.plot(x, trend_y, color="#aaa", lw=1.2, linestyle=(0, (3, 5)),
            label=f"Linear trend (slope={slope:.2f} d/cohort, p={p_val:.3f})")

    ax.set_xticks(x)
    ax.set_xticklabels(COHORT_ORDER, fontsize=10)
    ax.set_xlabel("Cohort", fontsize=11)
    ax.set_ylabel("Response Latency (days)", fontsize=11)
    ax.set_title(
        "Figure 3 — Declaration Response Latency by Cohort (2000–2025)\n"
        "OpenFEMA Major Disaster Declarations",
        fontsize=12, fontweight="bold",
    )
    ax.set_ylim(0, max(mean) * 1.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.9)

    for i, row in grp.iterrows():
        ax.text(i, -1.8, f"n={int(row['n']):,}", ha="center",
                fontsize=7.5, color="#666")

    plt.tight_layout()
    out1 = f"{FIG_DIR}/fig_latency_trend.png"
    plt.savefig(out1, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    log(f"Saved → {out1}")

    # ── Figure B — incident type × cohort heatmap ─────────────────────
    it_col  = "incidentType_clean" if "incidentType_clean" in df.columns else "incidentType"
    df_top  = df[df[it_col].isin(TOP_TYPES)].copy()
    pivot   = df_top.pivot_table(
        index=it_col, columns="cohort_5yr",
        values="response_latency_days", aggfunc="mean",
    ).round(1)
    pivot = pivot[[c for c in COHORT_ORDER if c in pivot.columns]]
    pivot = pivot.reindex([t for t in TOP_TYPES if t in pivot.index])

    fig2, ax2 = plt.subplots(figsize=(10, 4.5))
    im = ax2.imshow(pivot.values, cmap="Blues", aspect="auto", vmin=0, vmax=20)
    ax2.set_xticks(range(pivot.shape[1]))
    ax2.set_xticklabels(pivot.columns, fontsize=10)
    ax2.set_yticks(range(pivot.shape[0]))
    ax2.set_yticklabels(pivot.index, fontsize=10)
    ax2.set_title(
        "Mean Response Latency (days) — Incident Type × Cohort Heatmap",
        fontsize=12, fontweight="bold",
    )
    for r in range(pivot.shape[0]):
        for c in range(pivot.shape[1]):
            val = pivot.values[r, c]
            if not np.isnan(val):
                colour = "white" if val > 12 else "black"
                ax2.text(c, r, f"{val:.1f}", ha="center", va="center",
                         fontsize=10, fontweight="bold", color=colour)
    cbar = plt.colorbar(im, ax=ax2, shrink=0.85)
    cbar.set_label("Mean latency (days)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    plt.tight_layout()
    out2 = f"{FIG_DIR}/fig_latency_heatmap.png"
    plt.savefig(out2, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    log(f"Saved → {out2}")

    print(
        f"\n✓ Latency analysis complete."
        f"\n  Mean trend: {mean[0]:.1f} → {mean[-1]:.1f} days"
        f"\n  Slope = {slope:.2f} d/cohort  (p = {p_val:.4f})"
    )


if __name__ == "__main__":
    main()
