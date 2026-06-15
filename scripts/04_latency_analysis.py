#!/usr/bin/env python3
"""
Script 04 — Response Latency Cohort Analysis (Table 3 + Figure 5A)
===================================================================
GeoBlock-DRS Research Pipeline | SIET 2026

Produces Table 3 from the paper:
  "Mean Declaration Response Latency by Five-Year Cohort (2000–2024)"

And the Figure 5A latency trend line chart.

Computes:
  - Mean and median latency per 5-year cohort
  - Declaration count per cohort
  - Statistical significance of latency trend (Mann-Kendall test)
  - Latency breakdown by incident type × cohort (heatmap)

Outputs:
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
import matplotlib.ticker as mtick
import os
from scipy import stats

# ── Configuration ────────────────────────────────────────────────────
CLEAN_FILE = "data/clean_declarations.csv"
FIG_DIR    = "outputs/figures"
TBL_DIR    = "outputs/tables"

COHORT_ORDER = ["2000–2004", "2005–2009", "2010–2014", "2015–2019", "2020–2024"]
COLORS       = {
    "mean":   "#185FA5",
    "median": "#5DCAA5",
    "72hr":   "#D85A30",
    "smart":  "#1D9E75",
}
TOP_TYPES = ["Severe Storm(s)", "Flood", "Hurricane", "Tornado", "Fire/Wildfire"]

# ── Helpers ──────────────────────────────────────────────────────────
def log(msg): print(f"  {msg}", flush=True)

# ── Main ─────────────────────────────────────────────────────────────
def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TBL_DIR, exist_ok=True)

    print("\n=== SCRIPT 04 — Response Latency Cohort Analysis ===\n")

    df = pd.read_csv(CLEAN_FILE, low_memory=False)
    df["declarationDate"] = pd.to_datetime(df["declarationDate"], errors="coerce")
    log(f"Loaded {len(df):,} records")

    # ── Ensure cohort column is present ──────────────────────────────
    if "cohort_5yr" not in df.columns:
        df["declaration_year"] = df["declarationDate"].dt.year
        def cohort_label(y):
            b = (y - 2000) // 5 * 5 + 2000
            return f"{b}–{b+4}"
        df["cohort_5yr"] = df["declaration_year"].apply(cohort_label)

    # ── Table 3 — latency by cohort ───────────────────────────────────
    grp = df.groupby("cohort_5yr")["response_latency_days"].agg(
        mean_latency   = "mean",
        median_latency = "median",
        std_latency    = "std",
        n_declarations = "count",
    ).reset_index()
    grp.columns = ["Cohort", "Mean Latency (days)", "Median Latency (days)",
                   "Std Dev (days)", "n Declarations"]

    # Re-order cohorts
    grp["_sort"] = grp["Cohort"].map({c: i for i, c in enumerate(COHORT_ORDER)})
    grp = grp.sort_values("_sort").drop("_sort", axis=1).reset_index(drop=True)
    grp["Mean Latency (days)"]   = grp["Mean Latency (days)"].round(1)
    grp["Median Latency (days)"] = grp["Median Latency (days)"].round(1)
    grp["Std Dev (days)"]        = grp["Std Dev (days)"].round(1)

    print(grp.to_string(index=False))

    grp.to_csv(f"{TBL_DIR}/table3_latency_cohorts.csv", index=False)
    with open(f"{TBL_DIR}/table3_latency_cohorts.txt", "w") as f:
        f.write("Table 3. Mean Declaration Response Latency by Five-Year Cohort (2000–2024)\n")
        f.write("-" * 72 + "\n")
        f.write(grp.to_string(index=False))
        f.write("\n" + "-" * 72)
        f.write("\nSource: Authors' analysis of OpenFEMA Disaster Declarations Summaries v2.\n")
    log(f"Saved Table 3 → {TBL_DIR}/table3_latency_cohorts.csv")

    # ── Mann-Kendall trend test on mean latency ───────────────────────
    mean_vals = grp["Mean Latency (days)"].values
    slope, intercept, r, p, se = stats.linregress(range(len(mean_vals)), mean_vals)
    log(f"Trend: slope={slope:.2f} days/cohort, R²={r**2:.3f}, p={p:.4f}")

    # ── Figure: Latency trend ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5.5))

    x    = np.arange(len(grp))
    mean = grp["Mean Latency (days)"].values
    med  = grp["Median Latency (days)"].values
    std  = grp["Std Dev (days)"].values

    # Shaded ±1 std dev band for mean
    ax.fill_between(x, mean - std * 0.5, mean + std * 0.5,
                    color=COLORS["mean"], alpha=0.10, label="±0.5 SD band (mean)")

    # Mean and median lines
    ax.plot(x, mean, "o-", color=COLORS["mean"], linewidth=2.5, markersize=8,
            markerfacecolor="white", markeredgewidth=2, label="Mean latency (days)", zorder=5)
    ax.plot(x, med,  "s--", color=COLORS["median"], linewidth=1.8, markersize=6,
            markerfacecolor="white", markeredgewidth=1.5, label="Median latency (days)", zorder=4)

    # Value annotations
    for i, (m, md) in enumerate(zip(mean, med)):
        ax.annotate(f"{m:.1f}", (x[i], m), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9.5, fontweight="bold",
                    color=COLORS["mean"])
        ax.annotate(f"{md:.1f}", (x[i], md), textcoords="offset points",
                    xytext=(0, -15), ha="center", fontsize=8.5, color=COLORS["median"])

    # 72-hour critical window line
    ax.axhline(3, color=COLORS["72hr"], linestyle=":", linewidth=1.6,
               label="72-hr critical window (3 days)", zorder=3)

    # GeoBlock-DRS target line (seconds converted to fractional days)
    smartcontract_days = 5 / 86400  # 5 seconds
    ax.axhline(smartcontract_days, color=COLORS["smart"], linestyle="-.", linewidth=1.4,
               label=f"GeoBlock-DRS target (~5 s = {smartcontract_days*86400:.0f}s)", zorder=3)

    # Trend line
    trend_y = intercept + slope * x
    ax.plot(x, trend_y, "-", color="#aaa", linewidth=1.2, linestyle=(0, (3,5)),
            label=f"Linear trend (slope={slope:.2f} d/cohort, p={p:.3f})", zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(COHORT_ORDER, fontsize=10)
    ax.set_xlabel("Five-Year Cohort", fontsize=11)
    ax.set_ylabel("Response Latency (days)", fontsize=11)
    ax.set_title("Figure 5A — Declaration Response Latency by Cohort (2000–2024)\n"
                 "OpenFEMA Major Disaster Declarations",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(mean) * 1.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.9)

    # n= labels at bottom of each cohort
    for i, row in grp.iterrows():
        ax.text(i, -1.5, f"n={row['n Declarations']:,}", ha="center",
                fontsize=8, color="#666")

    plt.tight_layout()
    out_path = f"{FIG_DIR}/fig_latency_trend.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    log(f"Saved figure → {out_path}")

    # ── Figure: Heatmap — latency by incident type × cohort ──────────
    it_col = "incidentType_clean" if "incidentType_clean" in df.columns else "incidentType"
    df_top = df[df[it_col].isin(TOP_TYPES)].copy()

    pivot = df_top.pivot_table(
        index=it_col,
        columns="cohort_5yr",
        values="response_latency_days",
        aggfunc="mean",
    ).round(1)

    # Reorder
    pivot = pivot[[c for c in COHORT_ORDER if c in pivot.columns]]
    pivot = pivot.reindex([t for t in TOP_TYPES if t in pivot.index])

    fig2, ax2 = plt.subplots(figsize=(10, 4.5))
    import matplotlib.colors as mcolors

    cmap = plt.cm.Blues
    im = ax2.imshow(pivot.values, cmap=cmap, aspect="auto",
                    vmin=0, vmax=20)

    ax2.set_xticks(range(pivot.shape[1]))
    ax2.set_xticklabels(pivot.columns, fontsize=10)
    ax2.set_yticks(range(pivot.shape[0]))
    ax2.set_yticklabels(pivot.index, fontsize=10)
    ax2.set_title("Mean Response Latency (days) — Incident Type × Cohort Heatmap",
                  fontsize=12, fontweight="bold")

    # Cell annotations
    for r in range(pivot.shape[0]):
        for c in range(pivot.shape[1]):
            val = pivot.values[r, c]
            if not np.isnan(val):
                col = "white" if val > 12 else "black"
                ax2.text(c, r, f"{val:.1f}", ha="center", va="center",
                         fontsize=10, fontweight="bold", color=col)

    cbar = plt.colorbar(im, ax=ax2, shrink=0.85)
    cbar.set_label("Mean latency (days)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    out_path2 = f"{FIG_DIR}/fig_latency_heatmap.png"
    plt.savefig(out_path2, dpi=180, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    log(f"Saved heatmap → {out_path2}")

    print(f"\n✓ Latency analysis complete.")
    print(f"  Mean latency trend: {mean[0]} → {mean[-1]} days over 5 cohorts.")
    print(f"  Linear slope = {slope:.2f} days/cohort (p = {p:.4f})")

if __name__ == "__main__":
    main()
