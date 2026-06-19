#!/usr/bin/env python3
"""
Script 06 — Geographic Clustering Analysis
==========================================
GeoBlock-DRS Research Pipeline | SIET 2026

Groups declarations by US state, identifies regional clusters, and
quantifies the geographic concentration of disaster risk documented
in Section 5.2 of the paper.

Outputs
-------
  outputs/tables/table_state_clusters.csv
  outputs/figures/fig_state_declaration_freq.png
  outputs/figures/fig_county_distribution.png
  outputs/figures/fig_geographic_clustering.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import os

# ── Configuration ────────────────────────────────────────────────────
CLEAN_FILE        = "data/clean_declarations.csv"
FIG_DIR           = "outputs/figures"
TBL_DIR           = "outputs/tables"

GULF_STATES       = ["LA", "TX", "MS", "AL", "FL"]
APPALACHIAN_STATES = ["WV", "KY", "TN", "VA", "NC"]
HIGH_SVI_STATES   = ["WV", "LA", "MS", "AL", "KY", "AR", "NM", "AK"]

STATE_NAMES = {
    "AL":"Alabama","AK":"Alaska","AZ":"Arizona","AR":"Arkansas",
    "CA":"California","CO":"Colorado","CT":"Connecticut","DE":"Delaware",
    "FL":"Florida","GA":"Georgia","HI":"Hawaii","ID":"Idaho",
    "IL":"Illinois","IN":"Indiana","IA":"Iowa","KS":"Kansas",
    "KY":"Kentucky","LA":"Louisiana","ME":"Maine","MD":"Maryland",
    "MA":"Massachusetts","MI":"Michigan","MN":"Minnesota","MS":"Mississippi",
    "MO":"Missouri","MT":"Montana","NE":"Nebraska","NV":"Nevada",
    "NH":"New Hampshire","NJ":"New Jersey","NM":"New Mexico","NY":"New York",
    "NC":"North Carolina","ND":"North Dakota","OH":"Ohio","OK":"Oklahoma",
    "OR":"Oregon","PA":"Pennsylvania","RI":"Rhode Island",
    "SC":"South Carolina","SD":"South Dakota","TN":"Tennessee",
    "TX":"Texas","UT":"Utah","VT":"Vermont","VA":"Virginia",
    "WA":"Washington","WV":"West Virginia","WI":"Wisconsin","WY":"Wyoming",
    "DC":"District of Columbia",
}

CLUSTER_COLOURS = {
    "Gulf Coast":   "#0C447C",
    "Appalachian":  "#534AB7",
    "High SVI":     "#D85A30",
    "Other":        "#85B7EB",
}


def log(msg: str) -> None:
    print(f"  {msg}", flush=True)


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TBL_DIR, exist_ok=True)
    print("\n=== Script 06 — Geographic Clustering Analysis ===\n")

    df = pd.read_csv(CLEAN_FILE, low_memory=False)
    log(f"Loaded {len(df):,} records")

    state_col = "state" if "state" in df.columns else None
    if state_col is None:
        log("WARNING: 'state' column not found. Skipping.")
        return

    # ── State-level summary ───────────────────────────────────────────
    state_counts = (
        df.groupby(state_col)
        .agg(
            declarations = ("disasterNumber", "count"),
            mean_latency = ("response_latency_days", "mean"),
            ia_rate      = ("ia_flag",  lambda x: x.sum() / len(x) * 100),
            pa_rate      = ("pa_flag",  lambda x: x.sum() / len(x) * 100),
        )
        .reset_index()
    )
    state_counts.columns = [
        "State", "Declarations", "Mean Latency (d)", "IA Rate (%)", "PA Rate (%)"
    ]
    state_counts["Mean Latency (d)"] = state_counts["Mean Latency (d)"].round(1)
    state_counts["IA Rate (%)"]      = state_counts["IA Rate (%)"].round(1)
    state_counts["PA Rate (%)"]      = state_counts["PA Rate (%)"].round(1)
    state_counts["State Name"]       = state_counts["State"].map(STATE_NAMES).fillna(
        state_counts["State"]
    )
    state_counts = state_counts.sort_values(
        "Declarations", ascending=False
    ).reset_index(drop=True)
    state_counts["Rank"] = state_counts.index + 1

    # Tag clusters
    state_counts["Cluster"] = "Other"
    mask_gulf = state_counts["State"].isin(GULF_STATES)
    mask_appa = state_counts["State"].isin(APPALACHIAN_STATES)
    mask_svi  = state_counts["State"].isin(HIGH_SVI_STATES)
    state_counts.loc[mask_gulf, "Cluster"] = "Gulf Coast"
    state_counts.loc[mask_appa, "Cluster"] = "Appalachian"
    state_counts.loc[mask_svi & (state_counts["Cluster"] == "Other"), "Cluster"] = "High SVI"

    total = state_counts["Declarations"].sum()
    log(f"States analysed: {len(state_counts)}")
    log(f"Top 5 by declarations:\n"
        + state_counts[["State", "Declarations"]].head(5).to_string(index=False))

    gulf_sum = state_counts[mask_gulf]["Declarations"].sum()
    appa_sum = state_counts[mask_appa]["Declarations"].sum()
    log(f"\nGulf Coast  : {gulf_sum:,}  ({gulf_sum / total * 100:.1f}% of total)")
    log(f"Appalachian : {appa_sum:,}  ({appa_sum / total * 100:.1f}% of total)")
    log(f"Combined    : {gulf_sum + appa_sum:,}  ({(gulf_sum + appa_sum) / total * 100:.1f}%)")

    # ── Save cluster table ────────────────────────────────────────────
    state_counts.to_csv(f"{TBL_DIR}/table_state_clusters.csv", index=False)
    log(f"Saved → {TBL_DIR}/table_state_clusters.csv")

    # ── Figure A — top 20 states horizontal bar ───────────────────────
    top20      = state_counts.head(20).copy()
    bar_colors = [CLUSTER_COLOURS.get(c, "#85B7EB") for c in top20["Cluster"]]

    fig, ax = plt.subplots(figsize=(11, 9))
    bars = ax.barh(
        top20["State Name"].values[::-1],
        top20["Declarations"].values[::-1],
        color=bar_colors[::-1],
        edgecolor="white", linewidth=0.4, height=0.72,
    )
    for bar, val in zip(bars, top20["Declarations"].values[::-1]):
        ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", ha="left", fontsize=8.5, color="#333")

    patches = [mpatches.Patch(color=v, label=k) for k, v in CLUSTER_COLOURS.items()]
    ax.legend(handles=patches, fontsize=9, loc="lower right",
              title="Region cluster", title_fontsize=9)
    ax.set_xlabel("Major Disaster Declarations (2000–2025)", fontsize=11)
    ax.set_title(
        "Top 20 States by FEMA Major Disaster Declaration Frequency\n"
        "Colour = geographic cluster  |  Source: OpenFEMA v2",
        fontsize=12, fontweight="bold",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_xlim(0, top20["Declarations"].max() * 1.15)
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    plt.tight_layout()
    out1 = f"{FIG_DIR}/fig_state_declaration_freq.png"
    plt.savefig(out1, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    log(f"Saved → {out1}")

    # ── Figure B — distribution + IA scatter ─────────────────────────
    fig2, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 5.5))

    vals = state_counts["Declarations"].values
    ax_l.hist(vals, bins=20, color="#185FA5", edgecolor="white",
              linewidth=0.5, alpha=0.85)
    ax_l.axvline(vals.mean(), color="#D85A30", linestyle="--", lw=1.4,
                 label=f"Mean = {vals.mean():.0f}")
    ax_l.axvline(np.median(vals), color="#1D9E75", linestyle=":", lw=1.4,
                 label=f"Median = {np.median(vals):.0f}")
    ax_l.set_xlabel("Declarations per State", fontsize=10)
    ax_l.set_ylabel("Number of States", fontsize=10)
    ax_l.set_title("Declaration Frequency Distribution", fontsize=11, fontweight="bold")
    ax_l.spines[["top", "right"]].set_visible(False)
    ax_l.legend(fontsize=9)
    ax_l.grid(axis="y", linestyle="--", alpha=0.35)

    sc_colors = [CLUSTER_COLOURS.get(c, "#85B7EB") for c in state_counts["Cluster"]]
    ax_r.scatter(
        state_counts["Declarations"], state_counts["IA Rate (%)"],
        c=sc_colors, s=60, edgecolors="white", linewidth=0.5, alpha=0.85,
    )
    for _, row in state_counts.head(8).iterrows():
        ax_r.annotate(row["State"],
                      (row["Declarations"], row["IA Rate (%)"]),
                      textcoords="offset points", xytext=(4, 2), fontsize=7.5)
    ax_r.set_xlabel("Declaration Frequency (2000–2025)", fontsize=10)
    ax_r.set_ylabel("IA Activation Rate (%)", fontsize=10)
    ax_r.set_title("IA Rate vs Declaration Frequency by State", fontsize=11,
                   fontweight="bold")
    ax_r.spines[["top", "right"]].set_visible(False)
    ax_r.grid(linestyle="--", alpha=0.3)
    patches2 = [mpatches.Patch(color=v, label=k) for k, v in CLUSTER_COLOURS.items()]
    ax_r.legend(handles=patches2, fontsize=8, loc="upper right")

    plt.tight_layout()
    out2 = f"{FIG_DIR}/fig_county_distribution.png"
    plt.savefig(out2, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    log(f"Saved → {out2}")

    # ── Figure C — 3-panel cluster summary ───────────────────────────
    fig3  = plt.figure(figsize=(14, 7))
    gs    = gridspec.GridSpec(1, 3, figure=fig3, wspace=0.35)

    cluster_sums = state_counts.groupby("Cluster")["Declarations"].sum()
    wedge_colors = [CLUSTER_COLOURS.get(c, "#aaa") for c in cluster_sums.index]

    ax_p1 = fig3.add_subplot(gs[0])
    wedges, _, autotexts = ax_p1.pie(
        cluster_sums.values, labels=cluster_sums.index,
        autopct="%1.1f%%", colors=wedge_colors,
        startangle=140, pctdistance=0.78,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight("bold")
        at.set_color("white")
    ax_p1.set_title("Declaration Share\nby Region Cluster", fontsize=11,
                    fontweight="bold")

    cluster_ia  = state_counts.groupby("Cluster")["IA Rate (%)"].mean().round(1)
    cluster_pa  = state_counts.groupby("Cluster")["PA Rate (%)"].mean().round(1)
    clusters    = cluster_ia.index.tolist()
    x_pos       = np.arange(len(clusters))

    ax_p2 = fig3.add_subplot(gs[1])
    b_ia = ax_p2.bar(x_pos - 0.2, cluster_ia.values, width=0.35,
                     color="#185FA5", label="IA Rate (%)", edgecolor="white")
    b_pa = ax_p2.bar(x_pos + 0.2, cluster_pa.values, width=0.35,
                     color="#EF9F27", label="PA Rate (%)", edgecolor="white")
    for bar, val in zip(list(b_ia) + list(b_pa),
                        list(cluster_ia.values) + list(cluster_pa.values)):
        ax_p2.text(bar.get_x() + bar.get_width() / 2,
                   bar.get_height() + 0.5,
                   f"{val:.0f}%", ha="center", fontsize=8, color="#333")
    ax_p2.set_xticks(x_pos)
    ax_p2.set_xticklabels(clusters, fontsize=8, rotation=12)
    ax_p2.set_ylabel("Activation Rate (%)", fontsize=10)
    ax_p2.set_title("IA vs PA Rates\nby Region Cluster", fontsize=11,
                    fontweight="bold")
    ax_p2.set_ylim(0, 115)
    ax_p2.spines[["top", "right"]].set_visible(False)
    ax_p2.legend(fontsize=8)
    ax_p2.grid(axis="y", linestyle="--", alpha=0.35)

    cluster_lat = state_counts.groupby("Cluster")["Mean Latency (d)"].mean().round(1)
    bar_c3      = [CLUSTER_COLOURS.get(c, "#aaa") for c in cluster_lat.index]

    ax_p3 = fig3.add_subplot(gs[2])
    bars3 = ax_p3.bar(cluster_lat.index, cluster_lat.values,
                      color=bar_c3, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars3, cluster_lat.values):
        ax_p3.text(bar.get_x() + bar.get_width() / 2,
                   bar.get_height() + 0.1,
                   f"{val:.1f}d", ha="center", fontsize=9,
                   fontweight="bold", color="#333")
    ax_p3.axhline(3.0, color="#D85A30", linestyle="--", lw=1.2,
                  label="72-hr threshold (3 d)")
    ax_p3.set_ylabel("Mean Latency (days)", fontsize=10)
    ax_p3.set_title("Mean Response Latency\nby Region Cluster", fontsize=11,
                    fontweight="bold")
    ax_p3.set_xticklabels(cluster_lat.index, rotation=12, fontsize=8)
    ax_p3.spines[["top", "right"]].set_visible(False)
    ax_p3.legend(fontsize=8)
    ax_p3.grid(axis="y", linestyle="--", alpha=0.35)

    fig3.suptitle(
        "Geographic Clustering Summary — FEMA Major Disaster Declarations (2000–2025)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    out3 = f"{FIG_DIR}/fig_geographic_clustering.png"
    plt.savefig(out3, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    log(f"Saved → {out3}")

    print(
        f"\n✓ Geographic clustering analysis complete."
        f"\n  Top state: {state_counts.iloc[0]['State Name']} "
        f"({state_counts.iloc[0]['Declarations']:,} declarations)"
        f"\n  Gulf Coast  : {gulf_sum:,}  ({gulf_sum / total * 100:.1f}%)"
        f"\n  Appalachian : {appa_sum:,}  ({appa_sum / total * 100:.1f}%)"
    )


if __name__ == "__main__":
    main()
