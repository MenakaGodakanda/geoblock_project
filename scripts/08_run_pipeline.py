#!/usr/bin/env python3
"""
Script 08 — Master Pipeline Runner
====================================
GeoBlock-DRS Research Pipeline | SIET 2026

Executes the complete analysis pipeline in the correct order and
produces a self-contained HTML summary report embedding all figures,
tables, and text reports.

Pipeline order
--------------
  00  Generate synthetic dataset  (or use 01 for real data)
  02  Clean and preprocess
  03  Incident type frequency analysis  (Table 1)
  04  Latency cohort analysis           (Table 3)
  05  Assistance activation analysis    (Table 2)
  06  Geographic clustering analysis
  07  Smart contract simulation

Usage
-----
  python3 scripts/08_run_pipeline.py                 # full run (includes 00)
  python3 scripts/08_run_pipeline.py --skip-download # skip Script 00/01

After Script 01 downloads the real FEMA data, run:
  python3 scripts/08_run_pipeline.py --skip-download

Outputs
-------
  outputs/reports/pipeline_run_log.txt
  outputs/reports/final_summary_report.html
"""

import subprocess
import sys
import os
import time
import glob
import base64
from datetime import datetime

# ── Configuration ────────────────────────────────────────────────────
SCRIPTS_DIR = "scripts"
LOG_FILE    = "outputs/reports/pipeline_run_log.txt"
HTML_REPORT = "outputs/reports/final_summary_report.html"

PIPELINE = [
    ("00_generate_synthetic_data.py", "Generate synthetic dataset (offline)"),
    ("02_clean_data.py",              "Clean and preprocess data"),
    ("03_incident_type_analysis.py",  "Incident type frequency — Table 1"),
    ("04_latency_analysis.py",        "Latency cohort analysis — Table 3"),
    ("05_activation_analysis.py",     "Assistance activation rates — Table 2"),
    ("06_geographic_clustering.py",   "Geographic clustering analysis"),
    ("07_smart_contract_simulation.py","Smart contract simulation"),
]


def log(msg: str, fh=None) -> None:
    ts  = datetime.now().strftime("%H:%M:%S")
    out = f"[{ts}] {msg}"
    print(out, flush=True)
    if fh:
        fh.write(out + "\n")
        fh.flush()


def run_script(script_name: str, python_exe: str, fh) -> tuple:
    path = os.path.join(SCRIPTS_DIR, script_name)
    log(f"▶  {script_name}", fh)
    t0  = time.perf_counter()
    res = subprocess.run([python_exe, path], capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    ok      = res.returncode == 0
    status  = "✓ OK" if ok else f"✗ FAILED (exit {res.returncode})"
    log(f"   {status}  ({elapsed:.1f}s)", fh)
    if not ok:
        log(f"   STDERR: {res.stderr[-400:]}", fh)
    if fh:
        fh.write(res.stdout[-1500:] + "\n")
    return ok, elapsed, res.stdout, res.stderr


def collect_outputs():
    figures = sorted(glob.glob("outputs/figures/*.png"))
    tables  = sorted(glob.glob("outputs/tables/*.csv"))
    reports = sorted(glob.glob("outputs/reports/*.txt"))
    return figures, tables, reports


def img_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def read_txt(path: str) -> str:
    try:
        with open(path) as f:
            return f.read()
    except Exception:
        return "(could not read)"


def csv_to_html(path: str) -> str:
    try:
        import pandas as pd
        df = pd.read_csv(path)
        return df.to_html(index=False, classes="dt", border=0)
    except Exception:
        return "<p>(could not parse CSV)</p>"


def build_html(run_summary: list) -> str:
    figures, tables, reports = collect_outputs()

    status_rows = "".join(
        f"<tr class='{'ok' if ok else 'fail'}'>"
        f"<td>{name}</td><td>{desc}</td>"
        f"<td>{'✓ OK' if ok else '✗ FAILED'}</td>"
        f"<td>{t:.1f}s</td></tr>"
        for name, desc, ok, t in run_summary
    )

    fig_html = ""
    for p in figures:
        name = os.path.basename(p)
        try:
            fig_html += (
                f'<div class="fb"><h4>{name}</h4>'
                f'<img src="data:image/png;base64,{img_b64(p)}" '
                f'alt="{name}"/></div>'
            )
        except Exception:
            fig_html += f"<p>Could not embed: {name}</p>"

    tbl_html = "".join(
        f"<h4>{os.path.basename(p)}</h4>{csv_to_html(p)}"
        for p in tables
    )
    rpt_html = "".join(
        f"<h4>{os.path.basename(p)}</h4><pre>{read_txt(p)}</pre>"
        for p in reports
    )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"/>
<title>GeoBlock-DRS Pipeline Report — SIET 2026</title>
<style>
  body{{font-family:Georgia,serif;font-size:10.5pt;color:#111;
       background:#f7f6f3;padding:20px;}}
  .c{{max-width:1100px;margin:0 auto;background:#fff;
      padding:32px 40px;border:1px solid #ddd;}}
  h1{{font-size:16pt;color:#042C53;border-bottom:2pt solid #185FA5;
      padding-bottom:6px;}}
  h2{{font-size:13pt;color:#185FA5;margin:24px 0 6px;
      border-bottom:1px solid #ddd;}}
  h4{{font-size:10pt;color:#534AB7;margin:12px 0 3px;}}
  pre{{background:#f4f3f0;padding:10px;font-size:8pt;overflow-x:auto;
       border-left:3px solid #185FA5;line-height:1.4;}}
  table.ps{{width:100%;border-collapse:collapse;font-size:9.5pt;}}
  table.ps th{{background:#185FA5;color:#fff;padding:5px 10px;text-align:left;}}
  table.ps td{{padding:4px 10px;border-bottom:0.5px solid #ddd;}}
  tr.ok td:nth-child(3){{color:#085041;font-weight:bold;}}
  tr.fail td:nth-child(3){{color:#D85A30;font-weight:bold;}}
  table.dt{{width:100%;border-collapse:collapse;font-size:8.5pt;margin-bottom:14px;}}
  table.dt th{{background:#e8e4dc;padding:3px 7px;text-align:left;
               border-top:1pt solid #555;border-bottom:0.5pt solid #888;}}
  table.dt td{{padding:3px 7px;border-bottom:0.25pt solid #ccc;}}
  .fb{{margin:14px 0;}}
  .fb img{{max-width:100%;border:0.5px solid #ccc;display:block;}}
  .meta{{background:#E1F5EE;border:1px solid #1D9E75;border-radius:5px;
         padding:10px 14px;font-size:9.5pt;margin-bottom:18px;}}
</style></head><body><div class="c">
<h1>GeoBlock-DRS Research Pipeline — Final Summary Report</h1>
<div class="meta">
  <strong>Paper:</strong> A Blockchain-Anchored Smart Contract and GIS Framework
  for Resilient Disaster Resource Coordination in Smart Cities<br/>
  <strong>Conference:</strong> SIET 2026 — Springer LNNS<br/>
  <strong>Dataset:</strong> OpenFEMA Disaster Declarations Summaries v2 (2000–2025)<br/>
  <strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
</div>
<h2>1. Pipeline Execution Status</h2>
<table class="ps">
  <tr><th>Script</th><th>Description</th><th>Status</th><th>Time</th></tr>
  {status_rows}
</table>
<h2>2. Figures</h2>{fig_html if fig_html else '<p>None generated.</p>'}
<h2>3. Tables</h2>{tbl_html if tbl_html else '<p>None generated.</p>'}
<h2>4. Reports</h2>{rpt_html if rpt_html else '<p>None generated.</p>'}
</div></body></html>"""


def main() -> None:
    os.makedirs("outputs/reports", exist_ok=True)
    skip_download = "--skip-download" in sys.argv

    # Detect Python executable
    venv_python = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "geoblock_env", "bin", "python3",
    )
    if not os.path.exists(venv_python):
        venv_python = sys.executable

    print("=" * 60)
    print("GeoBlock-DRS Analysis Pipeline")
    print(f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python  : {venv_python}")
    print(f"Mode    : {'skip-download' if skip_download else 'full run'}")
    print("=" * 60)

    run_summary = []
    with open(LOG_FILE, "w") as log_fh:
        log(f"Pipeline started: {datetime.now().isoformat()}", log_fh)

        for script_name, desc in PIPELINE:
            if script_name == "00_generate_synthetic_data.py" and skip_download:
                log(f"⏭  SKIPPED: {script_name}", log_fh)
                run_summary.append((script_name, desc, True, 0.0))
                continue

            ok, elapsed, stdout, stderr = run_script(script_name, venv_python, log_fh)
            run_summary.append((script_name, desc, ok, elapsed))

        success = sum(1 for _, _, ok, _ in run_summary if ok)
        log(f"Pipeline finished: {datetime.now().isoformat()}", log_fh)
        log(f"Scripts succeeded : {success}/{len(run_summary)}", log_fh)

    print(f"\nLog saved → {LOG_FILE}")

    print("Building HTML summary report ...")
    html = build_html(run_summary)
    with open(HTML_REPORT, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"Report saved → {HTML_REPORT}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    for name, desc, ok, t in run_summary:
        print(f"  {'✓' if ok else '✗'}  {name:<42s}  {t:.1f}s")

    figures, tables, reports = collect_outputs()
    print(f"\n  Figures : {len(figures)}")
    print(f"  Tables  : {len(tables)}")
    print(f"  Reports : {len(reports)}")
    print(f"\n  All outputs in: outputs/")
    print("=" * 60)


if __name__ == "__main__":
    main()
