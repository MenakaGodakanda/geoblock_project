#!/usr/bin/env python3
"""
Script 08 — Master Pipeline Runner & Final Summary Report
==========================================================
GeoBlock-DRS Research Pipeline | SIET 2026

Runs the complete analysis pipeline in sequence and produces
a final consolidated HTML report summarising all outputs.

Usage:
    python scripts/08_run_pipeline.py [--skip-download]

Flags:
    --skip-download   Skip Script 01 (useful if data already downloaded)

Pipeline order:
    01_download_data.py
    02_clean_data.py
    03_incident_type_analysis.py
    04_latency_analysis.py
    05_activation_analysis.py
    06_geographic_clustering.py
    07_smart_contract_simulation.py

Outputs:
    outputs/reports/pipeline_run_log.txt
    outputs/reports/final_summary_report.html
"""

import subprocess
import sys
import os
import time
from datetime import datetime
import glob

# ── Configuration ────────────────────────────────────────────────────
SCRIPTS_DIR = "scripts"
LOG_FILE    = "outputs/reports/pipeline_run_log.txt"
HTML_REPORT = "outputs/reports/final_summary_report.html"

PIPELINE = [
    ("01_download_data.py",        "Download OpenFEMA dataset"),
    ("02_clean_data.py",           "Clean and preprocess data"),
    ("03_incident_type_analysis.py","Incident type frequency (Table 1)"),
    ("04_latency_analysis.py",     "Response latency cohort analysis (Table 3)"),
    ("05_activation_analysis.py",  "Assistance activation rates (Table 2)"),
    ("06_geographic_clustering.py","Geographic clustering analysis"),
    ("07_smart_contract_simulation.py","Smart contract simulation"),
]

def log(msg, file=None):
    ts  = datetime.now().strftime("%H:%M:%S")
    out = f"[{ts}] {msg}"
    print(out, flush=True)
    if file:
        file.write(out + "\n")
        file.flush()

def run_script(script_name: str, venv_python: str, log_file):
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    log(f"▶  Running: {script_name}", log_file)
    t0  = time.perf_counter()
    res = subprocess.run(
        [venv_python, script_path],
        capture_output=True, text=True
    )
    elapsed = time.perf_counter() - t0
    if res.returncode == 0:
        log(f"   ✓  Completed in {elapsed:.1f}s", log_file)
    else:
        log(f"   ✗  FAILED (exit {res.returncode}) after {elapsed:.1f}s", log_file)
        log(f"   STDERR: {res.stderr[-500:]}", log_file)
    log_file.write(res.stdout[-2000:] + "\n")
    return res.returncode == 0, elapsed, res.stdout, res.stderr

def collect_outputs():
    """Collect all generated files for the HTML report."""
    figures = sorted(glob.glob("outputs/figures/*.png"))
    tables  = sorted(glob.glob("outputs/tables/*.csv"))
    reports = sorted(glob.glob("outputs/reports/*.txt"))
    return figures, tables, reports

def build_html_report(run_summary: list):
    """Build a self-contained HTML summary report."""
    import base64

    figures, tables, reports = collect_outputs()

    def img_to_b64(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    def read_txt(path):
        try:
            with open(path) as f:
                return f.read()
        except Exception:
            return "(could not read file)"

    def read_csv_html(path):
        try:
            import pandas as pd
            df = pd.read_csv(path)
            return df.to_html(index=False, classes="data-table", border=0)
        except Exception:
            return "<p>(could not parse CSV)</p>"

    status_rows = "".join(
        f"<tr class='{'ok' if ok else 'fail'}'>"
        f"<td>{name}</td><td>{desc}</td>"
        f"<td>{'✓ OK' if ok else '✗ FAILED'}</td>"
        f"<td>{t:.1f}s</td></tr>"
        for name, desc, ok, t in run_summary
    )

    figure_sections = ""
    for fig_path in figures:
        fig_name = os.path.basename(fig_path)
        try:
            b64 = img_to_b64(fig_path)
            figure_sections += f"""
            <div class="fig-block">
              <h4>{fig_name}</h4>
              <img src="data:image/png;base64,{b64}" alt="{fig_name}"/>
            </div>"""
        except Exception:
            figure_sections += f"<p>Could not embed: {fig_name}</p>"

    table_sections = ""
    for tbl_path in tables:
        tbl_name = os.path.basename(tbl_path)
        table_sections += f"<h4>{tbl_name}</h4>" + read_csv_html(tbl_path)

    report_sections = ""
    for rpt_path in reports:
        rpt_name = os.path.basename(rpt_path)
        report_sections += f"<h4>{rpt_name}</h4><pre>{read_txt(rpt_path)}</pre>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>GeoBlock-DRS Pipeline Report — SIET 2026</title>
<style>
  body {{ font-family: 'Georgia', serif; font-size: 10.5pt; color: #111;
          background: #f7f6f3; margin: 0; padding: 24px; }}
  .container {{ max-width: 1100px; margin: 0 auto; background: #fff;
                padding: 32px 40px; border: 1px solid #ddd; }}
  h1  {{ font-size: 17pt; border-bottom: 2pt solid #185FA5; padding-bottom: 8px; color: #042C53; }}
  h2  {{ font-size: 13pt; color: #185FA5; margin-top: 28px; border-bottom: 1px solid #ddd; }}
  h3  {{ font-size: 11pt; color: #333; margin-top: 18px; }}
  h4  {{ font-size: 10pt; color: #534AB7; margin: 14px 0 4px; }}
  pre {{ background: #f4f3f0; padding: 12px; font-size: 8.5pt; overflow-x: auto;
         border-left: 3px solid #185FA5; line-height: 1.45; }}
  table.pipeline-status {{ width: 100%; border-collapse: collapse; font-size: 9.5pt; }}
  table.pipeline-status th {{ background: #185FA5; color: #fff; padding: 6px 10px; text-align: left; }}
  table.pipeline-status td {{ padding: 5px 10px; border-bottom: 0.5px solid #ddd; }}
  tr.ok   td:nth-child(3) {{ color: #085041; font-weight: bold; }}
  tr.fail td:nth-child(3) {{ color: #D85A30; font-weight: bold; }}
  table.data-table {{ width: 100%; border-collapse: collapse; font-size: 8.5pt; margin-bottom: 16px; }}
  table.data-table th {{ background: #e8e4dc; padding: 4px 8px; text-align: left;
                         border-top: 1pt solid #555; border-bottom: 0.5pt solid #888; }}
  table.data-table td {{ padding: 3px 8px; border-bottom: 0.25pt solid #ccc; }}
  .fig-block {{ margin: 16px 0; }}
  .fig-block img {{ max-width: 100%; border: 0.5px solid #ccc; display: block; }}
  .meta-box {{ background: #E1F5EE; border: 1px solid #1D9E75; border-radius: 6px;
               padding: 12px 16px; font-size: 9.5pt; margin-bottom: 20px; }}
  .sdg-pill {{ display: inline-block; padding: 2px 10px; border-radius: 12px;
               font-size: 9pt; font-weight: bold; color: #fff; margin: 2px; }}
</style>
</head>
<body>
<div class="container">

<h1>GeoBlock-DRS Research Pipeline — Final Summary Report</h1>
<div class="meta-box">
  <strong>Paper:</strong> A Blockchain-Anchored Smart Contract and GIS Framework for Resilient
  Disaster Resource Coordination in Smart Cities: Evidence from OpenFEMA Disaster Declarations Data<br/>
  <strong>Conference:</strong> SIET 2026 — Intelligent and Secure Systems in Large-Scale Connected Digital Environments<br/>
  <strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
  <strong>SDGs:</strong>
  <span class="sdg-pill" style="background:#f0853a;">SDG 11</span>
  <span class="sdg-pill" style="background:#3f8e3a;">SDG 13</span>
  <span class="sdg-pill" style="background:#18486a;">SDG 17</span>
</div>

<h2>1. Pipeline Execution Status</h2>
<table class="pipeline-status">
  <tr><th>Script</th><th>Description</th><th>Status</th><th>Time</th></tr>
  {status_rows}
</table>

<h2>2. Generated Figures</h2>
{figure_sections if figure_sections else "<p>No figures generated.</p>"}

<h2>3. Output Tables</h2>
{table_sections if table_sections else "<p>No tables generated.</p>"}

<h2>4. Analysis Reports</h2>
{report_sections if report_sections else "<p>No reports generated.</p>"}

</div>
</body>
</html>"""
    return html

# ── Main ─────────────────────────────────────────────────────────────
def main():
    os.makedirs("outputs/reports", exist_ok=True)

    skip_download = "--skip-download" in sys.argv

    # Detect virtual environment Python
    venv_python = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "geoblock_env", "bin", "python3")
    if not os.path.exists(venv_python):
        venv_python = sys.executable

    print("=" * 60)
    print("GeoBlock-DRS Analysis Pipeline")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python : {venv_python}")
    print("=" * 60)

    run_summary = []

    with open(LOG_FILE, "w") as log_file:
        log(f"Pipeline started: {datetime.now().isoformat()}", log_file)
        log(f"Python: {venv_python}", log_file)

        for script_name, desc in PIPELINE:
            if script_name == "01_download_data.py" and skip_download:
                log(f"⏭  SKIPPED (--skip-download): {script_name}", log_file)
                run_summary.append((script_name, desc, True, 0.0))
                continue

            ok, elapsed, stdout, stderr = run_script(script_name, venv_python, log_file)
            run_summary.append((script_name, desc, ok, elapsed))

            if not ok:
                log(f"\n⚠ Pipeline halted at {script_name}. Check log for details.", log_file)
                # Continue anyway for report generation
                continue

        log(f"\nPipeline completed: {datetime.now().isoformat()}", log_file)
        success_count = sum(1 for _, _, ok, _ in run_summary if ok)
        log(f"Scripts succeeded: {success_count}/{len(run_summary)}", log_file)

    print(f"\nLog saved → {LOG_FILE}")

    # Build HTML report
    print("Building HTML summary report ...")
    html = build_html_report(run_summary)
    with open(HTML_REPORT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Report saved → {HTML_REPORT}")

    # Final summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE — Summary")
    print("=" * 60)
    for name, desc, ok, t in run_summary:
        status = "✓" if ok else "✗"
        print(f"  {status}  {name:<40s}  {t:.1f}s")

    figures, tables, reports = collect_outputs()
    print(f"\nTotal figures generated : {len(figures)}")
    print(f"Total tables generated  : {len(tables)}")
    print(f"Total reports generated : {len(reports)}")
    print(f"\nAll outputs in: outputs/")
    print("=" * 60)

if __name__ == "__main__":
    main()
