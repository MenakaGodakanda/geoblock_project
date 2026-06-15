# geoblock_project
A Blockchain-Anchored Smart Contract and GIS Framework for Resilient Disaster Resource Coordination in Smart Cities


## Quick-Start on  Ubuntu Machine
### 1. Create virtual environment
```
python3 -m venv geoblock_env
source geoblock_env/bin/activate
```

### 2. Install dependencies
```
cd geoblock_project
pip install -r requirements.txt
```

### 3. Generate data (offline) OR download real FEMA data
```
python3 scripts/01_download_data.py
```

### 4. Run full pipeline
```
python3 scripts/08_run_pipeline.py --skip-download
```

### 5. Open the HTML report in your browser
```
xdg-open outputs/reports/final_summary_report.html
```
