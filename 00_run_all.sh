#!/usr/bin/env bash
# Run the complete Python pipeline (steps 1-6). Step 7 needs the Java results (see java/run.sh).
set -e
cd "$(dirname "$0")"
python 01_simulate_data.py
python 02_quality_control.py
python 03_ld_and_haplotype_blocks.py
python 04_build_features_and_folds.py
python -W ignore 05_run_models_cv.py
python -W ignore 06_summarize_and_plot.py
