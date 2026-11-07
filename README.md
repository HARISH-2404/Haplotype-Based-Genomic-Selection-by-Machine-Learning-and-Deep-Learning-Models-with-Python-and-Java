# Haplotype-based Genomic Selection with Machine Learning and Deep Learning (Python + Java)

End-to-end, reproducible pipeline for cotton (*G. hirsutum*) genomic selection: simulated SNP data → QC → LD haplotype blocks →
haplotype features → classical / ML / DL models → cross-validation → figures. Implemented in **Python** and **Java**.

## Quick start
```bash
pip install -r requirements.txt
cd python/scripts && ./00_run_all.sh          # steps 1-6 (~25 min on 1 CPU)
cd ../../java && ./test.sh && ./run.sh        # Java pipeline (after Python steps 1-4)
python ../python/scripts/07_compare_java_python.py
```

## Layout
```
data/raw, data/processed, data/truth   simulated inputs, processed matrices, simulation truth
python/src/hgs, python/scripts, tests  package, numbered step scripts, unit/gradient tests
java/src/main/java/org/hgs             pure-Java QC, LD blocks, haplotypes, RR-BLUP, MLP, CV
docs/                                  step-by-step documentation (00-08)
results/tables, results/figures        all outputs;  results/RESULTS.md = written summary
```

## Headline results (simulated data, 300 lines)
* LD-block detection recovers true boundaries (F1 0.71) far better than fixed windows (0.30).
* Haplotype features improve random-CV accuracy for ridge/BayesRidge/LASSO/RF/boosting (best: gradient boosting 0.493 → 0.539; RR-BLUP 0.491 → 0.508).
* Neural networks (n=300) underperform ridge and boosting; across-family prediction is much harder (r ≈ 0.25-0.31).
* Java RR-BLUP reproduces Python (fold-level r = 0.9995). Details and caveats: `results/RESULTS.md`.

Data are **simulated**; replace `data/raw` with real data following `docs/08_real_data_guide.md`. MIT licence.
