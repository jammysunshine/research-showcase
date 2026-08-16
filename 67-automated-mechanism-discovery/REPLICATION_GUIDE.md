# Replication guide

From the repository root, run:

```bash
python3 -m unittest discover -s tests -v
python3 scripts/run_public_project_study.py
python3 scripts/verify_public_project_certificate.py
```

Inspect `artifacts/public_project_study.json` for every rule, witness, metric, and cost row; inspect `artifacts/public_project_scaling.csv` for the n=3..5 extension; inspect `artifacts/public_project_certificate.json` for the independent digests and held-out failures. The original Experiment 67 regression commands remain in `README.md`.
