# Visualisations

Generated plots from the cross-platform **sentiment classification** pipeline. All figures are produced by the notebooks and correspond to results in [`../results/`](../results/).

---

## Figure Index

| File | Stage | Description |
|------|-------|-------------|
| `length_distribution.png` | 6 — Domain analysis | Distribution of text length (characters or tokens) for Reddit vs Twitter |
| `f1_drop_bar.png` | 11 — Domain shift | Bar chart comparing Reddit validation F1 vs Twitter test F1 per model |
| `compression_vs_transfer.png` | 19 — Correlation | Relationship between linguistic compression (e.g., TTR diff) and F1 drop per sentiment class |
| `calibration_curves.png` | 20 — Calibration | Reliability diagrams showing predicted probability vs observed frequency per class |
| `clinical_risk_chart.png` | 22 — Risk framing | Visual summary of F1 drop and missed detection rates per sentiment class |

> Local copies may include a ` (1)` suffix from manual re-exports. Fresh notebook runs write canonical filenames.

---

## Regenerating Plots

Run the full notebook pipeline:

- Colab / local: [`../notebooks/ML_Project_final.ipynb`](../notebooks/ML_Project_final.ipynb)
- Kaggle: [`../notebooks/ML_Project_kaggle.ipynb`](../notebooks/ML_Project_kaggle.ipynb)

Output directory:

| Environment | Path |
|-------------|------|
| Colab | `/content/sentiment_project/visualisations/` |
| Kaggle | `/kaggle/working/sentiment_project/visualisations/` |

All plots are saved with `dpi=300` and `bbox_inches='tight'` for publication quality.

---

## Related Documentation

- Results interpretation: [`../docs/RESULTS.md`](../docs/RESULTS.md)
- Methodology (what each plot measures): [`../docs/METHODOLOGY.md`](../docs/METHODOLOGY.md)
- CSV artifacts: [`../results/README.md`](../results/README.md)
