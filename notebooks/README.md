# Notebooks

This folder contains the end-to-end Jupyter notebooks for the cross-platform **sentiment classification** project. All pipeline logic lives here — there are no separate Python scripts.

---

## Available Notebooks

| Notebook | Environment | Description |
|----------|-------------|-------------|
| [`ML_Project_final.ipynb`](ML_Project_final.ipynb) | Google Colab / local | Primary notebook. Uses `/content/sentiment_project` paths. |
| [`ML_Project_kaggle.ipynb`](ML_Project_kaggle.ipynb) | Kaggle | Same 22-stage pipeline with Kaggle path auto-discovery. |

Both notebooks implement the **identical analysis pipeline**; only paths and environment setup differ.

---

## Which Notebook Should I Use?

| Situation | Notebook |
|-----------|----------|
| Running on Kaggle with an attached dataset | `ML_Project_kaggle.ipynb` |
| Running on Google Colab | `ML_Project_final.ipynb` |
| Running locally with Jupyter | `ML_Project_final.ipynb` (adjust base paths — see [`../docs/REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md)) |

---

## Pipeline Stages (22 Total)

| # | Stage | Output |
|---|-------|--------|
| 1 | Dataset loading | Standardized DataFrames |
| 2 | Label alignment | Shared sentiment labels only (-1, 0, 1) |
| 3 | Text cleaning | `text_clean` column |
| 4 | Class balancing | 1000 samples/class/platform |
| 5 | Save processed data | `*_processed.csv` |
| 6 | Domain length analysis | `length_distribution.png` |
| 7 | Train/val/test split | Reddit 80/20, Twitter test |
| 8 | TF-IDF features | Fitted vectorizer |
| 9 | Baseline model training | 4 sklearn classifiers |
| 10 | Leaderboard | `model_leaderboard.csv` |
| 11 | Domain shift plots | `f1_drop_bar.png` |
| 12 | Confusion matrices | In-notebook + saved figures |
| 13 | Classification reports | Per-class metrics |
| 14 | Error analysis | Misclassified Twitter samples |
| 15 | Model persistence | `best_model.pkl`, `vectorizer.pkl` |
| 16 | Per-class tracking | `per_sentiment_performance.csv` |
| 17 | Linguistic features | `*_linguistic.csv` |
| 18 | Compression analysis | `compression_by_sentiment.csv` |
| 19 | Correlation analysis | `compression_vs_transfer.png` |
| 20 | Calibration | `calibration_summary.csv`, curves plot |
| 21 | CORAL adaptation | `coral_results.csv`, `coral_model.pkl` |
| 22 | Risk framing | `clinical_risk_summary.csv` |

Full methodology: [`../docs/METHODOLOGY.md`](../docs/METHODOLOGY.md)

---

## Kaggle-Specific Notes

`ML_Project_kaggle.ipynb` includes:

1. **Markdown header** — dataset requirements and path overview
2. **Auto-discovery cell** — finds `Reddit_Data.csv` and `Twitter_Data.csv` anywhere under `/kaggle/input/`
3. **Working directory** — all outputs go to `/kaggle/working/sentiment_project/`

If file discovery fails, the notebook prints every file under `/kaggle/input/` to help debug dataset attachment issues.

---

## Running Tips

- **Run cells in order.** Later stages depend on variables and files from earlier stages.
- **Restart kernel** if you re-run preprocessing after changing data paths.
- **Internet required** on Kaggle for TextBlob installation (`%pip install textblob`).
- Expect **5–15 minutes** total runtime on CPU.

Setup details: [`../docs/REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md)
