# Reproducibility Guide

Instructions for reproducing the full cross-platform sentiment classification pipeline across local, Google Colab, and Kaggle environments.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Setup](#local-setup)
3. [Google Colab Setup](#google-colab-setup)
4. [Kaggle Setup](#kaggle-setup)
5. [NLTK and TextBlob Data](#nltk-and-textblob-data)
6. [Expected Outputs](#expected-outputs)
7. [Random Seeds and Determinism](#random-seeds-and-determinism)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.10+ recommended |
| RAM | ≥ 4 GB (8 GB recommended for linguistic feature extraction) |
| Disk | ~500 MB for raw + processed data and outputs |

### Install Dependencies

```bash
git clone https://github.com/<your-username>/emotion-classification-cross-platform.git
cd emotion-classification-cross-platform
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate          # macOS / Linux
.venv\Scripts\activate             # Windows

pip install -r requirements.txt
```

### Additional Runtime Packages

The notebook also uses `tqdm` (progress bars). Install if missing:

```bash
pip install tqdm
```

---

## Local Setup

### 1. Prepare data

```text
data/raw/Reddit_Data.csv
data/raw/Twitter_Data.csv
```

See [`../data/README.md`](../data/README.md) for schema requirements.

### 2. Launch Jupyter

```bash
jupyter notebook notebooks/ML_Project_final.ipynb
```

### 3. Adjust paths (if needed)

The Colab-oriented notebook uses `/content/sentiment_project` as its base directory. For local runs, either:

- Run in Google Colab as-is, or
- Modify the first path-setup cell to use a local base, e.g.:

```python
BASE = './sentiment_project'   # or an absolute path
```

Create the folder structure:

```python
import os
for sub in ['data', 'models', 'results', 'visualisations']:
    os.makedirs(f'{BASE}/{sub}', exist_ok=True)
```

### 4. Point to raw CSVs

Ensure the data-loading cell reads from `data/raw/`:

```python
reddit = pd.read_csv('data/raw/Reddit_Data.csv')
twitter = pd.read_csv('data/raw/Twitter_Data.csv')
```

### 5. Run all cells

Runtime: approximately **5–15 minutes** depending on hardware (linguistic feature extraction is the slowest stage).

---

## Google Colab Setup

1. Upload [`../notebooks/ML_Project_final.ipynb`](../notebooks/ML_Project_final.ipynb) to Colab.
2. Upload `Reddit_Data.csv` and `Twitter_Data.csv` to the Colab session (or mount Google Drive).
3. Run the folder-setup cell — it creates `/content/sentiment_project/`.
4. Place raw CSVs in the working directory or update paths in the loading cell.
5. Run all cells sequentially.

**Output location:** `/content/sentiment_project/`

Download artifacts from the Colab file browser when complete.

---

## Kaggle Setup

Use the Kaggle-adapted notebook: [`../notebooks/ML_Project_kaggle.ipynb`](../notebooks/ML_Project_kaggle.ipynb)

### Steps

1. **Create a Kaggle dataset** containing:
   - `Reddit_Data.csv`
   - `Twitter_Data.csv`

2. **Create a new Kaggle notebook** and upload `ML_Project_kaggle.ipynb`.

3. **Attach the dataset:** Notebook → **Add Data** → select your dataset.

4. **Enable internet** (Settings → Internet: On) for `%pip install textblob`.

5. **Run all cells** from top to bottom.

### Path auto-discovery

The first code cell searches `/kaggle/input/` recursively for the required CSV filenames. Kaggle dataset slugs often differ from sidebar folder names — auto-discovery handles this.

Expected startup output:

```text
Kaggle paths ready!
  Reddit:  /kaggle/input/<dataset-slug>/Reddit_Data.csv
  Twitter: /kaggle/input/<dataset-slug>/Twitter_Data.csv
  Output:  /kaggle/working/sentiment_project
```

**Output location:** `/kaggle/working/sentiment_project/`

---

## NLTK and TextBlob Data

The linguistic analysis stage requires additional corpora. The notebook installs these automatically, but you can pre-download locally:

```python
import nltk
nltk.download('punkt')
nltk.download('brown')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')
```

```bash
python -m textblob.download_corpora
```

NLTK data is gitignored (`nltk_data/`) and regenerated per environment.

---

## Expected Outputs

After a successful run, you should have:

### Processed Data (`data/` or `{BASE}/data/`)

| File | Rows |
|------|------|
| `reddit_processed.csv` | 3,000 |
| `twitter_processed.csv` | 3,000 |
| `reddit_linguistic.csv` | 3,000 |
| `twitter_linguistic.csv` | 3,000 |

### Models (`models/`)

| File | Description |
|------|-------------|
| `best_model.pkl` | Best baseline classifier (Linear SVM) |
| `vectorizer.pkl` | Fitted TF-IDF vectorizer |
| `logistic_model.pkl` | Logistic Regression for calibration |
| `coral_model.pkl` | CORAL-adapted Logistic Regression |

### Results (`results/`)

| File | Description |
|------|-------------|
| `model_leaderboard.csv` | Model comparison table |
| `per_sentiment_performance.csv` | Per-class sentiment metrics |
| `compression_by_sentiment.csv` | Linguistic compression stats |
| `calibration_summary.csv` | ECE per class |
| `coral_results.csv` | CORAL vs baseline |
| `clinical_risk_summary.csv` | Risk framing table |

### Visualisations (`visualisations/`)

| File | Description |
|------|-------------|
| `length_distribution.png` | Text length by platform |
| `f1_drop_bar.png` | Domain shift bar chart |
| `compression_vs_transfer.png` | Compression vs F1 drop |
| `calibration_curves.png` | Reliability diagrams |
| `clinical_risk_chart.png` | Risk summary chart |

---

## Random Seeds and Determinism

| Step | Seed / Control |
|------|----------------|
| Class balancing | `random_state=42` |
| Train/validation split | `random_state=42`, stratified |
| TF-IDF vocabulary | Deterministic given train set |
| Model training | scikit-learn default random states |

Exact floating-point results may vary slightly across scikit-learn versions or BLAS backends, but rankings and conclusions should remain stable.

---

## Troubleshooting

### `FileNotFoundError` for CSV files (Kaggle)

- Confirm the dataset is attached via **Add Data**.
- Check the first cell output — it lists all files under `/kaggle/input/`.
- Ensure filenames are exactly `Reddit_Data.csv` and `Twitter_Data.csv`.

### NLTK `LookupError`

Run the NLTK download cell or manually download corpora (see above).

### TextBlob sentiment errors

```bash
python -m textblob.download_corpora
```

### Out of memory during linguistic extraction

- Reduce batch size or run on a machine with more RAM.
- On Kaggle, ensure GPU is not required (this pipeline is CPU-only).

### Different results vs committed artifacts

- Verify scikit-learn version.
- Confirm raw data files match the original source.
- Ensure all cells ran sequentially without skipped preprocessing steps.

---

## Related Documentation

- Methodology: [`METHODOLOGY.md`](METHODOLOGY.md)
- Results reference: [`RESULTS.md`](RESULTS.md)
- Data schema: [`../data/README.md`](../data/README.md)
