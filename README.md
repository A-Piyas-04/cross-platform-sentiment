# Cross-Platform Sentiment Classification (Reddit → Twitter)

A research-oriented machine learning project that studies **cross-domain generalization** for ternary sentiment classification (negative / neutral / positive): models are trained on Reddit text and evaluated on Twitter text without target-domain labels. The work combines classical NLP baselines, diagnostic analysis of domain shift, and a domain-adaptation experiment (CORAL).

---

## Table of Contents

- [Motivation](#motivation)
- [Research Question](#research-question)
- [Key Findings](#key-findings)
- [Pipeline at a Glance](#pipeline-at-a-glance)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Documentation](#documentation)
- [Tech Stack](#tech-stack)
- [Future Work](#future-work)

---

## Motivation

Social media language is not uniform across platforms. Reddit comments tend to be longer and more expressive; tweets are shorter and more compressed. A classifier that performs well on one platform may fail on another—not because the task changed, but because the **distribution of language** changed.

This project quantifies that gap, explains part of it through **linguistic compression**, and tests whether a lightweight adaptation method (CORAL) can close it.

---

## Research Question

> How well do sentiment classifiers trained on Reddit generalize to Twitter, and can statistical domain adaptation reduce the performance drop?

**Experimental setup**

| Role | Platform | Split | Size (after balancing) |
|------|----------|-------|-------------------------|
| Source (train + validation) | Reddit | 80% train / 20% val | 2,400 / 600 |
| Target (test only) | Twitter | Full balanced set | 3,000 |

Labels are ternary sentiment scores shared across both datasets:

| Label | Meaning |
|-------|---------|
| `-1` | Negative |
| `0` | Neutral |
| `1` | Positive |

---

## Key Findings

1. **Domain shift is real.** The best baseline (Linear SVM) drops from **0.72 F1** on Reddit validation to **0.653 F1** on Twitter test (~9.3% relative drop).
2. **Compression correlates with transfer loss.** Sentiment classes whose wording is more compressed on Twitter tend to see larger F1 drops; type-token ratio differences help explain per-class degradation.
3. **Models are overconfident cross-domain.** Expected Calibration Error (ECE) on Twitter ranges from **0.11–0.14** across classes.
4. **CORAL did not help.** Covariance alignment reduced Twitter F1 from **0.633 → 0.586** for Logistic Regression, suggesting the mismatch is not purely distributional.
5. **Negative sentiment carries operational risk.** When framed as a high-risk class, negative sentiment shows the largest missed-detection concern under domain shift.

See [`docs/RESULTS.md`](docs/RESULTS.md) for full metrics and artifact references.

---

## Pipeline at a Glance

```
Raw CSVs (Reddit + Twitter)
        │
        ▼
┌───────────────────────────┐
│  Preprocessing            │  label alignment, text cleaning, class balancing
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│  TF-IDF + Baseline Models │  LR, SVM, Naive Bayes, Random Forest
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│  Cross-Domain Evaluation  │  leaderboard, confusion matrices, error analysis
└───────────────────────────┘
        │
        ├──► Linguistic compression analysis
        ├──► Calibration analysis (ECE)
        ├──► CORAL domain adaptation
        └──► Clinical / risk framing (negative class)
```

Detailed stage-by-stage documentation: [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md)

---

## Repository Structure

```
emotion-classification-cross-platform/   # repo name; classifies sentiment (-1 / 0 / 1)
├── notebooks/
│   ├── ML_Project_final.ipynb    # Full pipeline (Colab / local)
│   └── ML_Project_kaggle.ipynb   # Same pipeline for Kaggle
├── data/
│   ├── raw/                      # Source CSVs (not committed)
│   └── processed/                # Generated balanced + linguistic tables
├── models/                       # Saved .pkl artifacts (generated, gitignored)
├── results/                      # Evaluation CSVs
├── visualisations/               # Generated plots
├── reports/
│   └── final-report.pdf          # Written project report
├── docs/                         # Extended documentation
│   ├── METHODOLOGY.md
│   ├── RESULTS.md
│   └── REPRODUCIBILITY.md
├── requirements.txt
└── README.md
```

Each major folder has its own README with specifics:

| Path | Description |
|------|-------------|
| [`data/README.md`](data/README.md) | Data layout, schema, acquisition |
| [`notebooks/README.md`](notebooks/README.md) | Which notebook to run and where |
| [`models/README.md`](models/README.md) | Serialized model artifacts |
| [`results/README.md`](results/README.md) | Output tables and metrics files |
| [`reports/README.md`](reports/README.md) | Final written report |

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/emotion-classification-cross-platform.git
cd emotion-classification-cross-platform
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add raw data

Place the source files in `data/raw/`:

```
data/raw/Reddit_Data.csv
data/raw/Twitter_Data.csv
```

See [`data/README.md`](data/README.md) for column requirements.

### 3. Run the notebook

**Local / Google Colab**

```bash
jupyter notebook notebooks/ML_Project_final.ipynb
```

**Kaggle**

Upload [`notebooks/ML_Project_kaggle.ipynb`](notebooks/ML_Project_kaggle.ipynb), attach your dataset containing both CSV files, and run all cells. The notebook auto-discovers files under `/kaggle/input/`.

Full environment instructions: [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)

---

## Documentation

| Document | Contents |
|----------|----------|
| [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) | Experimental design, preprocessing, features, models, analysis stages |
| [`docs/RESULTS.md`](docs/RESULTS.md) | Leaderboard, per-class metrics, CORAL comparison, calibration, risk summary |
| [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) | Local, Colab, and Kaggle setup; seeds; expected outputs |

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10+ |
| Data | pandas, NumPy |
| ML | scikit-learn |
| NLP | NLTK (VADER), TextBlob |
| Visualization | matplotlib |
| Statistics | SciPy |
| Environment | Jupyter Notebook |

---

## Future Work

- Replace TF-IDF with contextual embeddings (BERT, RoBERTa) to capture semantic shift.
- Explore adversarial or neural domain-adaptation beyond CORAL.
- Extend to additional platforms and finer-grained sentiment schemes (e.g., star ratings, aspect-based sentiment).
- Add a proper Python package (`src/`) and inference CLI for reproducible deployment.

---

## License & Data

Raw datasets are **not redistributed** in this repository. Download them separately and follow the original dataset licenses. See [`data/README.md`](data/README.md) for file requirements and placement.
