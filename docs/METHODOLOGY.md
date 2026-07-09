# Methodology

This document describes the experimental design, preprocessing pipeline, modeling choices, and analysis stages used in the cross-platform sentiment classification project.

---

## Table of Contents

1. [Problem Formulation](#problem-formulation)
2. [Datasets](#datasets)
3. [Preprocessing Pipeline](#preprocessing-pipeline)
4. [Train / Validation / Test Protocol](#train--validation--test-protocol)
5. [Feature Engineering](#feature-engineering)
6. [Baseline Models](#baseline-models)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Diagnostic Analyses](#diagnostic-analyses)
9. [Domain Adaptation (CORAL)](#domain-adaptation-coral)
10. [Risk Framing](#risk-framing)
11. [Reproducibility Controls](#reproducibility-controls)

---

## Problem Formulation

**Task:** Ternary sentiment classification.

**Domain shift setup:**

- **Source domain:** Reddit (labeled data available for training)
- **Target domain:** Twitter (used only for evaluation; no target labels used during training)

This mirrors a realistic deployment scenario: train on one platform, deploy on another.

---

## Datasets

| Platform | Raw file | Text column | Label column |
|----------|----------|-------------|--------------|
| Reddit | `Reddit_Data.csv` | `clean_comment` | `category` |
| Twitter | `Twitter_Data.csv` | `clean_text` | `category` |

After label intersection and balancing, each platform contributes **3,000 samples** (1,000 per class).

See [`../data/README.md`](../data/README.md) for schema details.

---

## Preprocessing Pipeline

### Stage 1 — Load and Standardize

```python
reddit.rename(columns={'clean_comment': 'text', 'category': 'sentiment'})
twitter.rename(columns={'clean_text': 'text', 'category': 'sentiment'})
```

### Stage 2 — Label Cleaning

- Drop rows with missing `sentiment`
- Cast labels to integer

### Stage 3 — Label Alignment

Keep only sentiment labels present in **both** platforms:

```python
common = set(reddit['sentiment']).intersection(set(twitter['sentiment']))
```

### Stage 4 — Text Cleaning (`clean_text`)

Applied to create `text_clean`:

1. Lowercase
2. Remove URLs (`http...`, `www...`)
3. Remove `@mentions`
4. Strip `#` from hashtags (keep the word)
5. Collapse repeated whitespace
6. Strip leading/trailing space

### Stage 5 — Length Filter

Remove samples where `len(text_clean) <= 10`.

### Stage 6 — Class Balancing

For each platform independently:

- Group by `sentiment`
- Sample up to **1,000** rows per class
- Use `random_state=42`

Outputs: `reddit_processed.csv`, `twitter_processed.csv`

---

## Train / Validation / Test Protocol

```
Reddit (3,000 balanced)
├── Train: 80% stratified → 2,400 samples
└── Val:   20% stratified → 600 samples

Twitter (3,000 balanced)
└── Test: 100% → 3,000 samples (never seen during training)
```

**Critical constraint:** TF-IDF is **fit on Reddit training data only** to prevent leakage.

Split call:

```python
train_test_split(..., test_size=0.2, stratify=sentiment, random_state=42)
```

---

## Feature Engineering

### TF-IDF (Baseline Models)

| Parameter | Value |
|-----------|-------|
| `max_features` | 5000 |
| `ngram_range` | (1, 2) — unigrams + bigrams |
| `min_df` | 3 |
| `max_df` | 0.8 |
| `sublinear_tf` | True |

Fit on Reddit train; transform train, val, and Twitter test.

### TF-IDF (CORAL Section)

Same settings except `max_features=3000` and conversion to dense arrays for covariance alignment.

---

## Baseline Models

All models use scikit-learn implementations.

| Model | Key hyperparameters |
|-------|---------------------|
| Logistic Regression | `max_iter=1000`, `class_weight='balanced'` |
| Linear SVM | `max_iter=2000`, `class_weight='balanced'` |
| Multinomial Naive Bayes | `alpha=0.1` |
| Random Forest | `n_estimators=100`, `class_weight='balanced'` |

**Model selection criterion:** Highest **weighted F1 on Twitter test** (cross-domain performance).

---

## Evaluation Metrics

| Metric | Purpose |
|--------|---------|
| Accuracy | Overall correctness |
| Weighted F1 | Primary ranking metric (handles class imbalance weighting) |
| Acc / F1 Drop | Reddit val → Twitter test degradation |
| Per-class precision, recall, F1 | Class-level failure analysis |
| Confusion matrix | Misclassification patterns |
| ECE (Expected Calibration Error) | Probability calibration on Twitter |
| Missed rate | Fraction of class samples incorrectly classified (risk analysis) |

---

## Diagnostic Analyses

### A. Linguistic Compression

Extract 10 linguistic features per sample (see [`../data/README.md`](../data/README.md)), then compare Reddit vs Twitter means per sentiment class.

**Compression hypothesis:** Larger Reddit→Twitter structural differences (e.g., type-token ratio) correlate with larger F1 drops per sentiment class.

Outputs:

- `results/compression_by_sentiment.csv`
- `visualisations/compression_vs_transfer.png`
- `visualisations/f1_drop_bar.png`

### B. Calibration Analysis

Use Logistic Regression `predict_proba` on Twitter test.

- Reliability curves per class
- ECE per sentiment class

Outputs:

- `results/calibration_summary.csv`
- `visualisations/calibration_curves.png`

### C. Error Analysis

Extract misclassified Twitter samples from the best model for qualitative inspection of domain-shift failures.

### D. Length Distribution

Compare text length distributions between platforms before modeling.

Output: `visualisations/length_distribution.png`

---

## Domain Adaptation (CORAL)

**CORAL** (CORrelation ALignment) aligns second-order statistics between source and target feature distributions.

Procedure:

1. TF-IDF transform Reddit train and Twitter test (3000 features)
2. Convert sparse matrices to dense
3. Apply CORAL alignment: shift Reddit train covariance toward Twitter test covariance
4. Train Logistic Regression on aligned Reddit features
5. Evaluate on Twitter test (unaligned target features, aligned source training)

**Result:** CORAL **reduced** performance vs baseline LR, indicating that covariance alignment alone is insufficient for this semantic shift.

Output: `results/coral_results.csv`, `models/coral_model.pkl`

---

## Risk Framing

Negative sentiment (`-1`) is treated as a **high-risk / clinically relevant** class.

For each sentiment class:

- F1 on Reddit vs Twitter
- F1 drop under domain shift
- Missed rate on Twitter (false negatives / total class samples)

Output: `results/clinical_risk_summary.csv`, `visualisations/clinical_risk_chart.png`

---

## Reproducibility Controls

| Control | Value |
|---------|-------|
| Balancing seed | `random_state=42` |
| Train/val split seed | `random_state=42` |
| TF-IDF fit scope | Reddit train only |
| Class cap | 1000 per sentiment class per platform |

Full setup instructions: [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)

---

## Pipeline Stage Map

The notebooks implement 22 documented stages:

| Phase | Stages | Description |
|-------|--------|-------------|
| Data | 1–6 | Load, align, clean, balance, save, length analysis |
| Experiment setup | 7 | Reddit split, Twitter as test |
| Modeling | 8–9 | TF-IDF + train 4 baselines |
| Evaluation | 10–14 | Leaderboard, plots, confusion, errors |
| Persistence | 15–16 | Save best model, per-class CSV |
| Diagnostics | 17–20 | Linguistic features, compression, calibration |
| Adaptation | 21 | CORAL experiment |
| Risk | 22 | Clinical / missed-case framing |

Implementation: [`../notebooks/ML_Project_final.ipynb`](../notebooks/ML_Project_final.ipynb) or [`../notebooks/ML_Project_kaggle.ipynb`](../notebooks/ML_Project_kaggle.ipynb)
