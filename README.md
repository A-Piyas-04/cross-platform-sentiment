# Cross-Platform Emotion Classification (Reddit vs. Twitter)

A machine learning project studying how well emotion/sentiment classifiers trained on one
social media platform (Reddit) generalize to another (Twitter) — and what happens when you
try to fix the gap with domain adaptation.

## Overview

Social media text differs a lot by platform: Reddit comments tend to be longer and more
expressive, while tweets are short and compressed. This project trains classical ML models
on Reddit data, evaluates them on Twitter data, and investigates *why* performance drops
across the domain shift.

**Pipeline:**
1. **Dataset loading & label alignment** — load Reddit and Twitter sentiment datasets,
   standardize columns, and keep only emotion labels common to both platforms.
2. **Text cleaning** — lowercase, strip URLs/@mentions, normalize hashtags, drop very
   short texts.
3. **Class balancing** — balance both datasets by emotion class.
4. **Feature extraction** — TF-IDF vectorization (unigrams + bigrams, 5000 features).
5. **Model training** — Logistic Regression, Linear SVM, Naive Bayes, Random Forest,
   compared on a held-out Reddit validation set *and* on the Twitter test set.
6. **Cross-domain evaluation** — leaderboard sorted by Twitter F1, confusion matrices,
   per-emotion breakdown, and error analysis on misclassified samples.
7. **Linguistic compression analysis** — type-token ratio and related text statistics to
   quantify how much "shorter/less expressive" Twitter text is, and correlate that with
   the F1 drop per emotion.
8. **Domain adaptation attempt (CORAL)** — tried statistical feature alignment between
   Reddit and Twitter distributions to close the gap.

## Key Findings

- Models trained on Reddit show a clear **performance drop when evaluated on Twitter**,
  confirming the domain shift is real, not just noise.
- The drop correlates with **linguistic compression**: emotions that are expressed more
  tersely on Twitter than Reddit see a bigger F1 decline.
- **CORAL (statistical domain alignment) did not help** — performance decreased, suggesting
  the mismatch isn't just a distribution shift but also a semantic/linguistic one that
  simple statistical alignment can't fix.
- **Future direction:** semantic embedding models (e.g. BERT) are likely needed to capture
  meaning-level shifts that TF-IDF + statistical alignment can't.

## Repo Structure

```
├── notebooks/
│   └── ML_Project_final.ipynb   # full analysis, end to end
├── data/                        # not included — see data/README.md
├── models/                      # trained models saved here when run (gitignored)
├── results/                     # leaderboards, per-emotion metrics (gitignored)
├── visualisations/              # generated plots (gitignored)
├── requirements.txt
└── README.md
```

## Getting Started

```bash
git clone https://github.com/<your-username>/emotion-classification-cross-platform.git
cd emotion-classification-cross-platform
pip install -r requirements.txt
jupyter notebook notebooks/ML_Project_final.ipynb
```

You'll need to download the datasets first — see [`data/README.md`](data/README.md).

## Data

This project uses public Reddit and Twitter sentiment datasets (see `data/README.md` for
sources). Raw data files are not committed to this repo; download them and place them in
`data/` before running the notebook.

## Tech Stack

Python, pandas, scikit-learn, NLTK, TextBlob, matplotlib, SciPy.

## Future Work

- Replace TF-IDF with contextual embeddings (BERT/RoBERTa) to capture semantic shift.
- Explore adversarial or neural domain adaptation methods beyond CORAL.
- Extend to more platforms and finer-grained emotion taxonomies.
