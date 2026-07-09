# Results Summary

This document summarizes the main experimental outcomes from the Reddit → Twitter cross-domain sentiment classification pipeline. All values below come from committed artifacts in `results/` and `visualisations/`.

---

## Table of Contents

1. [Baseline Model Leaderboard](#baseline-model-leaderboard)
2. [Best Model Selection](#best-model-selection)
3. [Per-Class Sentiment Performance](#per-class-sentiment-performance)
4. [Domain Shift Visualization](#domain-shift-visualization)
5. [Linguistic Compression](#linguistic-compression)
6. [Calibration Analysis](#calibration-analysis)
7. [CORAL Domain Adaptation](#coral-domain-adaptation)
8. [Clinical Risk Summary](#clinical-risk-summary)
9. [Artifact Index](#artifact-index)

---

## Baseline Model Leaderboard

Models ranked by **Twitter test weighted F1** (primary cross-domain metric).

| Rank | Model | Reddit Val Acc | Reddit Val F1 | Twitter Test Acc | Twitter Test F1 | F1 Drop | Drop % | Train Time (s) |
|------|-------|----------------|---------------|------------------|-----------------|---------|--------|----------------|
| 1 | **Linear SVM** | 0.720 | 0.720 | 0.653 | **0.653** | 0.067 | 9.3% | 0.1 |
| 2 | Logistic Regression | 0.712 | 0.712 | 0.628 | 0.628 | 0.084 | 11.8% | 0.9 |
| 3 | Random Forest | 0.692 | 0.684 | 0.613 | 0.609 | 0.079 | 11.4% | 0.7 |
| 4 | Naive Bayes | 0.602 | 0.597 | 0.543 | 0.537 | 0.059 | 9.8% | 0.0 |

**Source:** `results/model_leaderboard (1).csv`

### Interpretation

- Every model performs **worse on Twitter** than on Reddit validation, confirming domain shift.
- Linear SVM achieves the best cross-domain F1 while also being the fastest to train.
- Naive Bayes has the smallest absolute drop but starts from a much lower baseline.

---

## Best Model Selection

**Selected model:** Linear SVM (highest Twitter F1 = 0.653)

Saved artifacts (when notebook completes):

| File | Description |
|------|-------------|
| `models/best_model.pkl` | Fitted Linear SVM |
| `models/vectorizer.pkl` | Fitted TF-IDF vectorizer |
| `models/logistic_model.pkl` | Logistic Regression (used for probability calibration) |

---

## Per-Class Sentiment Performance

Best-model context (SVM family results; per-class breakdown from evaluation stage):

| Sentiment | Label | Platform | Precision | Recall | F1 |
|-----------|-------|----------|-----------|--------|-----|
| Positive | 1 | Reddit | 0.730 | 0.730 | 0.730 |
| Positive | 1 | Twitter | 0.717 | 0.605 | 0.656 |
| Neutral | 0 | Reddit | 0.748 | 0.755 | 0.751 |
| Neutral | 0 | Twitter | 0.629 | 0.721 | 0.672 |
| Negative | -1 | Reddit | 0.682 | 0.675 | 0.678 |
| Negative | -1 | Twitter | 0.628 | 0.634 | 0.631 |

**Source:** `results/per_sentiment_performance.csv`

### Observations

- **Neutral** transfers best (Twitter F1 = 0.672).
- **Positive** shows the largest recall drop on Twitter (0.730 → 0.605).
- All classes degrade under domain shift; no class is immune.

---

## Domain Shift Visualization

| Plot | File | Description |
|------|------|-------------|
| Validation vs test F1 | `visualisations/f1_drop_bar.png` | Bar chart of performance drop per model |
| Text length distribution | `visualisations/length_distribution.png` | Reddit vs Twitter length comparison |

---

## Linguistic Compression

Platform-level feature differences per sentiment class (selected metrics):

| Sentiment | Δ Type-Token Ratio (Reddit − Twitter) | Δ Avg Word Length |
|-----------|---------------------------------------|-------------------|
| Positive (1) | −0.048 | −0.108 |
| Negative (-1) | −0.025 | −0.162 |
| Neutral (0) | −0.005 | +0.127 |

**Source:** `results/compression_by_sentiment.csv`

Twitter text tends toward **higher type-token ratio** (more unique tokens relative to length) but **shorter average word length** for negative sentiment. The positive class shows the largest TTR compression gap, aligning with its larger F1 drop.

| Plot | File |
|------|------|
| Compression vs transfer | `visualisations/compression_vs_transfer.png` |

---

## Calibration Analysis

Expected Calibration Error (ECE) on Twitter test using Logistic Regression probabilities:

| Sentiment | Label | ECE |
|-----------|-------|-----|
| Negative | -1 | 0.130 |
| Neutral | 0 | 0.111 |
| Positive | 1 | 0.144 |

**Source:** `results/calibration_summary.csv`

Models are **overconfident** on unseen-domain data: predicted probabilities do not match observed accuracy well (ECE > 0.11 for all classes).

| Plot | File |
|------|------|
| Reliability curves | `visualisations/calibration_curves.png` |

---

## CORAL Domain Adaptation

Comparison on Twitter test (Logistic Regression, 3000 TF-IDF features):

| Model | Accuracy | F1 |
|-------|----------|-----|
| Baseline LR | 0.633 | 0.633 |
| CORAL LR | 0.594 | 0.586 |

**Source:** `results/coral_results (1).csv`

**Conclusion:** CORAL alignment **hurt** cross-domain performance (−0.047 F1). Statistical covariance matching does not resolve the semantic/linguistic differences between Reddit and Twitter for this task.

---

## Clinical Risk Summary

Framing negative sentiment as a high-risk detection target:

| Sentiment | Reddit F1 | Twitter F1 | F1 Drop | Missed Rate | Flagged Clinical |
|-----------|-----------|------------|---------|-------------|------------------|
| Neutral | 0.751 | 0.672 | 0.080 | 0.106 | No |
| Positive | 0.730 | 0.656 | 0.074 | 0.101 | No |
| **Negative** | 0.678 | 0.631 | 0.047 | **0.070** | **Yes** |

**Source:** `results/clinical_risk_summary (1).csv`

Although negative sentiment has the smallest F1 drop in absolute terms, it is flagged as clinically relevant because **missed negative cases** in a cross-platform deployment carry the highest operational risk.

| Plot | File |
|------|------|
| Risk chart | `visualisations/clinical_risk_chart.png` |

---

## Artifact Index

| Category | Path | Description |
|----------|------|-------------|
| Leaderboard | `results/model_leaderboard (1).csv` | All baseline models compared |
| Per-class metrics | `results/per_sentiment_performance.csv` | Precision/recall/F1 by sentiment class × platform |
| Compression | `results/compression_by_sentiment.csv` | Linguistic feature diffs per class |
| Calibration | `results/calibration_summary.csv` | ECE per sentiment class |
| CORAL | `results/coral_results (1).csv` | Baseline vs CORAL comparison |
| Risk | `results/clinical_risk_summary (1).csv` | F1 drop + missed rate framing |
| Plots | `visualisations/*.png` | All generated figures |
| Report | `reports/final-report.pdf` | Full written project report |

See also [`../results/README.md`](../results/README.md) for column-level file descriptions.

---

## Key Takeaways

1. Cross-platform deployment incurs a **real, measurable F1 penalty** (~7–12% relative for top models).
2. The penalty is **partly explained** by linguistic compression differences between platforms.
3. **Probability outputs should not be trusted** cross-domain without recalibration.
4. **Simple domain adaptation (CORAL) is insufficient**; semantic methods (embeddings) are the logical next step.
5. **Negative sentiment detection** deserves special attention in any production cross-platform system.
