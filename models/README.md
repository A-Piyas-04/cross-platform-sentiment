# Models

This folder stores serialized scikit-learn models and the TF-IDF vectorizer produced by the notebook pipeline. Files are **generated at runtime** and excluded from git (see `.gitignore`).

---

## Expected Artifacts

| File | Type | Description |
|------|------|-------------|
| `best_model.pkl` | Classifier | Best baseline model by Twitter F1 (**Linear SVM**) |
| `vectorizer.pkl` | `TfidfVectorizer` | Fitted on Reddit training data (5000 features, uni+bigrams) |
| `logistic_model.pkl` | Classifier | Logistic Regression saved separately for `predict_proba` / calibration |
| `coral_model.pkl` | Classifier | Logistic Regression trained on CORAL-aligned features |

All files use Python `pickle` format.

---

## How Models Are Selected

**`best_model.pkl`** — chosen by highest weighted F1 on the **Twitter test set** (cross-domain criterion), not Reddit validation alone.

Based on committed results, Linear SVM is the best baseline:

| Model | Twitter F1 |
|-------|------------|
| Linear SVM | 0.653 |
| Logistic Regression | 0.628 |

See [`../docs/RESULTS.md`](../docs/RESULTS.md) for the full leaderboard.

---

## Loading a Saved Model

```python
import pickle

with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Example inference
text = "I am really happy today!"
X = vectorizer.transform([text])
prediction = model.predict(X)
```

**Label mapping:** `-1` = Negative, `0` = Neutral, `1` = Positive

---

## CORAL Model

`coral_model.pkl` was trained on Reddit features aligned to Twitter's covariance structure using CORAL. It underperformed the baseline Logistic Regression on Twitter test (F1: 0.586 vs 0.633) and is kept for experimental comparison only.

---

## Notes

- Models are tied to the specific TF-IDF vocabulary fit on Reddit train data. Always load the paired `vectorizer.pkl`.
- Cross-domain deployment on new platforms will likely require retraining or adaptation.
- For production use, consider migrating to a versioned format (e.g., `joblib` with metadata, MLflow, or ONNX).
