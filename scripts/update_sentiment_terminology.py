"""Update emotion -> sentiment terminology across notebooks and CSV artifacts."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def transform_text(text: str) -> str:
    """Context-aware terminology updates."""
    replacements = [
        # Project paths
        ("/content/emotion_project", "/content/sentiment_project"),
        ("/kaggle/working/emotion_project", "/kaggle/working/sentiment_project"),
        ("./emotion_project", "./sentiment_project"),
        # Titles and headings
        ("Cross-Platform Emotion Classification", "Cross-Platform Sentiment Classification"),
        ("emotion/sentiment classifiers", "sentiment classifiers"),
        ("emotion classifiers", "sentiment classifiers"),
        # Output filenames
        ("per_emotion_performance.csv", "per_sentiment_performance.csv"),
        ("compression_by_emotion.csv", "compression_by_sentiment.csv"),
        # Column / dict keys in code
        ("'category': 'emotion'", "'category': 'sentiment'"),
        ("'emotion': label", "'sentiment': label"),
        ("Emotion_Label", "Sentiment_Label"),
        ("Emotion_Name", "Sentiment_Name"),
        # Variable names
        ("per_emotion_df", "per_sentiment_df"),
        ("reddit_emotions", "reddit_labels"),
        ("twitter_emotions", "twitter_labels"),
        ("common_emotions", "common_labels"),
        # Column references
        ("subset=['emotion']", "subset=['sentiment']"),
        ("['emotion']", "['sentiment']"),
        ("['emotion',", "['sentiment',"),
        ("groupby('emotion')", "groupby('sentiment')"),
        ("reddit['emotion']", "reddit['sentiment']"),
        ("twitter['emotion']", "twitter['sentiment']"),
        ("reddit_balanced['emotion']", "reddit_balanced['sentiment']"),
        ("twitter_balanced['emotion']", "twitter_balanced['sentiment']"),
        (".set_index('emotion')", ".set_index('sentiment')"),
        ("risk_df['Emotion_Name']", "risk_df['Sentiment_Name']"),
        # Comments and print strings
        ("Standardized column names (text, emotion).", "Standardized column names (text, sentiment)."),
        ("Remove missing emotion values", "Remove rows with missing sentiment labels"),
        ("Find intersection of emotions", "Find intersection of sentiment labels shared by both platforms"),
        ('print(f"\\nReddit emotions: {reddit_labels}")', 'print(f"\\nReddit sentiment labels: {reddit_labels}")'),
        ('print(f"Twitter emotions: {twitter_labels}")', 'print(f"Twitter sentiment labels: {twitter_labels}")'),
        ('print(f"Common emotions: {common_labels}")', 'print(f"Shared sentiment labels: {common_labels}")'),
        ("Filter to common emotions", "Keep only rows with shared sentiment labels"),
        ("Collect per-emotion metrics", "Collect per-class sentiment metrics"),
        ("Load per-emotion data", "Load per-class sentiment metrics"),
        ("finer emotion labels", "finer-grained sentiment label schemes"),
        ("for emotion, row in plot_df.iterrows():", "for sentiment_label, row in plot_df.iterrows():"),
        ("label_map.get(emotion, str(emotion))", "label_map.get(sentiment_label, str(sentiment_label))"),
        ("for i, emotion in enumerate(classes):", "for i, sentiment_label in enumerate(classes):"),
        ('label=f"{emotion} (ECE={ece:.2f})"', 'label=f"{sentiment_label} (ECE={ece:.2f})"'),
        ("'emotion': emotion,", "'sentiment': sentiment_label,"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)

    # Fix any remaining unique() calls that used old column name
    text = text.replace("reddit['sentiment'].unique())", "reddit['sentiment'].unique())")
    text = re.sub(r"\bper_em\b", "per_class_df", text)

    return text


def update_notebook(path: Path) -> None:
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code" and "source" in cell:
            src = "".join(cell["source"])
            cell["source"] = transform_text(src).splitlines(keepends=True)
        elif cell.get("cell_type") == "markdown" and "source" in cell:
            src = "".join(cell["source"])
            cell["source"] = transform_text(src).splitlines(keepends=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"Updated notebook: {path.name}")


def update_csv_header(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if text.startswith(old):
        path.write_text(new + text[len(old):], encoding="utf-8")
        print(f"Updated CSV header: {path.name}")


def rename_csv(old_name: str, new_name: str) -> None:
    old = ROOT / "results" / old_name
    new = ROOT / "results" / new_name
    if old.exists() and not new.exists():
        old.rename(new)
        print(f"Renamed: {old_name} -> {new_name}")


if __name__ == "__main__":
    for nb in (ROOT / "notebooks").glob("*.ipynb"):
        update_notebook(nb)

    rename_csv("per_emotion_performance.csv", "per_sentiment_performance.csv")
    rename_csv("compression_by_emotion.csv", "compression_by_sentiment.csv")

    results = ROOT / "results"
    if (results / "per_sentiment_performance.csv").exists():
        update_csv_header(results / "per_sentiment_performance.csv", "emotion,", "sentiment,")
    if (results / "compression_by_sentiment.csv").exists():
        update_csv_header(results / "compression_by_sentiment.csv", "emotion,", "sentiment,")
    if (results / "calibration_summary.csv").exists():
        update_csv_header(results / "calibration_summary.csv", "emotion,", "sentiment,")

    clinical_files = list(results.glob("clinical_risk_summary*.csv"))
    for cf in clinical_files:
        text = cf.read_text(encoding="utf-8")
        text = text.replace("Emotion_Label", "Sentiment_Label").replace("Emotion_Name", "Sentiment_Name")
        cf.write_text(text, encoding="utf-8")
        print(f"Updated clinical summary: {cf.name}")

    for name in ["reddit_processed.csv", "twitter_processed.csv"]:
        p = ROOT / "data" / "processed" / name
        if p.exists():
            update_csv_header(p, "text,emotion,", "text,sentiment,")
    for name in ["reddit_linguistic.csv", "twitter_linguistic.csv"]:
        p = ROOT / "data" / "processed" / name
        if p.exists():
            update_csv_header(p, "emotion,", "sentiment,")

    print("Done.")
