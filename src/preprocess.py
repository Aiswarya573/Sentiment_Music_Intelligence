import pandas as pd
import re


# ─────────────────────────────────────────────
#  Text Cleaning Utilities
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Full preprocessing pipeline:
      1. Lowercase
      2. Remove URLs
      3. Remove @mentions and #hashtags
      4. Remove punctuation / special chars
      5. Collapse extra whitespace
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)          # URLs
    text = re.sub(r"@\w+", "", text)                    # mentions
    text = re.sub(r"#\w+", "", text)                    # hashtags
    text = re.sub(r"[^a-z\s']", " ", text)              # keep letters + apostrophes
    text = re.sub(r"\s+", " ", text).strip()            # extra spaces
    return text


def tokenize(text: str) -> list:
    """Simple whitespace tokenizer after cleaning."""
    return clean_text(text).split()


def map_sentiment_label(raw_label) -> str:
    """
    Sentiment140 uses 0 = negative, 2 = neutral, 4 = positive.
    Map to human-readable strings.
    """
    mapping = {0: "negative", 2: "neutral", 4: "positive"}
    try:
        return mapping.get(int(raw_label), "neutral")
    except (ValueError, TypeError):
        return "neutral"


# ─────────────────────────────────────────────
#  Dataset Loader
# ─────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """
    Load Sentiment140 CSV, clean text, and map labels.

    Returns a DataFrame with columns: ['sentiment', 'text', 'clean_text']
    """
    try:
        df = pd.read_csv(path, encoding="latin-1", header=None)

        df.columns = ["sentiment", "id", "date", "query", "user", "text"]
        df = df[["sentiment", "text"]].copy()

        # Map numeric labels → strings
        df["sentiment"] = df["sentiment"].apply(map_sentiment_label)

        # Clean text
        df["clean_text"] = df["text"].apply(clean_text)

        # Drop empty rows after cleaning
        df = df[df["clean_text"].str.strip() != ""].reset_index(drop=True)

        print(f"[Preprocess] Loaded {len(df):,} rows from '{path}'")
        print(f"[Preprocess] Sentiment distribution:\n{df['sentiment'].value_counts().to_string()}\n")

        return df

    except FileNotFoundError:
        print(f"[Preprocess] ERROR: Dataset not found at '{path}'. Check the path.")
        return pd.DataFrame(columns=["sentiment", "text", "clean_text"])
