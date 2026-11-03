import re
import html
import pandas as pd


def clean_review(text):

    if pd.isna(text):
        return ""

    text = str(text)

    # Convert HTML entities
    text = html.unescape(text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Convert text to lowercase
    text = text.lower()

    # Remove unwanted symbols while preserving apostrophes
    text = re.sub(r"[^a-z0-9\s']", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_texts(texts):
    """
    Preprocess a collection of customer reviews.

    Supports lists, Pandas Series and NumPy arrays.
    """

    if isinstance(texts, str):
        texts = [texts]

    texts = pd.Series(texts)

    return texts.apply(clean_review)