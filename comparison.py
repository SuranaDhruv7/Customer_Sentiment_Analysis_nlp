"""
Compare BoW, TF-IDF, and Word2Vec representations for sentiment classification.

"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from NLP.functions.text_preprocessing import clean_review
from NLP.functions.bow import create_bow_vectorizer
from NLP.functions.tf_idf import create_tfidf_vectorizer


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

BASE_DIR = Path.cwd()
DATA_PATH = BASE_DIR / "data" / "raw" / "customer_reviews_sentiment_practice.csv"
MODEL_PATH = BASE_DIR / "models" / "trained_word2vec_cbow.model"
RESULTS_PATH = BASE_DIR / "models" / "representation_comparison.csv"
CHART_PATH = BASE_DIR / "models" / "representation_comparison.png"


# ---------------------------------------------------------
# Data preparation
# ---------------------------------------------------------

def load_and_prepare_data():
    """Load the dataset, clean review text, and create a common split."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)

    required_columns = {"review_text", "sentiment"}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}"
        )

    data = data.dropna(subset=["review_text", "sentiment"]).copy()
    data["review_text"] = data["review_text"].astype(str)
    data = data[data["review_text"].str.strip().ne("")].copy()

    data["cleaned_text"] = data["review_text"].apply(clean_review)
    data = data[data["cleaned_text"].str.strip().ne("")].copy()

    X = data["cleaned_text"]
    y = data["sentiment"]

    if y.nunique() < 2:
        raise ValueError("At least two sentiment classes are required.")

    class_counts = y.value_counts()
    if (class_counts < 2).any():
        small_classes = class_counts[class_counts < 2].index.tolist()
        raise ValueError(
            "Stratified splitting requires at least two records in every class. "
            f"Classes with fewer than two records: {small_classes}"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("\nDATA PREPARATION")
    print("----------------")
    print("Total usable reviews:", len(data))
    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))
    print("Sentiment classes:", sorted(y.unique().tolist()))

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------
# Word2Vec sentence vector generation
# ---------------------------------------------------------

def get_sentence_vectors(texts, word2vec_model):
    """Average the known word vectors to create one vector per review."""

    sentence_vectors = []

    for sentence in texts:
        words = sentence.split()
        word_vectors = [
            word2vec_model.wv[word]
            for word in words
            if word in word2vec_model.wv
        ]

        if word_vectors:
            sentence_vector = np.mean(word_vectors, axis=0)
        else:
            sentence_vector = np.zeros(word2vec_model.vector_size)

        sentence_vectors.append(sentence_vector)

    return np.asarray(sentence_vectors, dtype=np.float32)


# ---------------------------------------------------------
# Model training and evaluation
# ---------------------------------------------------------

def train_and_evaluate(name, X_train_features, X_test_features, y_train, y_test):
    """Train Logistic Regression and calculate classification metrics."""

    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(X_train_features, y_train)
    predictions = classifier.predict(X_test_features)

    result = {
        "Technique": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision (Macro)": precision_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "Recall (Macro)": recall_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "F1 Score (Macro)": f1_score(
            y_test, predictions, average="macro", zero_division=0
        ),
    }

    print(f"\n{name} CLASSIFICATION REPORT")
    print("-" * 45)
    print(classification_report(y_test, predictions, zero_division=0))

    return result


# ---------------------------------------------------------
# Comparison visualization
# ---------------------------------------------------------

def create_comparison_chart(comparison_df):
    """Plot and save the evaluation metric comparison."""

    chart_data = comparison_df.set_index("Technique")[
        ["Accuracy", "Precision (Macro)", "Recall (Macro)", "F1 Score (Macro)"]
    ]

    ax = chart_data.plot(kind="bar", figsize=(11, 6), ylim=(0, 1), rot=0)
    ax.set_title("BoW vs TF-IDF vs Word2Vec (CBOW)")
    ax.set_xlabel("Text Representation Technique")
    ax.set_ylabel("Score")
    ax.legend(title="Evaluation Metric")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=300, bbox_inches="tight")
    plt.show()

    print("\nComparison chart saved to:", CHART_PATH)


# ---------------------------------------------------------
# Main comparison function
# ---------------------------------------------------------

def compare_text_representations():
    """Run the complete text representation comparison experiment."""

    X_train, X_test, y_train, y_test = load_and_prepare_data()

    # BoW representation
    print("\nGenerating BoW features...")
    bow_vectorizer = create_bow_vectorizer(
        ngram_range=(1, 2),
        max_features=5000,
    )
    X_train_bow = bow_vectorizer.fit_transform(X_train)
    X_test_bow = bow_vectorizer.transform(X_test)

    # TF-IDF representation
    print("Generating TF-IDF features...")
    tfidf_vectorizer = create_tfidf_vectorizer(
        ngram_range=(1, 2),
        max_features=5000,
    )
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # Word2Vec representation (using the already-trained CBOW model)
    print("Loading the trained CBOW Word2Vec model...")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Word2Vec model not found: {MODEL_PATH}\n"
            "Train and save the CBOW model before running this comparison."
        )

    word2vec_model = Word2Vec.load(str(MODEL_PATH))

    print("Generating Word2Vec sentence vectors...")
    X_train_w2v = get_sentence_vectors(X_train, word2vec_model)
    X_test_w2v = get_sentence_vectors(X_test, word2vec_model)

    print("\nFEATURE REPRESENTATION SHAPES")
    print("-----------------------------")
    print("BoW:", X_train_bow.shape)
    print("TF-IDF:", X_train_tfidf.shape)
    print("Word2Vec:", X_train_w2v.shape)

    # Train the same classifier for each representation.
    results = [
        train_and_evaluate(
            "BoW", X_train_bow, X_test_bow, y_train, y_test
        ),
        train_and_evaluate(
            "TF-IDF", X_train_tfidf, X_test_tfidf, y_train, y_test
        ),
        train_and_evaluate(
            "Word2Vec (CBOW)", X_train_w2v, X_test_w2v, y_train, y_test
        ),
    ]

    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values(
        by="F1 Score (Macro)",
        ascending=False,
    ).reset_index(drop=True)

    print("\nFINAL REPRESENTATION COMPARISON")
    print("-------------------------------")
    print(comparison_df.to_string(index=False))

    # Save the table and chart in the existing models folder.
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(RESULTS_PATH, index=False)
    print("\nComparison results saved to:", RESULTS_PATH)

    create_comparison_chart(comparison_df)

    return comparison_df


# ---------------------------------------------------------
# Program execution
# ---------------------------------------------------------

if __name__ == "__main__":
    compare_text_representations()
