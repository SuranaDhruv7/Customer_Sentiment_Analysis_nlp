# Import required libraries
import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer

from NLP.functions.text_preprocessing import preprocess_texts
from NLP.functions.tf_idf import create_tfidf_vectorizer


# Project paths
BASE_DIR = Path.cwd()

DATA_PATH = BASE_DIR / "data" / "raw" / "customer_reviews_sentiment_practice.csv"

MODEL_PATH = BASE_DIR / "models" / "trained_tfidf_model.pkl"

# Train TF-IDF model
def train_tfidf_model():

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Input and target
    X = df["review_text"]
    y = df["sentiment"]

    # Remove missing and empty records
    valid_data = X.notna() & y.notna() & X.str.strip().ne("")

    X = X[valid_data]
    y = y[valid_data]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Create TF-IDF pipeline
    tfidf_pipeline = Pipeline([

        ("preprocessing", FunctionTransformer(
            preprocess_texts,
            validate=False
        )),

        ("vectorizer", create_tfidf_vectorizer(
            ngram_range=(1, 2),
            max_features=5000
        )),

        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))

    ])

    # Train model
    tfidf_pipeline.fit(X_train, y_train)

    # Create models directory if it does not exist
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save trained model
    joblib.dump(tfidf_pipeline, MODEL_PATH)

    print("\nTF-IDF Model Trained Successfully!")
    print("Trained Model Saved Successfully!")
    print("Model Location:", MODEL_PATH)

    return tfidf_pipeline


# Confirm successful execution
if __name__ == "__main__":

    train_tfidf_model()