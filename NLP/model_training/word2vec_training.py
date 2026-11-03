# Import required libraries
import pandas as pd

from pathlib import Path

from NLP.functions.word2vec import (
    tokenize_reviews,
    create_word2vec_model,
    train_word2vec_model,
    display_word2vec_info
)


# Project paths
BASE_DIR = Path.cwd()

DATA_PATH = BASE_DIR / "data" / "raw" / "customer_reviews_sentiment_practice.csv"

MODEL_PATH = BASE_DIR / "models"


# Train Word2Vec model
def train_word2vec_models():

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Select customer reviews
    reviews = df["review_text"]

    # Remove missing and empty reviews
    reviews = reviews.dropna()
    reviews = reviews[reviews.str.strip().ne("")]

    # Tokenize customer reviews
    tokenized_data = tokenize_reviews(reviews)

    # Train CBOW model
    print("\n========== TRAINING CBOW MODEL ==========")

    cbow_model = create_word2vec_model(
        method="cbow",
        vector_size=100,
        window=5,
        min_count=1,
        epochs=10
    )

    cbow_model = train_word2vec_model(
        cbow_model,
        tokenized_data
    )

    # Display CBOW information
    display_word2vec_info(cbow_model)

    # Save CBOW model
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    cbow_path = MODEL_PATH / "trained_word2vec_cbow.model"

    cbow_model.save(str(cbow_path))

    print("\nCBOW Model Saved Successfully!")
    print("Model Location:", cbow_path)

    # Train Skip-gram model
    print("\n========== TRAINING SKIP-GRAM MODEL ==========")

    skipgram_model = create_word2vec_model(
        method="skipgram",
        vector_size=100,
        window=5,
        min_count=1,
        epochs=10
    )

    skipgram_model = train_word2vec_model(
        skipgram_model,
        tokenized_data
    )

    # Display Skip-gram information
    display_word2vec_info(skipgram_model)

    # Save Skip-gram model
    skipgram_path = MODEL_PATH / "trained_word2vec_skipgram.model"

    skipgram_model.save(str(skipgram_path))

    print("\nSkip-gram Model Saved Successfully!")
    print("Model Location:", skipgram_path)

    return cbow_model, skipgram_model


# Confirm successful execution
if __name__ == "__main__":

    train_word2vec_models()