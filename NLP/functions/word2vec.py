# Import required libraries
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

# Download tokenizer resource
nltk.download("punkt_tab", quiet=True)


# Tokenize customer reviews
def tokenize_reviews(text_data):

    # Convert reviews into lists of words
    tokenized_data = [
        word_tokenize(str(review).lower())
        for review in text_data
        if str(review).strip()
    ]

    return tokenized_data


# Create Word2Vec model
def create_word2vec_model(
    method="cbow",
    vector_size=100,
    window=5,
    min_count=1,
    epochs=10
):

    # Select CBOW or Skip-gram
    sg = 0 if method.lower() == "cbow" else 1

    # Initialize Word2Vec model
    model = Word2Vec(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        workers=1,
        seed=42,
        epochs=epochs
    )

    return model


# Train Word2Vec model
def train_word2vec_model(model, tokenized_data):

    # Build vocabulary from tokenized reviews
    model.build_vocab(tokenized_data)

    # Train word embeddings
    model.train(
        tokenized_data,
        total_examples=model.corpus_count,
        epochs=model.epochs
    )

    return model


# Get vector representation of a word
def get_word_vector(model, word):

    # Check whether word exists in vocabulary
    if word in model.wv:

        return model.wv[word]

    return None


# Get sentence or review vector
def get_sentence_vector(model, tokens):

    # Collect vectors of words present in vocabulary
    word_vectors = [
        model.wv[word]
        for word in tokens
        if word in model.wv
    ]

    # Return zero vector if no words are available
    if not word_vectors:

        return [0.0] * model.vector_size

    # Calculate average word embeddings
    return sum(word_vectors) / len(word_vectors)


# Get similar words
def get_similar_words(model, word, topn=5):

    # Check whether word exists in vocabulary
    if word in model.wv:

        return model.wv.most_similar(word, topn=topn)

    return []


# Display Word2Vec model information
def display_word2vec_info(model):

    print("\n========== WORD2VEC INFORMATION ==========")

    # Display vocabulary size
    print("Vocabulary Size:", len(model.wv))

    # Display embedding dimensions
    print("Vector Size:", model.vector_size)

    # Display context window
    print("Window Size:", model.window)

    # Display training approach
    print(
        "Training Method:",
        "Skip-gram" if model.sg == 1 else "CBOW"
    )

    # Display training epochs
    print("Training Epochs:", model.epochs)
    print("===========================================")


# Confirm successful execution
if __name__ == "__main__":
    print("Word2Vec Module Executed Successfully!")