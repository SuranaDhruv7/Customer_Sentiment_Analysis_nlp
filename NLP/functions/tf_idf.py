# Import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Create TF-IDF vectorizer
def create_tfidf_vectorizer(ngram_range=(1, 1), max_features=None):

    # Initialize vectorizer with n-gram and feature settings
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features
    )

    return vectorizer

# Train TF-IDF vectorizer on training data
def train_tfidf_vectorizer(vectorizer, text_data):

    # Learn vocabulary and calculate TF-IDF weights
    X_tfidf = vectorizer.fit_transform(text_data)

    return X_tfidf

# Transform testing or new text data
def transform_tfidf_data(vectorizer, text_data):

    # Use the existing vocabulary and IDF weights
    X_tfidf = vectorizer.transform(text_data)

    return X_tfidf

# Get vocabulary words from the vectorizer
def get_tfidf_feature_names(vectorizer):

    return vectorizer.get_feature_names_out()


# Convert sparse matrix into a dense array
def get_tfidf_feature_matrix(X_tfidf):

    return X_tfidf.toarray()


# Display basic TF-IDF information
def display_tfidf_info(vectorizer, X_tfidf):

    print("\n========== TF-IDF INFORMATION ==========")

    # Display vocabulary size
    print("Vocabulary Size:", len(vectorizer.vocabulary_))

    # Display number of reviews
    print("Number of Reviews:", X_tfidf.shape[0])

    # Display number of features
    print("Number of Features:", X_tfidf.shape[1])

    # Display matrix dimensions
    print("Matrix Shape:", X_tfidf.shape)

    print("=========================================")


# Confirm successful execution
if __name__ == "__main__":
    print("TF-IDF Module Executed Successfully!")