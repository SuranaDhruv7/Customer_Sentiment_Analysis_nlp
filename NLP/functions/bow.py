# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer


# Create BoW vectorizer
def create_bow_vectorizer(ngram_range=(1, 1), max_features=None):

    # Initialize vectorizer with n-gram and feature settings
    vectorizer = CountVectorizer(
        ngram_range=ngram_range,
        max_features=max_features
    )

    return vectorizer


# Train BoW vectorizer on training data
def train_bow_vectorizer(vectorizer, text_data):

    # Learn vocabulary and convert text into word counts
    X_bow = vectorizer.fit_transform(text_data)

    return X_bow


# Transform testing or new text data
def transform_bow_data(vectorizer, text_data):

    # Use the existing vocabulary to convert text
    X_bow = vectorizer.transform(text_data)

    return X_bow


# Get vocabulary words from the vectorizer
def get_bow_feature_names(vectorizer):

    return vectorizer.get_feature_names_out()


# Convert sparse matrix into a dense array
def get_bow_feature_matrix(X_bow):

    return X_bow.toarray()


# Display basic BoW information
def display_bow_info(vectorizer, X_bow):

    print("\n========== BAG OF WORDS INFORMATION ==========")

    # Display vocabulary size
    print("Vocabulary Size:", len(vectorizer.vocabulary_))

    # Display number of reviews
    print("Number of Reviews:", X_bow.shape[0])

    # Display number of features
    print("Number of Features:", X_bow.shape[1])

    # Display matrix dimensions
    print("Matrix Shape:", X_bow.shape)

    print("===============================================")


# Confirm successful execution
if __name__ == "__main__":
    print("Bag of Words Module Executed Successfully!")