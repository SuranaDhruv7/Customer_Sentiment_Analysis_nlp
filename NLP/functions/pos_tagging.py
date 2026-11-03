# Import required libraries
import nltk
from nltk.tokenize import word_tokenize


# Download required NLTK resources
nltk.download("punkt_tab", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)


# Perform POS tagging on a single review
def perform_pos_tagging(text):

    # Tokenize the review
    tokens = word_tokenize(str(text))

    # Identify the grammatical role of each word
    pos_tags = nltk.pos_tag(tokens)

    return pos_tags


# Perform POS tagging on multiple reviews
def tag_multiple_reviews(reviews, limit=5):

    # Select the required number of reviews
    selected_reviews = reviews[:limit]

    results = []

    for review in selected_reviews:

        tags = perform_pos_tagging(review)

        results.append({
            "review": review,
            "pos_tags": tags
        })

    return results


# Display POS tagging results
def display_pos_results(results):

    for i, result in enumerate(results, start=1):

        print(f"\nReview {i}:")
        print("Text:", result["review"])
        print("POS Tags:", result["pos_tags"])


# Add POS tags as additional text features
def add_pos_features(text):

    text = str(text).strip()

    if not text:
        return ""

    # Generate POS tags
    pos_tags = perform_pos_tagging(text)

    # Add POS tags to individual words
    tagged_words = [
        f"{word}_{tag}"
        for word, tag in pos_tags
    ]

    # Preserve original words and append POS features
    enriched_text = text + " " + " ".join(tagged_words)

    return enriched_text


# Confirm successful execution
if __name__ == "__main__":

    print("POS Tagging Module Executed Successfully!")