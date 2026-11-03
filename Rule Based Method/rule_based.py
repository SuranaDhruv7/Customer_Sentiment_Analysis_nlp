"""
Rule-based / heuristic sentiment analysis for customer reviews.

"""

import pandas as pd
from NLP.functions.text_preprocessing import clean_review

# ---------------------------------------------------------
# Sentiment word lists
# ---------------------------------------------------------

POSITIVE_WORDS = {
    "amazing", "awesome", "best", "beautiful", "brilliant", "comfortable",
    "excellent", "fantastic", "fast", "good", "great", "happy", "impressive",
    "love", "loved", "perfect", "positive", "recommend", "satisfied",
    "smooth", "useful", "value", "wonderful", "worth"
}

NEGATIVE_WORDS = {
    "awful", "bad", "broken", "cheap", "damage", "damaged", "disappointed",
    "disappointing", "dislike", "hate", "hated", "horrible", "issue", "poor",
    "problem", "refund", "return", "slow", "terrible", "uncomfortable",
    "useless", "waste", "worst"
}

# These words reverse the sentiment of a nearby sentiment word.
NEGATION_WORDS = {
    "not", "never", "no", "isn't", "wasn't", "weren't", "aren't",
    "don't", "doesn't", "didn't", "can't", "couldn't", "won't",
    "wouldn't", "shouldn't", "hardly"
}


# ---------------------------------------------------------
# Rule-based sentiment classification
# ---------------------------------------------------------

def classify_review_rule_based(review_text, negation_window=3):
    """
    Classify one review using positive/negative word matching.

    Returns a dictionary containing the predicted sentiment, scores,
    and matched words. If both positive and negative evidence is found,
    the review is labelled mixed.
    """

    if pd.isna(review_text) or not str(review_text).strip():
        return {
            "predicted_sentiment": "neutral",
            "positive_score": 0,
            "negative_score": 0,
            "positive_words": [],
            "negative_words": [],
        }

    cleaned_text = clean_review(review_text)
    words = cleaned_text.split()

    positive_matches = []
    negative_matches = []

    # Track how many following words are affected by a negation.
    negation_remaining = 0

    for word in words:
        if word in NEGATION_WORDS:
            negation_remaining = negation_window
            continue

        is_positive = word in POSITIVE_WORDS
        is_negative = word in NEGATIVE_WORDS

        if is_positive:
            if negation_remaining > 0:
                negative_matches.append(word)
            else:
                positive_matches.append(word)

        elif is_negative:
            if negation_remaining > 0:
                positive_matches.append(word)
            else:
                negative_matches.append(word)

        if negation_remaining > 0:
            negation_remaining -= 1

    positive_score = len(positive_matches)
    negative_score = len(negative_matches)

    if positive_score > 0 and negative_score > 0:
        predicted_sentiment = "mixed"
    elif positive_score > 0:
        predicted_sentiment = "positive"
    elif negative_score > 0:
        predicted_sentiment = "negative"
    else:
        predicted_sentiment = "neutral"

    return {
        "predicted_sentiment": predicted_sentiment,
        "positive_score": positive_score,
        "negative_score": negative_score,
        "positive_words": positive_matches,
        "negative_words": negative_matches,
    }


# ---------------------------------------------------------
# Apply rule-based method to multiple reviews
# ---------------------------------------------------------

def analyze_reviews_rule_based(reviews):
    """Apply rule-based sentiment classification to a collection of reviews."""

    results = []

    for review in reviews:
        result = classify_review_rule_based(review)
        results.append(result)

    return pd.DataFrame(results)


# ---------------------------------------------------------
# Program execution
# ---------------------------------------------------------

if __name__ == "__main__":

    sample_reviews = [
        "The product is amazing and the delivery was fast.",
        "The product is terrible and the quality is poor.",
        "The product is good but the delivery was slow.",
        "The product is not good.",
        "The product arrived yesterday."
    ]

    sample_results = analyze_reviews_rule_based(sample_reviews)

    output = pd.concat(
        [
            pd.Series(sample_reviews, name="review_text"),
            sample_results
        ],
        axis=1
    )

    print("\nRULE-BASED SENTIMENT ANALYSIS")
    print("-----------------------------")
    print(output.to_string(index=False))
