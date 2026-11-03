# 📊 Customer Sentiment Analysis Using NLP

## 📌 Project Overview

This project focuses on analysing customer product reviews using **Natural Language Processing (NLP)** and Machine Learning.

It explores different text representation techniques, builds sentiment classification models, performs rule-based sentiment analysis, and provides an interactive Streamlit application for analysing individual and multiple customer reviews.

## 🎯 Project Objectives

- Analyse customer reviews and their sentiment distribution.
- Clean and preprocess raw text data.
- Implement Bag of Words (BoW), TF-IDF and Word2Vec.
- Build sentiment classification models using Logistic Regression.
- Explore CBoW and Skip-gram word embeddings.
- Implement rule-based sentiment analysis and POS tagging.
- Compare different text representation techniques.
- Develop an interactive Streamlit application.

## 📂 Dataset

**Dataset:** `customer_reviews_sentiment_practice.csv`

- Total Records: 600
- Total Columns: 10
- Missing Values: 0
- Duplicate Records: 0
- Sentiment Classes: Positive, Negative, Neutral and Mixed.

### 📊 Sentiment Distribution

| Sentiment | Reviews | Percentage |
|---|---:|---:|
| Positive | 246 | 41.00% |
| Negative | 196 | 32.67% |
| Neutral | 127 | 21.17% |
| Mixed | 31 | 5.17% |

## 🧹 Text Preprocessing

Implemented a reusable text preprocessing module using Python, Regex and Pandas.

**Preprocessing steps:**
- Handle missing text values.
- Convert HTML entities and remove HTML tags.
- Remove URLs.
- Convert text to lowercase.
- Remove unwanted symbols while preserving apostrophes.
- Normalize extra whitespace.
- Preserve important negation words such as *not* and *never*.

## 🧠 NLP Techniques Implemented

### 1. Bag of Words (BoW)
- Converts text into numerical word-frequency features.
- Uses `CountVectorizer`.
- Supports unigram and bigram features.

### 2. TF-IDF
- Converts text into numerical features based on term importance.
- Uses `TfidfVectorizer`.
- Supports unigram and bigram features.

### 3. Word2Vec
Implemented Word2Vec using two approaches:

- **CBoW (Continuous Bag of Words):** Predicts a target word using surrounding context words.
- **Skip-gram:** Predicts surrounding context words using a target word.

**Configuration:**
- Vector Size: 100
- Window Size: 5
- Minimum Word Count: 1
- Training Epochs: 10

Also implemented word vectors, sentence vectors and similar-word retrieval.

## 🤖 Sentiment Classification

Built two sentiment classification pipelines using **Logistic Regression**:

- BoW + Logistic Regression
- TF-IDF + Logistic Regression

**Training Configuration:**
- Train-Test Split: 80/20
- Stratified Sampling
- Random State: 42
- Maximum Iterations: 1000
- Maximum Features: 5000

Both pipelines include preprocessing, vectorization and classification, and are saved using Joblib for application use.

## 📈 Model Comparison

Compared three text representation techniques using the same Logistic Regression classifier:

- BoW
- TF-IDF
- Word2Vec (CBoW sentence vectors)

### Evaluation Results

| Technique | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| BoW | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| TF-IDF | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Word2Vec (CBoW) | 0.5583 | 0.3918 | 0.4219 | 0.3856 |

**Observation:**
- BoW and TF-IDF achieved the highest scores on the comparison test split.
- Word2Vec (CBoW) achieved 55.83% accuracy.
- BoW and TF-IDF performed better than the Word2Vec sentence-vector approach in this experiment.

*Note: These results are specific to the current dataset and test split. Further evaluation on independent, unseen customer reviews is recommended, particularly to investigate the perfect BoW and TF-IDF scores.*

## ⚙️ Additional NLP Techniques

### Rule-Based Sentiment Analysis
Implemented a heuristic sentiment classification approach using:

- Positive and negative word lists.
- Negation detection.
- A three-word negation window.
- Positive, negative, neutral and mixed sentiment classification.

### Part-of-Speech (POS) Tagging
Implemented POS tagging using NLTK to:
- Tokenize customer reviews.
- Identify grammatical roles of words.
- Generate POS-tagged text features.
- Process multiple reviews.

## 🌐 Streamlit Application

Developed an interactive application with five main sections:

1. **Overview** – Displays project information and dataset summary.
2. **Single Review** – Predicts sentiment for an individual customer review.
3. **Batch Prediction** – Uploads a CSV file and generates multiple predictions.
4. **Sentiment Dashboard** – Displays sentiment distribution, category-level analysis, product reviews and monthly trends.
5. **Model Comparison** – Displays evaluation metrics and comparison charts.

**Application Features:**
- Select between saved BoW and TF-IDF prediction models.
- Analyse individual customer reviews.
- Upload CSV files containing a `review_text` column.
- View batch prediction results.
- Download predictions as CSV.
- Filter dashboard results by product category.

*Word2Vec is used in the representation comparison experiment. It is not a direct prediction model in the Streamlit application.*

## 🛠️ Technologies & Libraries

- Python
- Pandas & NumPy
- Scikit-learn
- NLTK
- Gensim
- Regex
- Joblib
- Matplotlib & Seaborn
- Streamlit

## 📁 Project Structure

```text
Customer Sentiment Analysis/
│
├── data/
│   └── raw/
│       └── customer_reviews_sentiment_practice.csv
│
├── models/
│   ├── trained_bow_model.pkl
│   ├── trained_tfidf_model.pkl
│   ├── trained_word2vec_cbow.model
│   ├── trained_word2vec_skipgram.model
│   ├── representation_comparison.csv
│   └── representation_comparison.png
│
├── NLP/
│   ├── functions/
│   │   ├── bow.py
│   │   ├── tf_idf.py
│   │   ├── word2vec.py
│   │   ├── text_preprocessing.py
│   │   ├── pos_tagging.py
│   │   └── rule_based.py
│   │
│   └── model_training/
│       ├── bow_training.py
│       ├── tf_idf_training.py
│       └── word2vec_training.py
│
├── NLP Fundamentals/
│
├── Streamlit/
│   └── app.py
│
├── comparison.py
├── model.ipynb
├── requirements.txt
└── Readme.md
```

## 🚀 Execution

### 1. Run the Streamlit Application

```bash
streamlit run Streamlit/app.py
```

## ⚠️ Project Limitations

- Sentiment predictions depend on the quality and coverage of the training dataset.
- Rule-based analysis relies on predefined sentiment words and negation rules.
- Word2Vec sentence vectors are generated by averaging available word embeddings.
- Sarcasm, context and complex mixed opinions can be difficult to interpret.
- High classification scores require further validation on independent data.

## 🏁 Conclusion

This project demonstrates an end-to-end NLP workflow for customer review analysis, combining text preprocessing, feature extraction, Word2Vec embeddings, Logistic Regression, rule-based sentiment analysis and POS tagging.

It also demonstrates the practical use of trained NLP pipelines through an interactive Streamlit application.

**The project provided hands-on experience in building NLP modules, training machine-learning models, comparing text representation techniques and developing an application for customer sentiment analysis.**

---

⭐ *Built as a practical NLP and Machine Learning project.*