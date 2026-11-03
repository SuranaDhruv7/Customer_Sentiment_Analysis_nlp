
"""
Customer Sentiment Analysis - Streamlit Application

Run from project root:
streamlit run Streamlit/app.py
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# =========================================================
# 1. PROJECT PATHS AND IMPORTS
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[1]

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

DATA_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "customer_reviews_sentiment_practice.csv"
)

MODEL_DIR = BASE_DIR / "models"

BOW_MODEL_PATH = MODEL_DIR / "trained_bow_model.pkl"
TFIDF_MODEL_PATH = MODEL_DIR / "trained_tfidf_model.pkl"

COMPARISON_PATH = MODEL_DIR / "representation_comparison.csv"

# Import before loading Joblib pipelines.
from NLP.functions.text_preprocessing import preprocess_texts


# =========================================================
# 2. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Sentiment Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 3. PROFESSIONAL APPLICATION THEME
# =========================================================

st.markdown(
    """
    <style>

    /* =========================================
       GLOBAL THEME
    ========================================= */

    :root {
        color-scheme: light;
    }

    .stApp {
        background-color: #F1F5F9;
        color: #334155;
        font-family: "Segoe UI", Arial, sans-serif;
    }

    [data-testid="stHeader"] {
        background-color: #F1F5F9;
    }

    [data-testid="stMain"] {
        background-color: #F1F5F9;
    }

    [data-testid="stMain"] .block-container {
        max-width: 1450px;
        padding-top: 2.2rem;
        padding-bottom: 2.5rem;
    }

    /* =========================================
       MAIN TYPOGRAPHY
    ========================================= */

    [data-testid="stMain"] h1 {
        color: #14243B !important;
        font-size: 40px !important;
        font-weight: 750 !important;
        line-height: 1.25 !important;
    }

    [data-testid="stMain"] h2 {
        color: #14243B !important;
        font-size: 31px !important;
        font-weight: 700 !important;
    }

    [data-testid="stMain"] h3 {
        color: #1E3A5F !important;
        font-size: 24px !important;
        font-weight: 700 !important;
    }

    [data-testid="stMain"] h4 {
        color: #1E3A5F !important;
        font-size: 21px !important;
        font-weight: 650 !important;
    }

    [data-testid="stMain"] p,
    [data-testid="stMain"] li {
        color: #334155 !important;
        font-size: 18px !important;
        line-height: 1.65 !important;
    }

    [data-testid="stMain"] strong,
    [data-testid="stMain"] b {
        color: #1E293B !important;
    }

    [data-testid="stMain"] small {
        color: #526174 !important;
    }

    /* =========================================
       SIDEBAR
    ========================================= */

    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #172B4D 0%,
            #1D3558 100%
        ) !important;
        min-width: 290px;
    }

    [data-testid="stSidebar"] > div {
        background: transparent;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #FFFFFF !important;
    }

    [data-testid="stSidebar"] h2 {
        font-size: 29px !important;
        font-weight: 750 !important;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] .stCaption {
        color: #D3DEEC !important;
        font-size: 17px !important;
        line-height: 1.6 !important;
    }

    [data-testid="stSidebar"] label {
        color: #FFFFFF !important;
        font-size: 19px !important;
        font-weight: 550 !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] > label {
        color: #AFC3DC !important;
        font-size: 17px !important;
        font-weight: 700 !important;
        letter-spacing: 0.7px;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
        gap: 12px;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label p {
        color: #F1F5F9 !important;
        font-size: 19px !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background-color: rgba(255, 255, 255, 0.09);
        border-radius: 8px;
    }

    [data-testid="stSidebar"] hr {
        border-color: #526985 !important;
        margin-top: 22px;
        margin-bottom: 22px;
    }

    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #D3DEEC !important;
    }

    /* =========================================
       HERO HEADER
    ========================================= */

    .hero {
        background: linear-gradient(
            115deg,
            #172B4D 0%,
            #294B70 100%
        );
        padding: 38px 42px;
        border-radius: 18px;
        margin-bottom: 32px;
        box-shadow: 0 8px 22px rgba(23, 43, 77, 0.12);
    }

    .hero h1 {
        color: #FFFFFF !important;
        font-size: 41px !important;
        font-weight: 750 !important;
        margin: 0 0 12px 0;
        line-height: 1.25 !important;
    }

    .hero p {
        color: #E2EAF4 !important;
        font-size: 19px !important;
        font-weight: 400;
        margin: 0;
        line-height: 1.6 !important;
    }

    /* =========================================
       CONTENT CARDS
    ========================================= */

    .content-card {
        background-color: #FFFFFF;
        border: 1px solid #D9E2EC;
        border-radius: 15px;
        padding: 25px;
        margin-bottom: 18px;
        box-shadow: 0 4px 14px rgba(23, 43, 77, 0.045);
    }

    .content-card h3 {
        color: #1E3A5F !important;
        margin-top: 0;
    }

    .content-card p {
        color: #475569 !important;
    }

    /* =========================================
       METRIC CARDS
    ========================================= */

    div[data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 1px solid #D9E2EC !important;
        border-radius: 15px !important;
        padding: 23px !important;
        box-shadow: 0 4px 14px rgba(23, 43, 77, 0.05);
    }

    [data-testid="stMetricLabel"] {
        color: #475569 !important;
        font-size: 17px !important;
        font-weight: 650 !important;
    }

    [data-testid="stMetricValue"] {
        color: #172B4D !important;
        font-size: 35px !important;
        font-weight: 750 !important;
    }

    [data-testid="stMetricDelta"] {
        font-size: 15px !important;
    }

    /* =========================================
       INPUT LABELS
    ========================================= */

    [data-testid="stTextArea"] label,
    [data-testid="stTextInput"] label,
    [data-testid="stSelectbox"] label,
    [data-testid="stFileUploader"] label,
    [data-testid="stDateInput"] label {
        color: #1E3A5F !important;
        font-size: 19px !important;
        font-weight: 650 !important;
    }

    /* =========================================
       TEXT AREA AND TEXT INPUT
    ========================================= */

    [data-testid="stTextArea"] textarea,
    [data-testid="stTextInput"] input {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
        border: 1.5px solid #CBD5E1 !important;
        border-radius: 11px !important;
        font-size: 18px !important;
        padding: 13px !important;
        caret-color: #0F766E !important;
    }

    [data-testid="stTextArea"] textarea::placeholder,
    [data-testid="stTextInput"] input::placeholder {
        color: #64748B !important;
        opacity: 1 !important;
    }

    [data-testid="stTextArea"] textarea:focus,
    [data-testid="stTextInput"] input:focus {
        border-color: #0F766E !important;
        box-shadow: 0 0 0 2px rgba(15, 118, 110, 0.13) !important;
    }

    /* =========================================
       SELECT BOX
    ========================================= */

    [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 10px !important;
        min-height: 45px;
    }

    [data-testid="stSelectbox"] div[data-baseweb="select"] span {
        color: #1E293B !important;
        font-size: 17px !important;
    }

    [data-baseweb="popover"] {
        background-color: #FFFFFF !important;
    }

    [data-baseweb="popover"] ul {
        background-color: #FFFFFF !important;
    }

    [data-baseweb="popover"] li {
        color: #1E293B !important;
        font-size: 17px !important;
    }

    /* =========================================
       BUTTONS
    ========================================= */

    div.stButton > button,
    div.stDownloadButton > button {
        border-radius: 10px !important;
        font-size: 18px !important;
        font-weight: 650 !important;
        min-height: 3.2rem;
        transition: all 0.2s ease;
    }

    div.stButton > button[kind="primary"],
    div.stDownloadButton > button[kind="primary"] {
        background-color: #0F766E !important;
        color: #FFFFFF !important;
        border: 1px solid #0F766E !important;
    }

    div.stButton > button[kind="primary"]:hover,
    div.stDownloadButton > button[kind="primary"]:hover {
        background-color: #115E59 !important;
        border-color: #115E59 !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 12px rgba(15, 118, 110, 0.2);
    }

    div.stButton > button:not([kind="primary"]),
    div.stDownloadButton > button:not([kind="primary"]) {
        background-color: #FFFFFF !important;
        color: #1E3A5F !important;
        border: 1px solid #CBD5E1 !important;
    }

    div.stButton > button:not([kind="primary"]):hover,
    div.stDownloadButton > button:not([kind="primary"]):hover {
        background-color: #E6F3F1 !important;
        border-color: #0F766E !important;
        color: #115E59 !important;
    }

    /* =========================================
       FILE UPLOADER
    ========================================= */

    [data-testid="stFileUploader"] {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 15px !important;
        padding: 18px !important;
        box-shadow: 0 4px 14px rgba(23, 43, 77, 0.045);
    }

    [data-testid="stFileUploaderDropzone"] {
        background-color: #F8FAFC !important;
        border: 1.5px dashed #94A3B8 !important;
        border-radius: 11px !important;
        min-height: 115px;
    }

    [data-testid="stFileUploaderDropzone"] * {
        color: #334155 !important;
        font-size: 17px !important;
    }

    [data-testid="stFileUploaderDropzone"] button {
        background-color: #E6F3F1 !important;
        color: #115E59 !important;
        border: 1px solid #0F766E !important;
        border-radius: 8px !important;
        font-weight: 650 !important;
        min-height: 40px;
    }

    [data-testid="stFileUploaderDropzone"] button:hover {
        background-color: #D1EAE6 !important;
        color: #115E59 !important;
    }

    [data-testid="stFileUploaderFile"] {
        background-color: #F1F5F9 !important;
        border-radius: 9px !important;
    }

    /* =========================================
       CHART CONTAINERS
    ========================================= */

    [data-testid="stVegaLiteChart"],
    [data-testid="stArrowVegaLiteChart"],
    [data-testid="stPyplotChart"],
    [data-testid="stDeckGlJsonChart"] {
        background-color: #FFFFFF !important;
        border: 1px solid #D9E2EC !important;
        border-radius: 13px !important;
        padding: 12px !important;
        box-shadow: 0 3px 12px rgba(23, 43, 77, 0.04);
    }

    [data-testid="stChart"] {
        background-color: #FFFFFF !important;
        border-radius: 13px !important;
    }

    /* =========================================
       DATA TABLES
    ========================================= */

    [data-testid="stDataFrame"] {
        background-color: #FFFFFF !important;
        border: 1px solid #D9E2EC !important;
        border-radius: 12px !important;
        overflow: hidden;
    }

    [data-testid="stDataFrame"] * {
        color: #334155;
    }

    /* =========================================
       ALERTS AND INFO BOXES
    ========================================= */

    [data-testid="stAlert"] {
        border-radius: 12px !important;
        font-size: 17px !important;
    }

    [data-testid="stAlert"] p {
        font-size: 17px !important;
    }

    /* =========================================
       DIVIDERS AND CAPTIONS
    ========================================= */

    [data-testid="stMain"] hr {
        border-color: #D5DEE8 !important;
        margin-top: 25px;
        margin-bottom: 25px;
    }

    [data-testid="stMain"] [data-testid="stCaptionContainer"] p {
        color: #64748B !important;
        font-size: 15px !important;
    }

    /* =========================================
       FOOTER
    ========================================= */

    .app-footer {
        color: #64748B;
        font-size: 15px;
        line-height: 1.7;
        text-align: center;
        padding: 15px 10px;
    }

    footer {
        visibility: hidden;
    }

    /* =========================================
       RESPONSIVE ADJUSTMENTS
    ========================================= */

    @media (max-width: 768px) {

        [data-testid="stMain"] .block-container {
            padding: 1.2rem 1rem 2rem 1rem;
        }

        .hero {
            padding: 26px 23px;
        }

        .hero h1 {
            font-size: 30px !important;
        }

        .hero p {
            font-size: 17px !important;
        }

        [data-testid="stMain"] h1 {
            font-size: 31px !important;
        }

        [data-testid="stMain"] h2 {
            font-size: 26px !important;
        }

        [data-testid="stMain"] h3 {
            font-size: 21px !important;
        }

        [data-testid="stSidebar"] {
            min-width: 250px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 4. LOAD MODELS AND DATA
# =========================================================

@st.cache_resource(show_spinner="Loading trained models...")
def load_models():

    loaded_models = {}

    if BOW_MODEL_PATH.exists():
        loaded_models["BoW"] = joblib.load(BOW_MODEL_PATH)

    if TFIDF_MODEL_PATH.exists():
        loaded_models["TF-IDF"] = joblib.load(TFIDF_MODEL_PATH)

    return loaded_models


@st.cache_data(show_spinner=False)
def load_dataset():

    if not DATA_PATH.exists():
        return None

    return pd.read_csv(DATA_PATH)


@st.cache_data(show_spinner=False)
def load_comparison_results():

    if not COMPARISON_PATH.exists():
        return None

    return pd.read_csv(COMPARISON_PATH)


try:
    models = load_models()

except Exception as error:
    models = {}
    st.error(f"Unable to load trained models: {error}")


dataset = load_dataset()
comparison_results = load_comparison_results()


# =========================================================
# 5. REUSABLE INTERFACE FUNCTIONS
# =========================================================

def show_header(title, subtitle):

    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_prediction_result(review, prediction, model_name):

    st.markdown("### Prediction Result")

    st.success(
        f"Predicted Sentiment: {str(prediction).title()}"
    )


def get_available_prediction_models():

    return list(models.keys())


# =========================================================
# 6. SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:

    st.markdown("## 📊 Sentiment Studio")

    st.caption("Customer Review Intelligence")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Single Review",
            "Batch Prediction",
            "Sentiment Dashboard",
            "Model Comparison"
        ],
        label_visibility="visible"
    )

    st.divider()

    st.caption("NLP-based customer review analysis")

    st.caption("BoW  ·  TF-IDF  ·  Word2Vec")


# =========================================================
# 7. OVERVIEW PAGE
# =========================================================

if page == "Overview":

    show_header(
        "Customer Sentiment Analysis",
        "Understand customer feedback through natural language processing."
    )

    st.markdown("## Welcome")

    st.write(
        "This application analyses customer product reviews "
        "and classifies their sentiment as positive, negative, "
        "neutral, or mixed using trained machine-learning models."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Prediction Models Available",
        len(models)
    )

    col2.metric(
        "Dataset Records",
        f"{len(dataset):,}" if dataset is not None else "Not available"
    )

    col3.metric(
        "Representation Techniques",
        "3"
    )

    st.markdown("## Explore the Application")

    feature_cols = st.columns(2)

    with feature_cols[0]:

        st.markdown(
            """
            <div class="content-card">
                <h3>✍️ Single Review Analysis</h3>
                <p>Enter an individual customer review and predict its sentiment.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="content-card">
                <h3>📁 Batch CSV Prediction</h3>
                <p>Upload a CSV file and generate predictions for multiple reviews.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="content-card">
                <h3>📈 Sentiment Dashboard</h3>
                <p>Explore sentiment distribution and customer feedback patterns.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with feature_cols[1]:

        st.markdown(
            """
            <div class="content-card">
                <h3>⚖️ Model Comparison</h3>
                <p>Review the performance of BoW, TF-IDF, and Word2Vec representations.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="content-card">
                <h3>⬇️ Download Results</h3>
                <p>Export batch predictions as a CSV file for further analysis.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.info(
        "Predictions are generated using the saved BoW and TF-IDF "
        "classification pipelines. Word2Vec is included in the "
        "representation comparison; its saved file is an embedding "
        "model rather than a sentiment classification pipeline."
    )


# =========================================================
# 8. SINGLE REVIEW ANALYSIS
# =========================================================

elif page == "Single Review":

    show_header(
        "Single Review Analysis",
        "Enter a customer review to identify its predicted sentiment."
    )

    if not models:

        st.error(
            "No saved prediction models were found. "
            "Train and save the BoW and/or TF-IDF pipeline first."
        )

    else:

        selected_model = st.selectbox(
            "Choose a Classification Model",
            get_available_prediction_models()
        )

        review_text = st.text_area(
            "Customer Review",
            placeholder=(
                "Example: The product quality is excellent "
                "and delivery was very fast."
            ),
            height=180
        )

        if st.button(
            "Analyse Review",
            type="primary",
            use_container_width=True
        ):

            if not review_text.strip():

                st.warning(
                    "Please enter a review before analysing."
                )

            else:

                try:

                    prediction = models[selected_model].predict(
                        [review_text]
                    )[0]

                    show_prediction_result(
                        review_text,
                        prediction,
                        selected_model
                    )

                except Exception as error:

                    st.error(
                        f"Unable to generate prediction: {error}"
                    )


# =========================================================
# 9. BATCH CSV PREDICTION
# =========================================================

elif page == "Batch Prediction":

    show_header(
        "Batch CSV Prediction",
        "Upload customer reviews and analyse them in one step."
    )

    st.write(
        "Upload a CSV file containing a column named "
        "**review_text**. The application will preserve "
        "the original columns and add predicted sentiment."
    )

    selected_model = None

    if models:

        selected_model = st.selectbox(
            "Choose a Classification Model",
            get_available_prediction_models(),
            key="batch_model"
        )

    else:

        st.error(
            "No saved BoW or TF-IDF prediction model is available."
        )

    uploaded_file = st.file_uploader(
        "Upload Customer Reviews CSV",
        type=["csv"],
        help="The CSV file must contain a review_text column."
    )

    if uploaded_file is not None and selected_model is not None:

        try:

            batch_data = pd.read_csv(uploaded_file)

            if "review_text" not in batch_data.columns:

                st.error(
                    "Invalid CSV: a 'review_text' column is required."
                )

            elif batch_data.empty:

                st.warning(
                    "The uploaded CSV contains no records."
                )

            else:

                valid_reviews = (
                    batch_data["review_text"].notna()
                    &
                    batch_data["review_text"]
                    .astype(str)
                    .str.strip()
                    .ne("")
                )

                if not valid_reviews.any():

                    st.warning(
                        "The CSV does not contain any non-empty reviews."
                    )

                else:

                    st.markdown("### Uploaded Data Preview")

                    st.dataframe(
                        batch_data.head(10),
                        use_container_width=True
                    )

                    if st.button(
                        "Generate Batch Predictions",
                        type="primary",
                        use_container_width=True
                    ):

                        output_data = batch_data.copy()

                        output_data["predicted_sentiment"] = pd.NA

                        predictions = models[selected_model].predict(
                            batch_data.loc[
                                valid_reviews,
                                "review_text"
                            ].astype(str)
                        )

                        output_data.loc[
                            valid_reviews,
                            "predicted_sentiment"
                        ] = predictions

                        st.session_state["batch_output"] = output_data

                        st.session_state["batch_output_model"] = selected_model

        except pd.errors.EmptyDataError:

            st.error(
                "The uploaded CSV file is empty or cannot be read."
            )

        except Exception as error:

            st.error(
                f"Unable to process the uploaded CSV: {error}"
            )

    if "batch_output" in st.session_state:

        output_data = st.session_state["batch_output"]

        st.markdown("## Prediction Results")

        if "batch_output_model" in st.session_state:
            st.caption(
                f"Classification model: {st.session_state['batch_output_model']}"
            )

        st.dataframe(
            output_data,
            use_container_width=True
        )

        predicted_counts = (
            output_data["predicted_sentiment"]
            .value_counts(dropna=True)
        )

        st.markdown("## Predicted Sentiment Distribution")

        st.bar_chart(predicted_counts)

        csv_data = output_data.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download Predictions CSV",
            data=csv_data,
            file_name="customer_sentiment_predictions.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True
        )


# =========================================================
# 10. SENTIMENT DASHBOARD
# =========================================================

elif page == "Sentiment Dashboard":

    show_header(
        "Sentiment Dashboard",
        "Explore sentiment patterns in the original customer review dataset."
    )

    if dataset is None:

        st.error(
            f"Dataset not found at: {DATA_PATH}"
        )

    elif "sentiment" not in dataset.columns:

        st.error(
            "The dataset does not contain a 'sentiment' column."
        )

    else:

        dashboard_data = dataset.copy()

        if "category" in dashboard_data.columns:

            categories = ["All"] + sorted(
                dashboard_data["category"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            selected_category = st.selectbox(
                "Filter by Product Category",
                categories
            )

            if selected_category != "All":

                dashboard_data = dashboard_data[
                    dashboard_data["category"].astype(str)
                    == selected_category
                ]

        total_reviews = len(dashboard_data)

        sentiment_counts = (
            dashboard_data["sentiment"].value_counts()
        )

        positive_count = int(
            dashboard_data["sentiment"]
            .astype(str)
            .str.lower()
            .eq("positive")
            .sum()
        )

        negative_count = int(
            dashboard_data["sentiment"]
            .astype(str)
            .str.lower()
            .eq("negative")
            .sum()
        )

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric(
            "Total Reviews",
            f"{total_reviews:,}"
        )

        metric2.metric(
            "Positive Reviews",
            f"{positive_count:,}"
        )

        metric3.metric(
            "Negative Reviews",
            f"{negative_count:,}"
        )

        left, right = st.columns(2)

        with left:

            st.markdown("### Sentiment Distribution")

            st.bar_chart(sentiment_counts)

        with right:

            st.markdown("### Sentiment Share")

            sentiment_share = (
                sentiment_counts
                / sentiment_counts.sum()
                * 100
            ).round(1)

            st.dataframe(
                sentiment_share.rename(
                    "Percentage"
                ).to_frame(),
                use_container_width=True
            )

        if "category" in dashboard_data.columns:

            st.markdown("### Sentiment by Product Category")

            category_sentiment = pd.crosstab(
                dashboard_data["category"],
                dashboard_data["sentiment"]
            )

            st.bar_chart(category_sentiment)

        if "product_name" in dashboard_data.columns:

            st.markdown("### Reviews by Product")

            product_counts = (
                dashboard_data["product_name"]
                .value_counts()
                .head(10)
            )

            st.bar_chart(product_counts)

        if "review_date" in dashboard_data.columns:

            dates = pd.to_datetime(
                dashboard_data["review_date"],
                errors="coerce"
            )

            date_data = dashboard_data.assign(
                review_month=dates.dt.to_period("M").astype(str)
            )

            date_data = date_data[
                date_data["review_month"] != "NaT"
            ]

            if not date_data.empty:

                st.markdown("### Review Trend Over Time")

                monthly_counts = (
                    date_data
                    .groupby(
                        ["review_month", "sentiment"]
                    )
                    .size()
                    .unstack(fill_value=0)
                )

                st.line_chart(monthly_counts)


# =========================================================
# 11. MODEL COMPARISON
# =========================================================

elif page == "Model Comparison":

    show_header(
        "Model Comparison",
        "Compare the performance of different text representations."
    )

    if comparison_results is None:

        st.warning(
            "Comparison results are not available yet. "
            "Run comparison.py from the project root to generate "
            "the comparison CSV."
        )

    else:

        required_metrics = {
            "Technique",
            "Accuracy",
            "Precision (Macro)",
            "Recall (Macro)",
            "F1 Score (Macro)"
        }

        if not required_metrics.issubset(
            comparison_results.columns
        ):

            st.error(
                "The comparison CSV does not contain "
                "all expected metric columns."
            )

        else:

            st.markdown("### Performance Comparison")

            st.dataframe(
                comparison_results.style.format(
                    {
                        "Accuracy": "{:.3f}",
                        "Precision (Macro)": "{:.3f}",
                        "Recall (Macro)": "{:.3f}",
                        "F1 Score (Macro)": "{:.3f}"
                    }
                ),
                use_container_width=True,
                hide_index=True
            )

            chart_data = (
                comparison_results
                .set_index("Technique")[
                    [
                        "Accuracy",
                        "Precision (Macro)",
                        "Recall (Macro)",
                        "F1 Score (Macro)"
                    ]
                ]
            )

            st.markdown("### Metric Comparison")

            st.bar_chart(chart_data)

            st.caption(
                "Word2Vec results represent the CBOW sentence-vector "
                "experiment from comparison.py. The application uses "
                "saved BoW and TF-IDF pipelines for individual and "
                "batch predictions."
            )

            st.download_button(
                "Download Comparison Results",
                data=comparison_results.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="representation_comparison.csv",
                mime="text/csv",
                type="primary"
            )


# =========================================================
# 12. FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div class="app-footer">
        Customer Sentiment Analysis | NLP Project<br>
        Predictions are model-generated and may not always
        reflect the full context or intent of a review.
    </div>
    """,
    unsafe_allow_html=True
)
