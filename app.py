import streamlit as st
import joblib

from preprocessing import preprocess_text


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="News Classifier",
    page_icon="📰",
    layout="centered"
)


# =========================================================
# LOAD MODEL AND VECTORIZER
# =========================================================

@st.cache_resource
def load_model():

    tfidf = joblib.load("tf_idf_vectorizer.pkl")
    model = joblib.load("svm_model.pkl")

    return tfidf, model


tfidf, model = load_model()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <h1 style="text-align:center;">📰</h1>
        <h2 style="text-align:center;">News Classifier</h2>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("### 📌 About")

    st.write(
        """
        This application classifies news articles into
        four different categories using Natural Language
        Processing and Machine Learning.
        """
    )

    st.markdown("---")

    st.markdown("### 🤖 Model")

    st.write("**Algorithm:** Linear SVM")
    st.write("**Feature Extraction:** TF-IDF")
    st.write("**Preprocessing:** NLP Text Cleaning")

    st.markdown("---")

    st.markdown("### 🗂️ Categories")

    st.write("🌍 **World**")
    st.write("🏏 **Sports**")
    st.write("💼 **Business**")
    st.write("💻 **Sci/Tech**")

    st.markdown("---")

    st.markdown("### 🔄 Pipeline")

    st.code(
        """Raw Text
    ↓
Preprocessing
    ↓
TF-IDF
    ↓
Linear SVM
    ↓
Prediction""",
        language=""
    )

    st.markdown("---")

    st.caption(
        "Built with Python • Scikit-learn • Streamlit"
    )


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 900px;
        padding-top: 3rem;
        padding-bottom: 2rem;
    }

    /* Main title */
    .main-title {
        text-align: center;
        font-size: 46px;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 5px;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #9ca3af;
        margin-bottom: 35px;
    }

    /* Information card */
    .info-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 30px;
        color: #e2e8f0;
        font-size: 16px;
        line-height: 1.7;
    }

    .info-title {
        color: #f8fafc;
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    /* Prediction card */
    .prediction-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 30px;
        margin-top: 30px;
        text-align: center;
    }

    .prediction-title {
        color: #94a3b8;
        font-size: 16px;
        margin-bottom: 8px;
    }

    .prediction-result {
        color: #f8fafc;
        font-size: 34px;
        font-weight: 700;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 14px;
        margin-top: 35px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">📰 News Classifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Classify news articles into World, Sports, Business, or Sci/Tech'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# HOW IT WORKS
# =========================================================

st.markdown(
    '<div class="info-card">'
    '<div class="info-title">⚙️ How It Works</div>'
    'The entered news article is first processed using NLP '
    'text preprocessing. The cleaned text is then transformed '
    'into TF-IDF features and classified using a trained '
    'Linear SVM model.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# NEWS INPUT
# =========================================================

st.markdown("### ✍️ Enter News Article")

news_text = st.text_area(
    "Paste your news article below:",
    height=220,
    placeholder="Paste the news article here..."
)


# =========================================================
# CLASSIFICATION
# =========================================================

if st.button(
    "🔍 Classify News",
    use_container_width=True
):

    if not news_text.strip():

        st.warning("⚠️ Please enter a news article first.")

    else:

        with st.spinner("Analyzing the article..."):

            # Preprocessing
            cleaned_text = preprocess_text(news_text)

            # TF-IDF transformation
            text_vector = tfidf.transform([cleaned_text])

            # Prediction
            prediction = model.predict(text_vector)[0]


        # =================================================
        # LABEL MAPPING
        # =================================================

        label_mapping = {
            0: "🌍 World",
            1: "🏏 Sports",
            2: "💼 Business",
            3: "💻 Sci/Tech"
        }

        predicted_category = label_mapping[prediction]


        # =================================================
        # DISPLAY RESULT
        # =================================================

        st.markdown(
            '<div class="prediction-card">'
            '<div class="prediction-title">'
            'Predicted Category'
            '</div>'
            f'<div class="prediction-result">'
            f'{predicted_category}'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    '<div class="footer">'
    'Built with Python • NLP • TF-IDF • Linear SVM • Streamlit'
    '</div>',
    unsafe_allow_html=True
)