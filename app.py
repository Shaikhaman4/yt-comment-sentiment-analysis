import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer

# Load vectorizer and model
@st.cache_resource
def load_vectorizer_model():
    with open('tfidf_vectorizer.pkl', 'rb') as vec_file:
        vectorizer = pickle.load(vec_file)
    with open('lgbm_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return vectorizer, model

vectorizer, model = load_vectorizer_model()

# Mapping
label_map = {
    -1: "😠 Negative",
    0: "😐 Neutral",
    1: "😊 Positive"
}

# Page config
st.set_page_config(page_title="Sentiment Analysis App", page_icon="🧠", layout="wide")

# Custom styling
st.markdown("""
    <style>
    .main { background-color: #f9f9f9; }
    h1, h2, h3 { color: #004085; }
    .stButton>button { background-color: #4CAF50; color: white; }
    </style>
""", unsafe_allow_html=True)

# Sidebar info
with st.sidebar:
    st.header("🔧 Model Info")
    st.markdown("""
    - **Model:** LightGBM Classifier
    - **Vectorizer:** TF-IDF (1-3 grams)
    - **Classes:** -1: Negative, 0: Neutral, 1: Positive
    - **Author:** Aman Shaikh
    [📂 GitHub Repo](https://github.com/Shaikhaman4/yt-comment-sentiment-analysis)
    """)

st.title("🧠 Sentiment Analysis on Comments")
st.write("Enter a comment or upload a CSV to predict sentiment.")

# Sample comments
sample_comments = [
    "I love this video!",
    "Worst experience ever.",
    "It's okay, not great."
]
selected = st.selectbox("💡 Try a sample comment", sample_comments)
user_input = st.text_area("💬 Your Comment", value=selected)

# Session history
if 'history' not in st.session_state:
    st.session_state.history = []

# Predict on single input
if st.button("🔍 Analyze"):
    if user_input.strip():
        try:
            transformed_input = vectorizer.transform([user_input])
            pred = model.predict(transformed_input)[0]
            prob = model.predict_proba(transformed_input)[0]
            label = label_map.get(int(pred), "Unknown")
            st.success(f"Predicted Sentiment: **{label}** \n\n🔐 Confidence: `{max(prob) * 100:.2f}%`")
            st.session_state.history.append({"Comment": user_input, "Sentiment": label})
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.warning("Please enter a comment before analyzing.")

# Prediction history
if st.session_state.history:
    st.markdown("### 🕒 Prediction History")
    st.dataframe(pd.DataFrame(st.session_state.history))

# Batch prediction
st.markdown("---")
st.markdown("### 📄 Batch Prediction")
uploaded_file = st.file_uploader("Upload CSV file with 'comment' column")
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        if 'comment' not in df.columns:
            st.error("CSV must have a column named 'comment'")
        else:
            tfidf_comments = vectorizer.transform(df['comment'])
            df['prediction'] = model.predict(tfidf_comments)
            df['label'] = df['prediction'].apply(lambda x: label_map.get(x, 'Unknown'))
            st.dataframe(df[['comment', 'label']])
            st.download_button("📥 Download Results", df.to_csv(index=False), file_name="predictions.csv")
    except Exception as e:
        st.error(f"Failed to process file: {e}")

# Footer
st.markdown("""
---
Built with ❤️ by Aman Shaikh | LightGBM + TF-IDF + Streamlit
""")
