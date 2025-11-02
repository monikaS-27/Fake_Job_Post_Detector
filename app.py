# app.py
import streamlit as st
import joblib
import re
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

MODEL_DIR = "model"
model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf.pkl"))

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r'http\S+|www\S+',' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = text.lower().split()
    text = [lemmatizer.lemmatize(w) for w in text if w not in stop_words and len(w) > 2]
    return ' '.join(text)

def predict_job_post(text):
    cleaned = clean_text(text)
    vec = tfidf.transform([cleaned])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec).max()
    return pred, prob

st.set_page_config(page_title="Fake Job Post Detector", page_icon="💼")
st.title("💼 Fake Job Post Detector")

st.markdown("Paste a job posting (title + description + requirements). The model predicts if it's likely fake or real.")

txt = st.text_area("Job posting text", height=250)

if st.button("Analyze"):
    if not txt.strip():
        st.warning("Enter a job posting to analyze.")
    else:
        pred, prob = predict_job_post(txt)
        if pred == 1:
            st.error(f"🔴 Likely FAKE (confidence {prob:.2f})")
        else:
            st.success(f"🟢 Likely REAL (confidence {prob:.2f})")

st.markdown("---")
st.markdown("Created by Monika S — TF-IDF + LogisticRegression")
