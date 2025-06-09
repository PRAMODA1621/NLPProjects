# app.py
import streamlit as st
import joblib
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# Setup
nltk.download("stopwords")

# Load trained model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return " ".join(words)

# Streamlit UI
st.title("📰 Fake News Detector")
st.markdown("Enter a news headline or story and detect whether it's **Real** or **Fake**.")

user_input = st.text_area("Enter News Text", height=200)

if st.button("Detect"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        input_vector = vectorizer.transform([cleaned])
        prediction = model.predict(input_vector)[0]
        if prediction == 1:
            st.success("🟢 This is Real News")
        else:
            st.error("🔴 This is Fake News")
