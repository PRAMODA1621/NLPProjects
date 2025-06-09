# train_and_save.py
import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

# Download NLTK stopwords if not already
nltk.download("stopwords")

# Load dataset
fake_path = "C:/Users/nppra/PycharmProjects/PythonProject3/archive (16)/News _dataset/Fake.csv"
real_path = "C:/Users/nppra/PycharmProjects/PythonProject3/archive (16)/News _dataset/True.csv"

fake_df = pd.read_csv(fake_path).sample(n=2000, random_state=42)
real_df = pd.read_csv(real_path).sample(n=2000, random_state=42)

# Add labels
fake_df['label'] = 0  # Fake
real_df['label'] = 1  # Real

# Combine and preprocess
df = pd.concat([fake_df, real_df], ignore_index=True)
df['content'] = df['title'] + " " + df['text']

# Clean text
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return " ".join(words)

df['cleaned'] = df['content'].apply(clean_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=5000)
X = vectorizer.fit_transform(df['cleaned'])
y = df['label']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
