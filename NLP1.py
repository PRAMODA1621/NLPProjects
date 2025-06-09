import streamlit as st
import joblib
import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,accuracy_score
fake_path = "C:/Users/nppra/PycharmProjects/PythonProject3/archive (16)/News _dataset/Fake.csv"
real_path = "C:/Users/nppra/PycharmProjects/PythonProject3/archive (16)/News _dataset/True.csv"

fake_df = pd.read_csv(fake_path).sample(n=2000,random_state=42)
real_df = pd.read_csv(real_path).sample(n=2000,random_state=42)

# Add labels
fake_df['label'] = 0  # Fake
real_df['label'] = 1  # Real

# Combine datasets
df = pd.concat([fake_df, real_df], ignore_index=True)

# Combine title and text into one column
df['content'] = df['title'] + " " + df['text']

stop_words=set(stopwords.words("english"))
stemmer=PorterStemmer()
def clean_text(text):
    text=text.lower()
    text=re.sub(r'[^a-z\s]',' ',text)
    words=text.split()
    words=[stemmer.stem(w) for w in words if w not in stop_words]
    return " ".join(words)
df['cleaned'] = df['content'].apply(clean_text)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer(ngram_range=(1,3),max_features=5000)
X=vectorizer.fit_transform(df['cleaned'])

X_train,X_test,Y_train,Y_test=train_test_split(X,df['label'],test_size=0.2,random_state=42)
model=LogisticRegression()
model.fit(X_train,Y_train)
y_pred=model.predict(X_test)
print("Accuracy : ",accuracy_score(Y_test,y_pred),flush=True)
while True:
    user_input=input("\nEnter a news sentence(or type 'n' to quit)")
    if user_input.lower()=='n':
        break
    cleaned_input=clean_text(user_input)
    input_vector=vectorizer.transform([cleaned_input])
    prediction=model.predict(input_vector)[0]
    print("Prediction:", "🟢 Real News" if prediction == 1 else "🔴 Fake News")
print(df['label'].value_counts())