import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

st.title("📰 Fake News Detection App")
st.write("Enter a news article below to check if it's **Fake** or **Real**.")

news = st.text_area("News Article Text")

if st.button("Predict"):
    if news.strip() == "":
        st.warning("Please enter some news content!")
    else:
        transformed = vectorizer.transform([news])
        prediction = model.predict(transformed)

        if prediction[0] == 1:
            st.success("✅ This news is REAL.")
        else:
            st.error("❌ This news is FAKE.")
