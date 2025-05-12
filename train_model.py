print("🚀 Training script started...")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Load Datasets
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Add Labels
fake["label"] = 0
true["label"] = 1

# Combine and Shuffle
data = pd.concat([fake, true])
data = data.sample(frac=1).reset_index(drop=True)

# Features & Labels
X = data['text']
y = data['label']

# Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vect = vectorizer.fit_transform(X)

# Model
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model and vectorizer
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

print("✅ Model trained and saved successfully.")
