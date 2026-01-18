import pandas as pd
import pickle
from pathlib import Path

import nltk
nltk.download("punkt")
nltk.download("stopwords")

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# ============================================================
# LOAD DATA
# ============================================================
DATA_PATH = "/Users/pradyun/Downloads/train.csv"

df = pd.read_csv(DATA_PATH, low_memory=False)

# Keep genres with at least 2 samples
df = df.groupby("Genre").filter(lambda x: len(x) >= 2)

# Drop missing values
df = df.dropna(subset=["Lyrics", "Genre"])

X = df["Lyrics"].astype(str)
y = df["Genre"]

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ============================================================
# TF-IDF VECTORIZER
# ============================================================
tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=100000,
    ngram_range=(1, 2),
    min_df=5
)

X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# ============================================================
# MODEL
# ============================================================
model = LinearSVC(class_weight="balanced")
model.fit(X_train_vec, y_train)

# ============================================================
# EVALUATION
# ============================================================
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))

# ============================================================
# SAVE MODEL (FOR STREAMLIT)
# ============================================================
MODEL_PATH = Path(__file__).parent / "genre_classifier.pkl"

with open(MODEL_PATH, "wb") as f:
    pickle.dump((model, tfidf), f)

print(f"✅ Model saved to {MODEL_PATH}")
