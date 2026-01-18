import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')


df = pd.read_csv("/Users/pradyun/Downloads/File (2).csv")
df.head()
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [w for w in tokens if w.isalnum() and w not in stop_words]
    tokens = [stemmer.stem(w) for w in tokens]
    return ' '.join(tokens)

df_user = (df.groupby(['speaker_first_name','house'])['dialogue'].apply(lambda x: ' '.join(x.fillna('').astype(str))).reset_index())
df_user['clean_text'] = df_user['dialogue'].apply(preprocess)
X = df_user['clean_text']
y = df_user['house']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.feature_extraction.text import TfidfVectorizer


tfidf = TfidfVectorizer(max_features=5000,ngram_range=(1,2))

X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)
from sklearn.svm import LinearSVC

model = LinearSVC(class_weight='balanced')
model.fit(X_train_vec, y_train)
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))

import pickle
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "house_classifier.pkl"

with open(MODEL_PATH, "wb") as f:
    pickle.dump((model, tfidf), f)

print("Saved house_classifier.pkl")
