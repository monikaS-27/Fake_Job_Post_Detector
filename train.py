import pandas as pd
import re
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import os
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Resolve path to dataset robustly: prefer data/fake_job_postings.csv but fall back to
# fake_job_postings.csv in the repository root or current working directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
candidate_paths = [
    os.path.join(SCRIPT_DIR, "data", "fake_job_postings.csv"),
    os.path.join(SCRIPT_DIR, "fake_job_postings.csv"),
    os.path.join(SCRIPT_DIR, "..", "data", "fake_job_postings.csv"),
    os.path.join(os.getcwd(), "data", "fake_job_postings.csv"),
    os.path.join(os.getcwd(), "fake_job_postings.csv"),
]
DATA_PATH = next((p for p in candidate_paths if os.path.exists(p)), None)
if DATA_PATH is None:
    tried = "\n".join(candidate_paths)
    nearby = "\n".join(os.listdir(SCRIPT_DIR))
    raise FileNotFoundError(
        f"Could not find 'fake_job_postings.csv'. Tried these locations:\n{tried}\n\n"
        f"Files in script directory ({SCRIPT_DIR}):\n{nearby}"
    )
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
print("Loaded:", df.shape)

cols = ['title','company_profile','description','requirements','benefits','fraudulent']
existing = [c for c in cols if c in df.columns]
df = df[existing]

df = df.fillna('')
df['text'] = (df.get('title','') + ' ' + df.get('company_profile','') + ' ' +
              df.get('description','') + ' ' + df.get('requirements','') + ' ' +
              df.get('benefits','')).str.strip()
df = df[df['text'].str.len() > 10].reset_index(drop=True)
print("After filter:", df.shape)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+|www\S+',' ', text)   
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)     
    text = text.lower().split()
    text = [lemmatizer.lemmatize(w) for w in text if w not in stop_words and len(w) > 2]
    return ' '.join(text)

print("Cleaning text...")
df['clean_text'] = df['text'].apply(clean_text)

X = df['clean_text']
y = df['fraudulent'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", cm)

plt.figure(figsize=(5,4))
plt.imshow(cm, interpolation='nearest')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"))

joblib.dump(model, os.path.join(MODEL_DIR, "model.pkl"))
joblib.dump(tfidf, os.path.join(MODEL_DIR, "tfidf.pkl"))
print("Saved model and vectorizer to", MODEL_DIR)
