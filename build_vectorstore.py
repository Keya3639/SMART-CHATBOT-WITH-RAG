import os
import pandas as pd
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
df = pd.read_csv("data/rag.csv")

# Combine relevant columns (adjust if needed)
texts = df.astype(str).agg(" ".join, axis=1).tolist()

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X = vectorizer.fit_transform(texts).toarray().astype("float32")

# FAISS index
index = faiss.IndexFlatL2(X.shape[1])
index.add(X)

# Save
os.makedirs("vectorstore", exist_ok=True)
faiss.write_index(index, "vectorstore/tfidf.index")

# Save vectorizer
import pickle
with open("vectorstore/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ TF-IDF Vector Store created successfully")
