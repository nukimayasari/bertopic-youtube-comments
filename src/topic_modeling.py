import os
import numpy as np
import pandas as pd
import re
import emoji

from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# -----------------------
# Load data
# -----------------------
df = pd.read_csv("comments_sumbufilosofi.csv")
df1 = df.copy()


# -----------------------
# Cleaning function
# -----------------------
def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


df1["clean_text"] = df1["text"].apply(clean_text)

documents = df1["clean_text"].tolist()


# -----------------------
# Embedding model
# -----------------------
embedding_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

embeddings = embedding_model.encode(
    documents,
    show_progress_bar=True
)


# -----------------------
# BERTopic model
# -----------------------
topic_model = BERTopic(
    language="multilingual",
    min_topic_size=30,
    verbose=True
)

topics, probs = topic_model.fit_transform(documents, embeddings)

df1["topic"] = topics


# -----------------------
# Save outputs
# -----------------------
df1.to_csv("comments_with_topics.csv", index=False)

topic_info = topic_model.get_topic_info()
topic_info.to_csv("topic_info.csv", index=False)


# -----------------------
# Silhouette score
# -----------------------
mask = np.array(topics) != -1

score = silhouette_score(
    embeddings[mask],
    np.array(topics)[mask],
    metric="cosine"
)

print("Silhouette score:", score)


# -----------------------
# Predict engagement
# -----------------------
X = pd.get_dummies(df1["topic"])
y = (df1["likes"] > df1["likes"].median()).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

preds = clf.predict(X_test)
acc = accuracy_score(y_test, preds)

print("Logistic Regression accuracy:", acc)
