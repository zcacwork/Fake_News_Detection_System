import pandas as pd
import numpy as np
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# -----------------------------
# Step 1: Load Dataset
# -----------------------------
fake_df = pd.read_csv("fake.csv")
true_df = pd.read_csv("true.csv")

print("Fake News Samples:", fake_df.shape)
print("True News Samples:", true_df.shape)


# -----------------------------
# Step 2: Add Labels
# -----------------------------
fake_df["label"] = 0   # Fake
true_df["label"] = 1   # Real

# Combine datasets
df = pd.concat([fake_df, true_df], axis=0)

# Shuffle dataset
df = df.sample(frac=1, random_state=42)

print("Combined Dataset Shape:", df.shape)


# -----------------------------
# Step 3: Keep Important Columns
# -----------------------------
df = df[["title", "text", "label"]]

# Merge title + text
df["content"] = df["title"] + " " + df["text"]


# -----------------------------
# Step 4: Text Cleaning Function
# -----------------------------
def clean_text(text):
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


df["content"] = df["content"].apply(clean_text)


# -----------------------------
# Step 5: Features and Labels
# -----------------------------
X = df["content"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# -----------------------------
# Step 6: TF-IDF Vectorization
# -----------------------------
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# -----------------------------
# Step 7: Train Model
# -----------------------------
model = LogisticRegression()

model.fit(X_train_tfidf, y_train)


# -----------------------------
# Step 8: Evaluation
# -----------------------------
y_pred = model.predict(X_test_tfidf)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# -----------------------------
# Step 9: Custom Prediction
# -----------------------------
def predict_news(news_text):
    cleaned = clean_text(news_text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]

    if prediction == 0:
        return "Fake News"
    else:
        return "Real News"


print("\n------ Test Your Own News ------")
user_input = input("Enter news headline/article: ")

result = predict_news(user_input)

print("Prediction:", result)