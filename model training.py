import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pickle

# 1️⃣ Load cleaned dataset
data = pd.read_csv("cleaned_spam.csv")

# Ensure correct columns exist
data = data[['label', 'cleaned_message']]
data.columns = ['label', 'message']

# 🧹 Drop rows where message is NaN or empty
data = data.dropna(subset=['message'])
data = data[data['message'].str.strip() != ""]

# Convert labels to numeric
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# 2️⃣ Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42
)

# 3️⃣ Vectorize text
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 4️⃣ Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 5️⃣ Evaluate
y_pred = model.predict(X_test_vec)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# 6️⃣ Save model & vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
print("💾 Model & Vectorizer saved successfully!")
