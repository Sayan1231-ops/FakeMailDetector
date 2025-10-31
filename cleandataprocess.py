import re
import nltk
import pandas as pd
from nltk.corpus import stopwords

# Download stopwords (only once)
nltk.download('stopwords')

# 🧹 Function to clean text
def clean_text(text):
    # 1️⃣ Lowercase everything
    text = str(text).lower()

    # 2️⃣ Remove punctuation, numbers, and special chars
    text = re.sub(r'[^a-z\s]', '', text)

    # 3️⃣ Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]

    # 4️⃣ Join words back
    cleaned_text = " ".join(filtered_words)

    return cleaned_text

# 🧾 Load dataset
data = pd.read_csv("spam.csv", encoding='latin-1')

# Drop unnecessary unnamed columns if they exist
data = data[['v1', 'v2']]

# Rename columns for clarity
data.columns = ['label', 'message']

# 🧼 Apply cleaning function
data['cleaned_message'] = data['message'].apply(clean_text)

# ✅ Show sample output
print(data.head(10))

# Save cleaned dataset
data.to_csv('cleaned_spam.csv', index=False)
