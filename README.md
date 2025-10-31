
# 📧 Fake Mail Detector (Spam Classifier)

A **Machine Learning + NLP** based web app that detects whether an email is **Spam (Fake)** or **Genuine**.
Built using **Python, scikit-learn, NLTK, and Streamlit**.


## 🚀 Project Overview

This project classifies emails as **Spam** or **Ham (Genuine)** using Natural Language Processing (NLP) and Machine Learning.
The app provides a clean **Streamlit** interface where users can paste any email and instantly see whether it’s fake or not.


## 🧠 Tech Stack

| Category               | Technologies / Tools Used                                                                        |
| ---------------------- | ------------------------------------------------------------------------------------------------ |
| **Language**           | Python                                                                                           |
| **Libraries**          | pandas, numpy, scikit-learn, nltk, re, pickle, streamlit                                         |
| **ML Model**           | Multinomial Naive Bayes (or Logistic Regression)                                                 |
| **Text Vectorization** | TF-IDF Vectorizer                                                                                |
| **Web Framework**      | Streamlit                                                                                        |
| **Dataset**            | [Spam Email Dataset (Kaggle)](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) |


## 📁 Project Structure

```
FakeMailDetector/
│
├── app.py                   # Streamlit web app
├── model.pkl                # Trained spam classifier
├── vectorizer.pkl           # Saved TF-IDF vectorizer
├── spam.csv                 # Original dataset
├── cleaned_spam.csv         # Preprocessed data
├── requirements.txt         # Required libraries
└── README.md                # Project documentation
```

## ⚙️ Installation & Setup

### 1️⃣ Clone or Download Project

```
git clone https://github.com/yourusername/FakeMailDetector.git
cd FakeMailDetector
```

### 2️⃣ Install Dependencies

```
pip install -r requirements.txt
```

If you don’t have `requirements.txt`, manually install:

```
pip install pandas numpy scikit-learn nltk streamlit
```

### 3️⃣ Run Streamlit App

```
python -m streamlit run app.py


Then open browser → [http://localhost:8501](http://localhost:8501)


## 🧹 Data Preprocessing

Steps performed on the raw dataset:

1. Convert text to lowercase
2. Remove punctuation, numbers, and special symbols
3. Remove stopwords (using NLTK)
4. Tokenize and rejoin cleaned text
5. Drop missing or NaN rows

Example:

```python
text = "Congratulations! You have won $1000!!!"
cleaned_text = "congratulations won"
```

---

## 🧮 Model Training

1. Split dataset (80% train, 20% test)
2. Convert text → numerical features using `TfidfVectorizer`
3. Train using `MultinomialNB()`
4. Evaluate with `accuracy_score` and `classification_report`
5. Save model using `pickle`


model = MultinomialNB()
model.fit(X_train_vec, y_train)


## 🧩 Streamlit Web App

The web interface allows users to:

* Paste any email text
* Click **"Analyze"**
* Instantly get result: ✅ Genuine / 🚨 Spam

Example output:

```
🚨 This email looks like SPAM / FAKE!
```

or

```
✅ This email seems Genuine / Safe!
```

## 🧪 Example Test Inputs

| Example Type | Subject / Text                                           |
| ------------ | -------------------------------------------------------- |
| **Spam**     | “You’ve won ₹1,00,000! Click here to claim your reward.” |
| **Spam**     | “Urgent! Verify your bank account now.”                  |
| **Ham**      | “Can we reschedule tomorrow’s meeting?”                  |

---

## 📊 Model Performance

| Metric    | Score |
| --------- | ----- |
| Accuracy  | ~97%  |
| Precision | 0.96  |
| Recall    | 0.95  |

*(Depends on dataset split and preprocessing)*

---

## 📦 requirements.txt

```
pandas
numpy
scikit-learn
nltk
streamlit
```

---

## 🏁 Future Improvements

* Add email sender/domain analysis
* Use deep learning (BERT / LSTM)
* Multi-language email detection
* Integrate Gmail API for live scanning

---

## 🏆 Credits

* Dataset: UCI SMS Spam Collection (via Kaggle)
* Developer: Sayan Das
* Framework: Streamlit + scikit-learn

---