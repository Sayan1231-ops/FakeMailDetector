import streamlit as st
import pickle
import re

# 1️ Load the trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# 2️ Simple text cleaning function (same as training phase)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# 3️ Streamlit UI
st.set_page_config(page_title="Fake Mail Detector", page_icon="📧", layout="centered")

st.sidebar.title("👨‍💻 About the Developer")
st.sidebar.markdown("""
**Name:** *Sayan Das*  
**Role:** Aspiring AI Engineer 💡  
**Email:** sayan70724@gmail.com
**LinkedIn:** https://www.linkedin.com/in/sayan-das-1891a5259 
**GitHub:** https://github.com/Sayan1231-ops
""")

st.title("📧 Fake Mail Detector (AI Powered)")
st.write("Type or paste an email message below to check if it's **Spam or Genuine** 🧠")

# 4️ Input box
user_input = st.text_area("✉️ Enter your email/message:", height=250)

# 5️ Predict button
if st.button("🔍 Detect"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        # Clean + vectorize user input
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        # Show result
        if prediction == 1:
            st.error("🚨 This looks like a **Fake / Spam Email!**")
        else:
            st.success("✅ This looks like a **Genuine / Real Email.**")

# 6️ Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit & Scikit-learn")
st.caption("Code available on [GitHub](https://github.com/iamshaunjp/fake-mail-detector)")
st.caption("Documentation available on [GitHub](https://github.com/iamshaunjp/fake-mail-detector)")
st.caption("© 2025 Fake Mail Detector | Created by Sayan Das")