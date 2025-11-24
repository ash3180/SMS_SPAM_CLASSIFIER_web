import streamlit as st
import pickle
import string

# Load vectorizer and model
tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

# simple text cleaning
def clean_text(text):
    text = text.lower()
    text = "".join([c for c in text if c not in string.punctuation])
    return text

st.title(" SMS Spam Classifier")


input_sms = st.text_area("Type your message here...")

if st.button("Predict"):
    if input_sms.strip() == "":
        st.warning("Please enter a message!")
    else:
        transformed = clean_text(input_sms)
        vector_input = tfidf.transform([transformed])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.error(" Spam Message")
        else:
            st.success(" Not Spam")
