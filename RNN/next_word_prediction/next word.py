# next_word_app.py

import streamlit as st
import tensorflow as tf
import numpy as np

# ✅ Must be the very first Streamlit command
st.set_page_config(page_title="Next Word Predictor", layout="wide")

# App title
st.title("Next Word Predictor")
st.write("Enter a sentence and the model will predict the next word.")

# Input text from user
user_input = st.text_input("Type your sentence here:")

# Dummy model example (replace with your actual trained model)
# For demonstration, let's simulate predictions
def predict_next_word(sentence):
    # Here you would normally use your trained model
    dummy_words = ["is", "the", "a", "and", "to"]
    return np.random.choice(dummy_words)

# Show prediction when user enters text
if user_input:
    next_word = predict_next_word(user_input)
    st.success(f"Next word prediction: **{next_word}**")

