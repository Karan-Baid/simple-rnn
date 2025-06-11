import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence

word_index=imdb.get_word_index()
reversed_word_index={value:key for key, value in word_index.items()}

model=load_model('simple_rnn_imdb.h5')

def decode_review(encoded_review):
    return ' '.join([reversed_word_index.get(i-3,'?') for i  in encoded_review])

def preprocess_text(text):
    words= text.lower().split()
    encoded_review = [word_index.get(word, 2) +3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment (positive or negative):")

review_input = st.text_area("Movie Review", height=200)

if st.button("Predict Sentiment"):
    preprocessed_input=preprocess_text(review_input)
    prediction = model.predict(preprocessed_input)
    sentiment ='Positive' if prediction[0][0] > 0.5 else 'Negative'

    st.write(f"Sentiment: {sentiment}")
    st.write(f"Prediction Score: {prediction[0][0]:.4f}")

else:
    st.write("Please enter a movie review")