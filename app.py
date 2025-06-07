import streamlit as st
import joblib

# Load the model and vectorizer
model = joblib.load("models/model.pkl")  # your saved model
vectorizer = joblib.load("models/vectorizer.pkl")  # optional, if used

# Streamlit app layout
st.set_page_config(page_title="Feedback Sentiment Predictor", layout="centered")

st.title("📝 Feedback Sentiment Classifier")
st.subheader("Enter your feedback below to check if it's Positive or Negative")

# Input text
feedback = st.text_area("✍️ Feedback", placeholder="Type the feedback here...")

# Predict button
if st.button("Predict Sentiment"):
    if feedback.strip() == "":
        st.warning("Please enter some feedback text.")
    else:
        # Preprocess and predict
        input_vector = vectorizer.transform([feedback])
        prediction = model.predict(input_vector)

        # Display result
        sentiment = "Positive 😊" if prediction[0] == 1 else "Negative 😠"
        st.success(f"Predicted Sentiment: **{sentiment}**")
