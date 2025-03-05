import streamlit as st
import numpy as np
import pickle  # To load the trained model

# Load the trained SVM model
with open("svm_model.pkl", "rb") as file:
    svm_model = pickle.load(file)

# Load feature names (all possible symptoms)
with open("symptom_list.pkl", "rb") as file:
    all_symptoms = pickle.load(file)

# Function to predict disease
def predict_disease(symptoms):
    input_data = np.zeros(len(all_symptoms))

    for symptom in symptoms:
        if symptom in all_symptoms:
            input_data[all_symptoms.index(symptom)] = 1

    prediction = svm_model.predict([input_data])
    return prediction[0]

# Streamlit UI
st.title("Medical Assistant - Disease Prediction")

# User selects symptoms
user_symptoms = st.multiselect("Select your symptoms", all_symptoms)

if st.button("Predict"):
    if user_symptoms:
        predicted_disease = predict_disease(user_symptoms)
        st.success(f"Predicted Disease: {predicted_disease}")
    else:
        st.warning("Please select at least one symptom.")
