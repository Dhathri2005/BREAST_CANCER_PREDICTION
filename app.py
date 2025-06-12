import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

# Load model and scaler
model = tf.keras.models.load_model("./breast_cancer_nn_model.h5")
scaler = joblib.load("./scaler.pkl")

# Feature list (30 features from breast cancer dataset)
FEATURE_NAMES = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error',
    'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
    'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

# App title and description
st.set_page_config(page_title="Breast Cancer Classifier", layout="centered")
st.title("Breast Cancer Tumor Classification")

# Input method
input_method = st.radio("Choose Input Method:", ["📝 Manual Entry", "📁 Upload CSV"])

# Manual input form
def get_manual_input():
    inputs = []
    st.info("Enter the 30 features below:")
    for feature in FEATURE_NAMES:
        val = st.number_input(f"{feature}", value=0.0, format="%.4f")
        inputs.append(val)
    return np.array([inputs])

# CSV input
def get_csv_input(uploaded_file):
    import pandas as pd
    try:
        df = pd.read_csv(uploaded_file)
        if df.shape[1] != 30:
            st.error("CSV must contain exactly 30 columns.")
            return None
        return df.values
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return None

# Get user input
if input_method == "📝 Manual Entry":
    input_data = get_manual_input()
else:
    uploaded_file = st.file_uploader("Upload a CSV file with 30 features", type=["csv"])
    input_data = get_csv_input(uploaded_file) if uploaded_file else None

# Prediction
if input_data is not None and st.button("🔍 Predict"):
    input_scaled = scaler.transform(input_data)
    predictions = model.predict(input_scaled)
    predicted_labels = [np.argmax(p) for p in predictions]

    for i, pred in enumerate(predicted_labels):
        tumor_type = "🧬 **Malignant (0)**" if pred == 0 else "✅ **Benign (1)**"
        st.success(f"Sample {i+1}: {tumor_type}")
        st.write(f"**Confidence:** Malignant: `{predictions[i][0]:.2f}`, Benign: `{predictions[i][1]:.2f}`")
