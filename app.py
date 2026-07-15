import streamlit as st
import joblib
import re

# Set page configuration
st.set_page_config(
    page_title="Emergency Management System - Disaster Classifier",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 1. Core Pipeline Ingestion
@st.cache_resource
def load_pipeline_artifacts():
    try:
        model = joblib.load('best_disaster_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        st.error("Execution Failure: Pipeline binaries not detected.")
        return None, None

def clean_input_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

model, vectorizer = load_pipeline_artifacts()

# 2. User Interface Design
st.title("Disaster Tweet Classification Engine")
st.markdown("""
### Enterprise-Grade Operational Monitoring Interface
This system utilizes an optimized NLP framework to parse text parameters and differentiate true emergencies from metaphorical context.
---
""")

# Wrapping inside a Form guarantees execution on button click
with st.form(key='prediction_form'):
    user_input = st.text_area(
        "Ingest Tweet Contents Here:",
        placeholder="Type or paste real-time transmission log data...",
        height=150
    )
    
    # Form submit button
    submit_button = st.form_submit_button(label="Execute Predictive Inference Pipeline")

# Inference execution logic on submit
if submit_button:
    if user_input.strip() == "":
        st.warning("Inference Halted: Empty textual fields detected.")
    else:
        processed_input = clean_input_text(user_input)
        vectorized_input = vectorizer.transform([processed_input])
        
        prediction = model.predict(vectorized_input)[0]
        prediction_proba = model.predict_proba(vectorized_input)[0]
        
        st.markdown("### Inference Diagnostics Output")
        
        if prediction == 1:
            confidence = prediction_proba[1] * 100
            st.error(f"🚨 CRITICAL ALERT: Actionable Disaster Event Verified (Confidence Score: {confidence:.2f}%)")
            st.markdown("**System Log Notification:** High risk parameters detected. Event flagged for urgent emergency routing.")
        else:
            confidence = prediction_proba[0] * 100
            st.success(f"✅ STABLE: Non-Emergency / Contextual Sentiment Verified (Confidence Score: {confidence:.2f}%)")
            st.markdown("**System Log Notification:** Baseline text logs analyzed. Context indicates nominal semantic features.")
