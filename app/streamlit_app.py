import streamlit as st
from scripts.hybrid_predict import hybrid_prediction

st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detector (Hybrid Model)")
st.write("Enter a news article URL to check if it's real or fake.")

url = st.text_input("Paste news link:")

if st.button("Analyze"):
    with st.spinner("Analyzing... please wait ⏳"):
        result = hybrid_prediction(url)
    if "error" in result:
        st.error(f"Error: {result['error']}")
    else:
        st.subheader("Model Prediction")
        st.write(f"Prediction: **{result['model_prediction']}** (Confidence: {result['confidence']})")
        st.subheader("Web Verification")
        st.write(f"Credibility: **{result['credibility']}**")
        st.write("Top Verified Sources:")
        for src in result["verified_sources"]:
            st.write(f"- {src}")
