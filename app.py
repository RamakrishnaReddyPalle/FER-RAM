import streamlit as st
import cv2
import numpy as np
from src.inference_pipeline.prediction import ExpressionPredictor

MODEL_PATH = 'src/models/expw_best_model.pth'
predictor = ExpressionPredictor(MODEL_PATH)

st.set_page_config(page_title="Facial Expression Identifier")
st.title("Real-time Emotion Detection")

input_source = st.radio("Select Input Source:", ("Upload Image", "Webcam"))

if input_source == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
else:
    webcam_image = st.camera_input("Take a picture with webcam")
    if webcam_image:
        image = cv2.imdecode(np.frombuffer(webcam_image.getvalue(), np.uint8), 1)

if 'image' in locals():
    predictions = predictor.predict(image)
    
    if not predictions:
        st.warning("No faces detected!")
    else:
        cols = st.columns(len(predictions))
        for i, (col, pred) in enumerate(zip(cols, predictions)):
            with col:
                st.image(cv2.cvtColor(pred['face'], cv2.COLOR_BGR2RGB), 
                        caption=f"Face {i+1}")
                st.success(f"**{pred['emotion']}**\nConfidence: {pred['confidence']*100:.1f}%")
