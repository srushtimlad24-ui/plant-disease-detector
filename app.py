import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from classes import PLANT_CLASSES

# Set layout parameters
st.set_page_config(page_title="Plant Disease Predictor", page_icon="🌱", layout="centered")

# Cache model loading so it doesn't slow down the site on every button click
@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model('plant_disease_mobilenetv2.h5')

try:
    model = load_trained_model()
except Exception as e:
    st.error("Error loading model file. Ensure 'plant_disease_mobilenetv2.h5' is in the same folder directory.")
    st.stop()

# Header text layout
st.title("🌱 Intelligent Plant Leaf Disease Detector")
st.write("Upload a crisp image of a crop leaf below, and our pre-trained network will detect its health status.")

# Upload widget
uploaded_file = st.file_uploader("Upload leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Render user's image onto dashboard
    user_image = Image.open(uploaded_file)
    st.image(user_image, caption='Uploaded Leaf Image', use_column_width=True)
    st.write("🔄 Analyzing visual markers...")
    
    # Standardize image array to match original model pipeline conditions
    resized_img = user_image.resize((224, 224))
    image_array = np.array(resized_img)
    
    # Strip alpha transparency channel if present
    if image_array.shape[-1] == 4:
        image_array = image_array[:, :, :3]
        
    # Standardize scale down to 0-1 metrics and inject batch dimensions
    image_array = image_array.astype('float32') / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    # Execute network prediction matrix
    raw_predictions = model.predict(image_array)
    top_index = np.argmax(raw_predictions[0])
    confidence_score = raw_predictions[0][top_index] * 100
    
    # Fetch prediction string and polish text appearance
    assigned_label = PLANT_CLASSES[top_index]
    presentable_name = assigned_label.replace("___", " - ").replace("_", " ")
    
    # Display analytics dashboard banner output
    st.success("🤖 Analysis complete!")
    
    # Check if the plant is healthy to fire off celebration balloons
    if "healthy" in presentable_name.lower():
        st.balloons()
        
    # Render the output layout to match the target layout perfectly
    st.markdown(f"### Prediction: `{presentable_name}`")
    
    st.write("**Confidence Score**")
    st.subheader(f"{confidence_score:.2f}%")
    st.progress(int(confidence_score))
