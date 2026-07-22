import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from classes import PLANT_CLASSES
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.set_page_config(
    page_title="Plant Disease Predictor",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Styling ----
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

.stApp {
    background-color: #f6f8f5;
}

div[data-testid="stMetric"] {
    background-color: white;
    padding: 16px;
    border-radius: 12px;
    border: 1px solid #e3e8e0;
}

section[data-testid="stSidebar"] {
    background-color: #ffffff;
}
</style>
""", unsafe_allow_html=True)


# Cache model loading so it doesn't slow down the site on every button click
@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model('plant_disease_mobilenetv2.h5')


try:
    model = load_trained_model()
except Exception:
    st.error("Error loading model file. Ensure 'plant_disease_mobilenetv2.h5' is in the same folder directory.")
    st.stop()

# ---- Sidebar ----
with st.sidebar:
    st.markdown("## 🌱 Plant doctor")
    st.write("Upload a leaf photo and this model will flag likely diseases from visual patterns.")
    st.divider()
    uploaded_file = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png"])
    st.divider()
    with st.expander("About this model"):
        st.write(
            "Built on MobileNetV2 and fine-tuned on labeled leaf images. "
            "Predictions are a starting point, not a substitute for expert diagnosis."
        )

# ---- Header ----
st.title("Intelligent plant leaf disease detector")
st.caption("Deep learning based leaf disease screening")

if uploaded_file is None:
    st.info("Upload a leaf image from the sidebar to get started.")
    st.stop()

user_image = Image.open(uploaded_file)

with st.spinner("Analyzing visual markers..."):
    # Standardize image array to match original model pipeline conditions
    resized_img = user_image.resize((224, 224))
    image_array = np.array(resized_img)

    # Strip alpha transparency channel if present
    if image_array.shape[-1] == 4:
        image_array = image_array[:, :, :3]

    # Inject batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Authentic MobileNetV2 scale (-1 to 1) instead of dividing by 255.0
    processed_image = preprocess_input(image_array.astype('float32'))

    raw_predictions = model.predict(processed_image)[0]
    top_index = int(np.argmax(raw_predictions))
    confidence_score = float(raw_predictions[top_index] * 100)

    assigned_label = PLANT_CLASSES[top_index]
    presentable_name = assigned_label.replace("___", " - ").replace("_", " ")
    is_healthy = "healthy" in presentable_name.lower()

# ---- Results layout ----
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.image(user_image, caption="Uploaded leaf image", use_container_width=True)

with col2:
    if is_healthy:
        st.success(f"✅ {presentable_name}")
    else:
        st.error(f"✅ {presentable_name}")

    st.metric("Confidence", f"{confidence_score:.1f}%")
    st.progress(min(max(int(confidence_score), 0), 100))

if is_healthy:
    st.balloons()

# ---- Full breakdown ----
with st.expander("See full prediction breakdown"):
    top_n = min(5, len(raw_predictions))
    top_indices = np.argsort(raw_predictions)[::-1][:top_n]
    breakdown = {
        PLANT_CLASSES[i].replace("___", " - ").replace("_", " "): float(raw_predictions[i] * 100)
        for i in top_indices
    }
    st.bar_chart(breakdown)
