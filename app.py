import streamlit as st
import tensorflow as tf
import numpy as np
import random
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
    color: #262626;
}

div[data-testid="stMetric"] {
    background-color: white;
    padding: 16px;
    border-radius: 12px;
    border: 1px solid #e3e8e0;
}

section[data-testid="stSidebar"] {
    background-color: #ffffff;
    color: #262626;
}

header[data-testid="stHeader"] {
    background: rgba(0, 0, 0, 0);
}
</style>
""", unsafe_allow_html=True)


# Lightweight keyword-based care tips, checked against the predicted label
CARE_TIPS = {
    "healthy": "No treatment needed. Keep up good watering habits, adequate spacing, and regular monitoring.",
    "rust": "Remove nearby host plants if possible, prune infected leaves, and apply a fungicide labeled for rust in early spring.",
    "blight": "Remove and destroy infected leaves, avoid overhead watering, rotate crops next season, and apply a copper-based fungicide.",
    "spot": "Prune for better airflow, avoid wetting foliage when watering, and apply a fungicide labeled for leaf spot.",
    "mildew": "Increase air circulation, avoid overhead watering, and apply a sulfur or potassium-bicarbonate based fungicide.",
    "mold": "Improve ventilation, reduce humidity around the plant, and remove affected leaves promptly.",
    "scab": "Rake and dispose of fallen leaves, prune for airflow, and apply a fungicide at bud break next season.",
    "virus": "There is no cure for viral infections. Remove and destroy the infected plant to prevent spread.",
    "bacterial": "Remove infected leaves with sterilized tools, avoid overhead watering, and consider a copper-based bactericide.",
}


def get_care_tip(label: str) -> str:
    lowered = label.lower()
    for keyword, tip in CARE_TIPS.items():
        if keyword in lowered:
            return tip
    return "Isolate the affected plant if possible and consult your local agricultural extension for a tailored treatment plan."


def show_emoji_burst(emoji: str, count: int = 16):
    spans = "".join(
        f'<span style="left:{random.randint(2, 96)}%; '
        f'animation-delay:{random.uniform(0, 0.6):.2f}s; '
        f'font-size:{random.randint(20, 36)}px;">{emoji}</span>'
        for _ in range(count)
    )
    st.markdown(f"""
        <style>
        .emoji-burst {{
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            pointer-events: none;
            overflow: hidden;
            z-index: 9999;
        }}
        .emoji-burst span {{
            position: absolute;
            top: 100%;
            animation: emoji-rise 2.2s ease-out forwards;
        }}
        @keyframes emoji-rise {{
            0% {{ transform: translateY(0) rotate(0deg); opacity: 1; }}
            100% {{ transform: translateY(-100vh) rotate(360deg); opacity: 0; }}
        }}
        </style>
        <div class="emoji-burst">{spans}</div>
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
st.title("🌿Intelligent plant leaf disease detector")
st.caption("Spotting plant disease with computer vision")

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
        accent = "#2e7d32"
        badge_bg = "#e8f5e9"
        icon = "✅"
        badge_label = "Healthy"
    else:
        accent = "#c2410c"
        badge_bg = "#fff3e0"
        icon = "🎉"
        badge_label = "Disease detected"

    st.markdown(f"""
        <div style="
            background-color: white;
            border-left: 6px solid {accent};
            border-radius: 12px;
            padding: 20px 24px;
            margin-bottom: 16px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        ">
            <span style="
                display: inline-block;
                background-color: {badge_bg};
                color: {accent};
                font-size: 12px;
                font-weight: 600;
                padding: 4px 12px;
                border-radius: 999px;
            ">{icon} {badge_label}</span>
            <h3 style="margin: 12px 0 0; font-size: 24px; color: #1a1a1a;">{presentable_name}</h3>
        </div>

        <div style="
            background-color: white;
            border-radius: 12px;
            padding: 20px 24px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        ">
            <p style="margin: 0; font-size: 13px; color: #6b7280;">Confidence</p>
            <p style="margin: 4px 0 12px; font-size: 34px; font-weight: 700; color: #1a1a1a;">{confidence_score:.1f}%</p>
            <div style="background-color: #e5e7eb; border-radius: 8px; height: 10px; overflow: hidden;">
                <div style="
                    width: {confidence_score}%;
                    height: 100%;
                    background: linear-gradient(90deg, {accent}, {accent}aa);
                    border-radius: 8px;
                "></div>
            </div>
        </div>

        <div style="
            background-color: white;
            border-radius: 12px;
            padding: 20px 24px;
            margin-top: 16px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        ">
            <p style="margin: 0 0 8px; font-size: 13px; color: #6b7280; font-weight: 600; text-transform: uppercase; letter-spacing: 0.03em;">
                Recommended care
            </p>
            <p style="margin: 0; font-size: 15px; line-height: 1.6; color: #374151;">
                {get_care_tip(presentable_name)}
            </p>
        </div>
    """, unsafe_allow_html=True)

show_emoji_burst("🎉")

# ---- Full breakdown ----
with st.expander("See full prediction breakdown"):
    top_n = min(5, len(raw_predictions))
    top_indices = np.argsort(raw_predictions)[::-1][:top_n]
    breakdown = {
        PLANT_CLASSES[i].replace("___", " - ").replace("_", " "): float(raw_predictions[i] * 100)
        for i in top_indices
    }
    st.bar_chart(breakdown)
