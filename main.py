import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

background_image_url = "https://w0.peakpx.com/wallpaper/62/257/HD-wallpaper-farm-house-grass-sunset-farm-skies-tree-nature-fields-sunrise-road.jpg"
st.markdown(
    f"""
    <style>
    .stApp {{
        background: url("{background_image_url}") no-repeat center center fixed;
        background-size: cover;
    }}
    .stApp > div:first-child {{
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 12px;
        padding: 2rem;
        margin: 2rem;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

class_names = [
    'Tomato___Late_blight', 'Tomato___healthy', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)',
    'Soybean___healthy', 'Squash___Powdery_mildew', 'Potato___healthy', 'Corn_(maize)___Northern_Leaf_Blight',
    'Tomato___Early_blight', 'Tomato___Septoria_leaf_spot', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Strawberry___Leaf_scorch', 'Peach___healthy', 'Apple___Apple_scab', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Bacterial_spot', 'Apple___Black_rot', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Peach___Bacterial_spot', 'Apple___Cedar_apple_rust', 'Tomato___Target_Spot', 'Pepper,_bell___healthy',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Potato___Late_blight', 'Tomato___Tomato_mosaic_virus',
    'Strawberry___healthy', 'Apple___healthy', 'Grape___Black_rot', 'Potato___Early_blight',
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Common_rust_', 'Grape___Esca_(Black_Measles)',
    'Raspberry___healthy', 'Tomato___Leaf_Mold', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Pepper,_bell___Bacterial_spot', 'Corn_(maize)___healthy'
]
st.title("🌿 Plant Disease Detector (TensorFlow Model)")

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model_v1.h5")

model = load_model()

show_camera = st.button("Use Camera")
if show_camera:
    camera_img = st.camera_input("Take a photo of a leaf")
    if camera_img:
        st.session_state.uploaded_files.append(camera_img)

uploaded_files = st.file_uploader("Or upload leaf images", type=["jpg", "png"], accept_multiple_files=True)
if uploaded_files:
    for file in uploaded_files:
        if file not in st.session_state.uploaded_files:
            st.session_state.uploaded_files.append(file)

remove_indices = []
for idx, img in enumerate(st.session_state.uploaded_files):
    st.write(f"Leaf {idx+1}:")
    image = Image.open(img).convert("RGB").resize((256, 256))
    st.image(image)
    
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 256, 256, 3)

    predictions = model.predict(img_array)
    pred_idx = np.argmax(predictions[0])
    confidence = predictions[0][pred_idx]

    st.subheader(f"🧠 Prediction: {class_names[pred_idx]}")
    st.write(f"Confidence: {confidence * 100:.2f}%")
    
    if st.button(f"Remove Leaf {idx+1}", key=f"remove_{idx}"):
        remove_indices.append(idx)

if remove_indices:
    for idx in sorted(remove_indices, reverse=True):
        st.session_state.uploaded_files.pop(idx)
    st.rerun()
