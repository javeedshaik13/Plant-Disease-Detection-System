import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

background_image_url = "https://w0.peakpx.com/wallpaper/62/257/HD-wallpaper-farm-house-grass-sunset-farm-skies-tree-nature-fields-sunrise-road.jpg"
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{background_image_url}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

class_names = [
    'Potato___healthy',
    'Potato___Late_blight',
    'Potato___Early_blight'
]

st.title("🌿 Plant Disease Detector")

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("potatoes.h5")

model = load_model()

show_camera = st.button("Use Camera")
if show_camera:
    camera_img = st.camera_input("Take a photo of a leaf")
    if camera_img:
        st.session_state.uploaded_files.append(camera_img)


uploaded_files = st.file_uploader(
    "Or upload leaf images", type=["jpg", "png"], accept_multiple_files=True
)
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
    img_tensor = np.expand_dims(img_array, axis=0)  # Shape: (1, 256, 256, 3)

    predictions = model.predict(img_tensor)
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
