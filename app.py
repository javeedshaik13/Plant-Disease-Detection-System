import streamlit as st
import torch
import torch.nn as nn
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

class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4)
        )
        self.res1 = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            )
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4)
        )
        self.res2 = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            )
        )
        self.classifier = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        res = self.res1[0](out)
        res = self.res1[1](res)
        out = out + res
        out = self.conv3(out)
        out = self.conv4(out)
        res = self.res2[0](out)
        res = self.res2[1](res)
        out = out + res
        out = self.classifier(out)
        return out

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

st.title("🌿 Plant Disease Detector")

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

model = ResNet9(3, len(class_names))
model.load_state_dict(torch.load("plant-disease-model.pth", map_location=torch.device('cpu')))
model.eval()


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
    img_array = np.transpose(img_array, (2, 0, 1))
    img_tensor = torch.tensor(img_array).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        pred_idx = torch.argmax(output, dim=1).item()
        confidence = torch.softmax(output, dim=1)[0, pred_idx].item()

    st.subheader(f"🧠 Prediction: {class_names[pred_idx]}")
    st.write(f"Confidence: {confidence * 100:.2f}%")
    if st.button(f"Remove Leaf {idx+1}", key=f"remove_{idx}"):
        remove_indices.append(idx)

if remove_indices:
    for idx in sorted(remove_indices, reverse=True):
        st.session_state.uploaded_files.pop(idx)
    st.rerun()
