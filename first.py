import streamlit as st
import tensorflow as tf 
import numpy as np
import cv2
import os
from PIL import Image

# Load your trained model
def model_predict(image_path):
    model = tf.keras.models.load_model(r"CNN_basedplantdisease_model.keras")
    img = cv2.imread(image_path)
    if img is None:
        return "Error loading image"
    H, W, C = 224, 224, 3
    img = cv2.resize(img, (H, W))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img)
    img = img.astype("float32")
    img = img / 255.0
    img = img.reshape(1, H, W, C)
    prediction = np.argmax(model.predict(img), axis=-1)[0]
    return prediction

# Add a simple validation function
def is_plant_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return False
    green_pixels = np.sum((img[:, :, 1] > img[:, :, 0]) & (img[:, :, 1] > img[:, :, 2]))
    total_pixels = img.shape[0] * img.shape[1]
    green_ratio = green_pixels / total_pixels
    return green_ratio > 0.1  # Adjust threshold as needed

# Sidebar
st.sidebar.title("Plant Disease Detection System for Sustainable Agriculture")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

# Display an image
img = Image.open(r"C:\Users\kumar\OneDrive\Desktop\python\OIP.jpg")
st.image(img)

# Main Page
if app_mode == "HOME":
    st.markdown("<h1 style='text-align: center;'>Plant Disease Detection System for Sustainable Agriculture</h1>", unsafe_allow_html=True)

# Prediction Page
elif app_mode == "DISEASE RECOGNITION":
    st.header("Plant Disease Detection System for Sustainable Agriculture")

    test_image = st.file_uploader("Choose an Image:")
    if test_image is not None:
        save_path = os.path.join(os.getcwd(), test_image.name)
        with open(save_path, "wb") as f:
            f.write(test_image.getbuffer())

        if st.button("Show Image"):
            st.image(test_image, use_column_width=True)

        if st.button("Predict"):
            st.snow()
            st.write("Our prediction:")
            
            # Validate if the image is of a plant
            if is_plant_image(save_path):
                result_index = model_predict(save_path)
                class_name = {
                    0: 'Apple - Apple scab',
                    1: 'Apple - Black rot',
                    2: 'Apple - Cedar apple rust',
                    3: 'Apple - Healthy',
                    4: 'Blueberry - Healthy',
                    5: 'Cherry (including sour) - Powdery mildew',
                    6: 'Cherry (including sour) - Healthy',
                    7: 'Corn (maize) - Cercospora leaf spot, Gray leaf spot',
                    8: 'Corn (maize) - Common rust',
                    9: 'Corn (maize) - Northern Leaf Blight',
                    10: 'Corn (maize) - Healthy',
                    11: 'Grape - Black rot',
                    12: 'Grape - Esca (Black Measles)',
                    13: 'Grape - Leaf blight (Isariopsis Leaf Spot)',
                    14: 'Grape - Healthy',
                    15: 'Orange - Haunglongbing (Citrus greening)',
                    16: 'Peach - Bacterial spot',
                    17: 'Peach - Healthy',
                    18: 'Pepper, bell - Bacterial spot',
                    19: 'Pepper, bell - Healthy',
                    20: 'Potato - Early blight',
                    21: 'Potato - Late blight',
                    22: 'Potato - Healthy',
                    23: 'Raspberry - Healthy',
                    24: 'Soybean - Healthy',
                    25: 'Squash - Powdery mildew',
                    26: 'Strawberry - Leaf scorch',
                    27: 'Strawberry - Healthy',
                    28: 'Tomato - Bacterial spot',
                    29: 'Tomato - Early blight',
                    30: 'Tomato - Late blight',
                    31: 'Tomato - Leaf Mold',
                    32: 'Tomato - Septoria leaf spot',
                    33: 'Tomato - Spider mites, Two-spotted spider mite',
                    34: 'Tomato - Target Spot',
                    35: 'Tomato - Tomato Yellow Leaf Curl Virus',
                    36: 'Tomato - Tomato mosaic virus',
                    37: 'Tomato - Healthy'
                }
                st.success(f"Model is predicting it's a {class_name.get(result_index, 'Unknown')}")
            else:
                st.error("The uploaded image does not appear to be of a plant.")
