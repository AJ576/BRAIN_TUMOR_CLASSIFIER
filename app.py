import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Define class labels
class_labels = ["Glioma Tumor", "Meningioma Tumor", "No Tumor", "Pituitary Tumor"]

# Load the model (update the path to your model file)
@st.cache_resource
def load_model():
    # Import the model architecture from your uploaded file
    from the_best_cyclic_algorithm import ConvNet  # Adjust the import as needed
    model = ConvNet(num_classes=4)  # Adjust num_classes to match your model
    model.load_state_dict(torch.load("model_epoch_18.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Define preprocessing transforms
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Adjust size as per your model
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Streamlit UI
st.title("Brain MRI Tumor Classification")
st.write("Upload an MRI image to predict the tumor type.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded MRI Image',  use_container_width=True)

    # Preprocess and predict
    with st.spinner('Analyzing...'):
        input_tensor = preprocess_image(image)
        output = model(input_tensor)
        _, predicted_class = torch.max(output, 1)
        prediction = class_labels[predicted_class.item()]

    # Display result
    st.success(f"The model predicts: **{prediction}**")
