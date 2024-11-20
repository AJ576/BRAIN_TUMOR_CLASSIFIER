# Brain Tumor Classification  

This is a Convolutional Neural Network model that predicts and classifies brain tumors. The model leverages advanced machine learning techniques, including a cyclic learning rate schedule, and applies image preprocessing to improve classification accuracy.  

---

## Problem Statement  

Brain tumors pose a significant health challenge, requiring accurate and efficient diagnostic tools. The goal of this project is to develop an automated classification system for brain tumors using MRI images, aiming to assist medical professionals in early detection and accurate diagnosis.  

---

## Key Results  

1. Achieved **95% accuracy** when excluding glioma predictions.  
2. Overall model accuracy: **75%**, with glioma being the most challenging tumor type to classify.  
3. Efficient deployment of the model through Streamlit, enabling real-time image classification.  

---

## Methodologies  

1. Trained a Convolutional Neural Network (CNN) with **cyclic learning rate scheduling** to improve convergence.  
2. Preprocessed MRI images using resizing and normalization techniques.  
3. Utilized **class-based accuracy evaluation** to identify model weaknesses, such as difficulties with glioma classification.  
4. Deployed the trained model using **Streamlit** for interactive image classification.  

---

## Data Sources  

- Dataset used: [Kaggle Brain MRI Images](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

---

## Technologies Used  

- **Python**  
  - NumPy, pandas  
  - PyTorch (for model training and evaluation)  
  - Matplotlib (for confusion matrix visualization)  
- **Streamlit** (for deployment as an interactive app)  
- **Google Colab** (for training and experimentation)  

---

## Authors  

This project was completed in collaboration with:  
- Aditya Jha
- Guillermo Marr

- [Aditya Jha](aditya.jha2020123@gmail.com)
- [Guillermo Marr](guillermomarr3@gmail.com)
