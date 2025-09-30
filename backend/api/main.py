from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import sys
import os

# Import the simplified model
from model import ConvNet

app = FastAPI(title="Brain Tumor Classification API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define class labels
CLASS_LABELS = ["Glioma Tumor", "Meningioma Tumor", "No Tumor", "Pituitary Tumor"]

# Global model variable
model = None

def load_model():
    """Load the trained model"""
    global model
    try:
        model = ConvNet(num_classes=4)
        # Try different possible locations for the model file
        possible_paths = [
            "model_epoch_18.pth",  # Docker/production path
            "../model_epoch_18.pth",  # Development path
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model_epoch_18.pth")
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            raise FileNotFoundError("Model file 'model_epoch_18.pth' not found in any expected location")
            
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        print(f"Model loaded successfully from: {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

def preprocess_image(image_bytes):
    """Preprocess the uploaded image for model prediction"""
    try:
        # Open image from bytes
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Define preprocessing transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Apply transforms and add batch dimension
        input_tensor = transform(image).unsqueeze(0)
        return input_tensor
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Brain Tumor Classification API is running!"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
async def predict_tumor(file: UploadFile = File(...)):
    """Predict tumor type from uploaded MRI image"""
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Preprocess image
        input_tensor = preprocess_image(image_bytes)
        
        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = probabilities[predicted_class].item()
        
        # Prepare response
        prediction_result = {
            "predicted_class": CLASS_LABELS[predicted_class],
            "confidence": round(confidence * 100, 2),
            "all_probabilities": {
                CLASS_LABELS[i]: round(prob.item() * 100, 2) 
                for i, prob in enumerate(probabilities)
            }
        }
        
        return JSONResponse(content=prediction_result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)