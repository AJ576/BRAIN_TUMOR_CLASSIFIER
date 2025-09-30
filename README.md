# Brain Tumor Classifier

A full-stack application for classifying brain tumors from MRI images using deep learning.

## Project Structure

```
BRAIN_TUMOR_CLASSIFIER/
├── backend/                 # Python FastAPI backend
│   ├── api/
│   │   └── main.py         # FastAPI application
│   ├── model_epoch_18.pth  # Trained PyTorch model
│   ├── the_best_cyclic_algorithm.py  # Model architecture
│   ├── requirements.txt    # Python dependencies
│   └── MRI/                # Training and testing data
└── frontend/               # React frontend
    ├── src/
    ├── public/
    └── package.json
```

## Features

- **Deep Learning Model**: Custom CNN architecture trained on brain MRI images
- **4 Classification Types**:
  - Glioma Tumor
  - Meningioma Tumor
  - No Tumor
  - Pituitary Tumor
- **Modern UI**: React-based frontend with professional design
- **REST API**: FastAPI backend with proper error handling
- **Real-time Predictions**: Upload image and get instant classification results

## Setup Instructions

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start the FastAPI server:
```bash
cd api
python main.py
```

The backend will be available at `http://localhost:8000`

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the React development server:
```bash
npm start
```

The frontend will be available at `http://localhost:3000`

## API Endpoints

- `GET /` - Health check
- `GET /health` - Detailed health status
- `POST /predict` - Upload image and get tumor classification

## Model Performance

The model was trained using a cyclic learning rate schedule and achieves high accuracy on the test dataset. The model uses ELU activation functions and includes dropout for regularization.

## Technologies Used

### Backend
- FastAPI
- PyTorch
- Pillow (PIL)
- Uvicorn

### Frontend
- React
- TypeScript
- CSS3
- Axios

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License.