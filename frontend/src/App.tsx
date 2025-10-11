import React, { useState, useRef } from 'react';
import axios from 'axios';
import './App.css';

interface PredictionResult {
  predicted_class: string;
  confidence: number;
  all_probabilities: { [key: string]: number };
}

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      const reader = new FileReader();
      reader.onload = (e) => {
        setPreviewUrl(e.target?.result as string);
      };
      reader.readAsDataURL(file);
      setPrediction(null);
      setError(null);
    }
  };

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      const reader = new FileReader();
      reader.onload = (e) => {
        setPreviewUrl(e.target?.result as string);
      };
      reader.readAsDataURL(file);
      setPrediction(null);
      setError(null);
    }
  };

  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
  };

  const handlePredict = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      // Use environment variable if provided, otherwise fall back to Render URL
      const API_BASE = (process.env.REACT_APP_API_URL && process.env.REACT_APP_API_URL.trim() !== "")
        ? process.env.REACT_APP_API_URL
        : 'https://brain-tumor-classifier-qrb1.onrender.com';

      // Ensure no trailing slash and append /predict
      const apiUrl = `${API_BASE.replace(/\/$/, '')}/predict`;

      const response = await axios.post<PredictionResult>(
        apiUrl,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );
      setPrediction(response.data);
    } catch (err: any) {
      // More robust error message for network errors
      if (err.response?.data?.detail) setError(err.response.data.detail);
      else if (err.response) setError(`Server error: ${err.response.status} ${err.response.statusText}`);
      else setError(err.message || 'An error occurred during prediction');
    } finally {
      setLoading(false);
    }
  };

  const clearSelection = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setPrediction(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const getTumorTypeColor = (tumorType: string) => {
    switch (tumorType.toLowerCase()) {
      case 'no tumor':
        return '#10b981'; // green
      case 'glioma tumor':
        return '#ef4444'; // red
      case 'meningioma tumor':
        return '#f59e0b'; // amber
      case 'pituitary tumor':
        return '#8b5cf6'; // violet
      default:
        return '#6b7280'; // gray
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🧠 Brain Tumor Classifier</h1>
        <p>Upload an MRI image to detect and classify brain tumors</p>
      </header>

      <main className="App-main">
        <div className="upload-section">
          <div
            className={`drop-zone ${selectedFile ? 'has-file' : ''}`}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onClick={() => fileInputRef.current?.click()}
          >
            {previewUrl ? (
              <div className="preview-container">
                <img src={previewUrl} alt="Preview" className="preview-image" />
                <button className="clear-btn" onClick={clearSelection}>
                  ✕
                </button>
              </div>
            ) : (
              <div className="drop-content">
                <div className="drop-icon">📁</div>
                <p>Drag and drop an MRI image here</p>
                <p className="drop-subtext">or click to browse</p>
                <p className="file-types">Supports: JPG, JPEG, PNG</p>
              </div>
            )}
          </div>
          
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileSelect}
            style={{ display: 'none' }}
          />

          {selectedFile && (
            <button
              className="predict-btn"
              onClick={handlePredict}
              disabled={loading}
            >
              {loading ? (
                <>
                  <div className="spinner"></div>
                  Analyzing...
                </>
              ) : (
                'Predict Tumor Type'
              )}
            </button>
          )}
        </div>

        {error && (
          <div className="error-message">
            <span className="error-icon">⚠️</span>
            {error}
          </div>
        )}

        {prediction && (
          <div className="results-section">
            <h2>Prediction Results</h2>
            <div className="prediction-card">
              <div 
                className="main-prediction"
                style={{ borderLeftColor: getTumorTypeColor(prediction.predicted_class) }}
              >
                <h3>{prediction.predicted_class}</h3>
                <div className="confidence">
                  Confidence: {prediction.confidence}%
                </div>
              </div>

              <div className="all-probabilities">
                <h4>All Probabilities:</h4>
                {Object.entries(prediction.all_probabilities).map(([type, prob]) => (
                  <div key={type} className="probability-item">
                    <span className="probability-label">{type}:</span>
                    <div className="probability-bar">
                      <div
                        className="probability-fill"
                        style={{
                          width: `${prob}%`,
                          backgroundColor: getTumorTypeColor(type)
                        }}
                      ></div>
                      <span className="probability-value">{prob}%</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </main>

      <footer className="App-footer">
        <p>Built with React and FastAPI • Advanced AI Medical Diagnostics</p>
      </footer>
    </div>
  );
}

export default App;
