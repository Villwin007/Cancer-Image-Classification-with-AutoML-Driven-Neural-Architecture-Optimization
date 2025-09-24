import { useState } from 'react';
import axios from 'axios';
import Results from '../Results/Results';
import './UploadInterface.css';

const UploadInterface = () => {
  const [uploadedImage, setUploadedImage] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [dragOver, setDragOver] = useState(false);

  const API_BASE_URL = 'http://localhost:5000';

  const handleImageUpload = (file) => {
    setUploadedImage(file);
    setResults(null);
    setError(null);
  };

  const handleAnalyze = async () => {
    if (!uploadedImage) return;
    
    setIsLoading(true);
    setError(null);
    
    try {
      const formData = new FormData();
      formData.append('image', uploadedImage);
      
      // Use the explanation endpoint
      const response = await axios.post(`${API_BASE_URL}/predict-with-explanation`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 45000, // Increase timeout for explanation generation
      });
      
      if (response.data.success) {
        setResults(response.data);
      } else {
        setError(response.data.error || 'Prediction failed');
      }
    } catch (err) {
      console.error('API Error:', err);
      if (err.code === 'NETWORK_ERROR' || err.message === 'Network Error') {
        setError('Cannot connect to the server. Please make sure the backend is running on port 5000.');
      } else if (err.response?.data?.error) {
        setError(err.response.data.error);
      } else {
        setError('An error occurred while processing the image. Please try again.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleImageUpload(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleImageUpload(e.target.files[0]);
    }
  };

  const handleNewAnalysis = () => {
    setUploadedImage(null);
    setResults(null);
    setError(null);
  };

  return (
    <section className="upload-interface">
      <div className="container">
        {!results ? (
          <>
            <div className="upload-container">
              <div 
                className={`upload-area ${dragOver ? 'drag-over' : ''} ${uploadedImage ? 'has-image' : ''}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
              >
                {uploadedImage ? (
                  <div className="image-preview">
                    <img src={URL.createObjectURL(uploadedImage)} alt="Upload preview" />
                    <p>{uploadedImage.name}</p>
                  </div>
                ) : (
                  <>
                    <div className="upload-icon">
                      <svg width="48" height="48" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M14 2H6C4.9 2 4.01 2.9 4.01 4L4 20C4 21.1 4.89 22 5.99 22H18C19.1 22 20 21.1 20 20V8L14 2Z" fill="#3498db"/>
                        <path d="M16 13H13V16H11V13H8V11H11V8H13V11H16V13Z" fill="white"/>
                      </svg>
                    </div>
                    <p>Drag & drop your image here or click to browse</p>
                    <p className="upload-hint">Supported formats: JPG, PNG, JPEG, BMP, TIFF</p>
                  </>
                )}
                <input 
                  type="file" 
                  accept="image/*" 
                  onChange={handleFileSelect}
                  className="file-input"
                />
              </div>

              {error && (
                <div className="error-message">
                  <span>⚠️ {error}</span>
                </div>
              )}

              <div className="upload-actions">
                <button 
                  onClick={handleAnalyze} 
                  disabled={!uploadedImage || isLoading}
                  className="analyze-btn"
                >
                  {isLoading ? (
                    <>
                      <div className="button-spinner"></div>
                      Analyzing with Explainable AI...
                    </>
                  ) : (
                    'Analyze Image with Explainable AI'
                  )}
                </button>
              </div>
            </div>

            <div className="upload-info">
              <h3>What to Upload</h3>
              <p>
                For best results, upload microscopic images of lung or colon tissue. 
                Our NAS-optimized model achieves 99.04% validation accuracy and can detect:
              </p>
              <ul>
                <li><strong>Colon Benign (colon_n):</strong> Non-cancerous colon tissue</li>
                <li><strong>Colon Adenocarcinoma (colon_aca):</strong> A type of colon cancer</li>
                <li><strong>Lung Benign (lung_n):</strong> Non-cancerous lung tissue</li>
                <li><strong>Lung Adenocarcinoma (lung_aca):</strong> A type of lung cancer</li>
                <li><strong>Lung Squamous Cell Carcinoma (lung_scc):</strong> Another type of lung cancer</li>
              </ul>
              
              <div className="ai-explanation-info">
                <h4>Explainable AI Features</h4>
                <div className="explanation-features">
                  <div className="feature">
                    <span className="feature-icon">🔍</span>
                    <div>
                      <strong>Visual Heatmaps</strong>
                      <p>See which areas of the image influenced the diagnosis</p>
                    </div>
                  </div>
                  <div className="feature">
                    <span className="feature-icon">📝</span>
                    <div>
                      <strong>Textual Explanations</strong>
                      <p>Understand the AI's reasoning in natural language</p>
                    </div>
                  </div>
                  <div className="feature">
                    <span className="feature-icon">🎯</span>
                    <div>
                      <strong>Key Factors</strong>
                      <p>Learn what pathological features were considered</p>
                    </div>
                  </div>
                </div>
              </div>
              
              <p className="disclaimer">
                <strong>Disclaimer:</strong> This tool is for research purposes only and should not be used 
                for actual medical diagnosis. Always consult a healthcare professional for medical advice.
              </p>
            </div>
          </>
        ) : (
          <Results results={results} isLoading={isLoading} onNewAnalysis={handleNewAnalysis} />
        )}
      </div>
    </section>
  );
};

export default UploadInterface;