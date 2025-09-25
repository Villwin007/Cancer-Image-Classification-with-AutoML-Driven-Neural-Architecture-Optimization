# 🩺 Cancer Image Classification with AutoML-Driven Neural Architecture Optimization

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![React](https://img.shields.io/badge/React-18%2B-blue)
![Flask](https://img.shields.io/badge/Flask-2.3%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Accuracy](https://img.shields.io/badge/Accuracy-99.04%25-brightgreen)

A cutting-edge medical AI system that automatically designs and trains highly accurate convolutional neural networks for lung and colon cancer detection using Neural Architecture Search (NAS) and Explainable AI (XAI).

## 🌟 Key Features

### 🤖 AutoML & Neural Architecture Search
- **Automated Model Design**: Eliminates manual CNN architecture tuning
- **Neural Architecture Search**: Tests 15+ configurations to find optimal architecture
- **99.04% Validation Accuracy**: Significantly outperforms traditional manual designs
- **Reduced Human Bias**: Objective architecture selection process

### 🔍 Explainable AI (XAI)
- **Grad-CAM Heatmaps**: Visualize which image regions influence predictions
- **Textual Explanations**: Natural language reasoning for diagnoses
- **Confidence Analysis**: Detailed probability distributions
- **Key Factor Identification**: Highlight pathological features considered

### 💻 Full-Stack Web Application
- **Modern React.js Interface**: Beautiful, responsive design
- **Real-time Predictions**: Instant cancer detection from uploaded images
- **Educational Content**: Detailed project explanations and methodology
- **Professional Dashboard**: Tabbed interface for information and testing

## 📊 Performance Highlights

| Metric | NAS-Optimized Model | Traditional Manual CNN | Improvement |
|--------|---------------------|------------------------|-------------|
| **Validation Accuracy** | **99.04%** | 89.6% | **+9.44%** |
| **Validation Loss** | **0.056** | 0.24 | **-76.7%** |
| **Generalization Gap** | **Small** | Large | **Better Stability** |
| **Training Time** | 5 hours (NAS + Training) | 1 hour | **Worthwhile Investment** |

## 🏗️ Project Architecture

```
cancer-detection-system/
├── backend/                 # Flask API with PyTorch model
│   ├── app.py              # Main Flask application
│   ├── requirements.txt    # Python dependencies
│   └── model.pth          # Trained NAS-optimized model
├── frontend/               # React.js web interface
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── App.jsx        # Main application
│   │   └── main.jsx       # Entry point
│   └── package.json       # Node.js dependencies
└── datasets/              # LC25000 dataset (not included)
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** and **Node.js 16+**
- **PyTorch 2.0+** with CUDA support (recommended)
- **Modern web browser** with JavaScript support

### Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Villwin007/Cancer-Image-Classification-with-AutoML-Driven-Neural-Architecture-Optimization.git
   cd Cancer-Image-Classification-with-AutoML-Driven-Neural-Architecture-Optimization
   ```

2. **Backend Setup**
   ```bash
   cd backend
   pip install -r requirements.txt
   python app.py
   ```
   *Server starts at http://localhost:5000*

3. **Frontend Setup** (New Terminal)
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
   *Application starts at http://localhost:5173*

4. **Access the Application**
   - Open http://localhost:5173 in your browser
   - Navigate to "Model Testing" tab
   - Upload tissue images for instant analysis

## 🧠 Technical Implementation

### Neural Architecture Search (NAS)

**Search Space Configuration:**
```python
SEARCH_SPACE = {
    'num_conv_blocks': [3, 4, 5],
    'filters_multiplier': [1, 2, 4],
    'kernel_sizes_options': [[3,3,3], [3,5,3], [5,3,5], [3,3,5,3]],
    'use_batch_norm': [True, False],
    'conv_dropout': [0.1, 0.2, 0.3],
    'dense_units_options': [[512,256], [256,128], [512], [256]],
    'dense_dropout': [0.3, 0.4, 0.5],
    'learning_rate': [1e-3, 1e-4, 1e-5]
}
```

**Optimal Architecture Discovered:**
- **Convolutional Blocks**: 4 layers with 4× filter multiplier
- **Kernel Sizes**: [3, 3, 5, 3] for optimal feature extraction
- **Classifier**: [256, 128] dense units with 0.5 dropout
- **Optimizer**: Adam with 0.0001 learning rate

### Dataset Information

**LC25000 Dataset Specifications:**
- **Total Images**: 25,000 high-quality histopathology images
- **Classes**: 5 distinct tissue types
- **Image Size**: 768×768 pixels (resized to 128×128 for training)
- **Split**: 80% training, 20% validation

**Class Distribution:**
1. **Colon Adenocarcinoma** (Cancerous)
2. **Colon Benign** (Non-cancerous)
3. **Lung Adenocarcinoma** (Cancerous)
4. **Lung Benign** (Non-cancerous)
5. **Lung Squamous Cell Carcinoma** (Cancerous)

## 🎯 Model Performance

### Validation Results
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Colon Adenocarcinoma | 0.99 | 0.98 | 0.99 | 500 |
| Colon Benign | 0.98 | 0.99 | 0.99 | 500 |
| Lung Adenocarcinoma | 0.99 | 0.99 | 0.99 | 500 |
| Lung Benign | 0.99 | 0.98 | 0.99 | 500 |
| Lung Squamous Cell Carcinoma | 0.99 | 0.99 | 0.99 | 500 |
| **Accuracy** | | | **0.99** | 2500 |
| **Macro Avg** | 0.99 | 0.99 | 0.99 | 2500 |
| **Weighted Avg** | 0.99 | 0.99 | 0.99 | 2500 |

### Training Process
- **NAS Phase**: 15 configurations × 15 epochs each
- **Final Training**: 30 epochs with ReduceLROnPlateau
- **Data Augmentation**: Rotation, flipping, cropping, color jittering
- **Regularization**: Dropout, batch normalization, weight decay

## 🔧 API Endpoints

### Backend API (Flask)

| Endpoint | Method | Description | Parameters |
|----------|--------|-------------|------------|
| `/` | GET | API information | - |
| `/health` | GET | System health check | - |
| `/model-info` | GET | Model configuration details | - |
| `/predict` | POST | Basic image prediction | `image` (file) |
| `/predict-with-explanation` | POST | Prediction with XAI features | `image` (file) |

### Example API Usage

```python
import requests

# Basic prediction
response = requests.post('http://localhost:5000/predict', 
                       files={'image': open('tissue_image.jpg', 'rb')})

# Prediction with explanations
response = requests.post('http://localhost:5000/predict-with-explanation',
                       files={'image': open('tissue_image.jpg', 'rb')})

print(response.json())
```

## 🎨 Web Interface Features

### Project Information Tab
- **Hero Section**: Project overview and key statistics
- **Technical Explanation**: How NAS and AutoML work
- **Model Details**: Architecture specifications and training process
- **Performance Metrics**: Validation results and comparisons

### Model Testing Tab
- **Drag & Drop Interface**: Easy image upload
- **Real-time Analysis**: Instant predictions with confidence scores
- **Explainable AI**: Grad-CAM heatmaps and textual explanations
- **Results Visualization**: Probability distributions and key factors

## 📈 Comparative Analysis

### Advantages Over Traditional Approaches

| Aspect | NAS Approach | Manual Design |
|--------|--------------|---------------|
| **Architecture Discovery** | Automated, unbiased | Manual, experience-dependent |
| **Performance** | **99.04% accuracy** | 89.6% accuracy |
| **Development Time** | 5 hours (automated) | Weeks of trial-and-error |
| **Generalization** | Better, less overfitting | Prone to overfitting |
| **Reproducibility** | High (systematic) | Low (expert-dependent) |

## 🔬 Research Significance

### Key Contributions
1. **Automated Medical AI**: Demonstrated AutoML's superiority in medical imaging
2. **Explainable Diagnostics**: Bridging the gap between AI and clinical trust
3. **Resource Efficiency**: Optimal architecture discovery with minimal human intervention
4. **Clinical Relevance**: High accuracy suitable for辅助诊断 applications

### Potential Applications
- **Pathology Assistance**: Support for pathologists in cancer diagnosis
- **Medical Education**: Training tool for histopathology recognition
- **Research Platform**: Baseline for medical AI architecture research
- **Telemedicine**: Remote cancer screening capabilities

## 🛠️ Development Guide

### Adding New Features

1. **Extend NAS Search Space**
   ```python
   # Add new parameters to SEARCH_SPACE
   SEARCH_SPACE['new_parameter'] = [value1, value2, value3]
   ```

2. **Custom Model Architectures**
   ```python
   class CustomCancerModel(nn.Module):
       def __init__(self, config, num_classes):
           # Custom architecture implementation
           pass
   ```

3. **Additional Explanation Methods**
   ```python
   def add_shap_explanations(model, input_tensor):
       # Implement SHAP or LIME explanations
       pass
   ```

### Deployment Options

**Option A: Local Deployment**
```bash
# Backend with network access
python app.py --host=0.0.0.0 --port=5000

# Frontend production build
npm run build
```

**Option B: Cloud Deployment**
- **Frontend**: Vercel, Netlify, or GitHub Pages
- **Backend**: Heroku, Railway, or AWS EC2
- **Model Serving**: AWS SageMaker or Google AI Platform

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@software{cancer_detection_nas_2025,
  title = {Cancer Image Classification with AutoML-Driven Neural Architecture Optimization},
  author = {Dhanush Saravanan},
  year = {2025},
  url = {https://github.com/Villwin007/Cancer-Image-Classification-with-AutoML-Driven-Neural-Architecture-Optimization},
  note = {Advanced medical AI system with Neural Architecture Search and Explainable AI}
}
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Areas for Contribution
- Additional explanation methods (SHAP, LIME)
- Multi-modal data integration
- Real-time video analysis
- Mobile application development
- Additional cancer type support

## ⚠️ Important Disclaimers

### Medical Use
> **Important**: This tool is for research and educational purposes only. It should not be used for actual medical diagnosis. Always consult qualified healthcare professionals for medical advice and diagnosis.

### Dataset Licensing
- The LC25000 dataset used for training should be obtained with proper licensing
- Ensure compliance with data usage agreements for medical images

### Model Limitations
- Trained on specific histopathology images
- Performance may vary with different imaging techniques
- Requires validation on diverse patient populations

## 📞 Support & Contact

- **Developer**: Dhanush Saravanan
- **Email**: s.dhanush1106@gmail.com
- **Issues**: [GitHub Issues](https://github.com/Villwin007/Cancer-Image-Classification-with-AutoML-Driven-Neural-Architecture-Optimization/issues)
- **Documentation**: [Project Wiki](https://github.com/Villwin007/Cancer-Image-Classification-with-AutoML-Driven-Neural-Architecture-Optimization/wiki)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE] file for details.

## 🙏 Acknowledgments

- **LC25000 Dataset Providers**: For the comprehensive lung and colon cancer image collection
- **PyTorch & Flask Communities**: For excellent documentation and support
- **Medical AI Researchers**: For pioneering work in explainable medical AI
- **Open Source Contributors**: For the tools and libraries that made this project possible

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

*Advancing medical AI through automated architecture discovery and explainable diagnostics*

</div>
