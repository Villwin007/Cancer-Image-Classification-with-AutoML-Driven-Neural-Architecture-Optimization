# 🎯 Cancer Image Classification with AutoML-Driven Neural Architecture Optimization

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![React](https://img.shields.io/badge/React-18%2B-blue)
![Flask](https://img.shields.io/badge/Flask-2.3%2B-green)
![Accuracy](https://img.shields.io/badge/Accuracy-99.04%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

A cutting-edge medical AI system that automatically designs and trains highly accurate convolutional neural networks for lung and colon cancer detection using Neural Architecture Search (NAS) and Explainable AI.

## 📖 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Results](#-results)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Usage](#-usage)
- [Model Details](#-model-details)
- [API Documentation](#-api-documentation)
- [Contributing](#-contributing)
- [License](#-license)

## 🌟 Overview

This project demonstrates a modern approach to medical AI by leveraging **Automated Machine Learning (AutoML)** and **Neural Architecture Search (NAS)** to automatically design optimal neural network architectures for cancer image classification. The system achieves **99.04% validation accuracy** on the LC25000 dataset, significantly outperforming traditional manually-designed models.

### 🎯 Key Achievements
- **99.04% validation accuracy** - State-of-the-art performance
- **Automated architecture design** - No manual neural network engineering required
- **Explainable AI** - Grad-CAM visualizations and textual explanations
- **Production-ready web interface** - Professional React.js frontend with Flask backend

## ✨ Key Features

### 🔬 Advanced AI Capabilities
- **Neural Architecture Search (NAS)**: Automatically explores thousands of possible CNN architectures
- **Explainable AI (XAI)**: Grad-CAM heatmaps showing decision regions
- **Real-time Predictions**: Fast inference with professional web interface
- **Multi-class Classification**: Detects 5 types of lung and colon tissues

### 💻 Technical Excellence
- **AutoML Pipeline**: End-to-end automated model design and training
- **Hyperparameter Optimization**: Intelligent search space exploration
- **Data Augmentation**: Comprehensive image preprocessing pipeline
- **Model Interpretability**: Detailed explanations for medical professionals

## 🏗️ Architecture

### System Overview
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   React.js      │    │   Flask API      │    │   PyTorch       │
│   Frontend      │◄──►│   Backend        │◄──►│   NAS Model     │
│                 │    │                  │    │                 │
│ - Model Testing │    │ - Image Preproc  │    │ - 99.04% Acc    │
│ - Explanations  │    │ - Predictions    │    │ - Grad-CAM      │
│ - Results Viz   │    │ - Explanations   │    │ - AutoML        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Neural Architecture Search Pipeline
1. **Search Space Definition**: 15+ architectural parameters
2. **Configuration Sampling**: Intelligent random sampling
3. **Rapid Evaluation**: Fast training with early stopping
4. **Best Model Selection**: Performance-based architecture selection
5. **Final Training**: Full training of optimal architecture

## 📊 Results

### Performance Metrics
| Metric | NAS-Optimized Model | Traditional CNN | Improvement |
|--------|---------------------|-----------------|-------------|
| **Validation Accuracy** | **99.04%** | 89.6% | **+9.44%** |
| **Validation Loss** | **0.015** | 0.24 | **+93.75%** |
| **Training Time** | 5 hours (NAS + Training) | 1 hour | +4 hours |
| **Generalization** | **Excellent** | Good | **Better** |

### Class Detection Performance
The model accurately detects five tissue types:
- **Lung Benign** (Non-cancerous lung tissue)
- **Lung Adenocarcinoma** (Lung cancer type)
- **Lung Squamous Cell Carcinoma** (Lung cancer type) 
- **Colon Adenocarcinoma** (Colon cancer type)
- **Colon Benign** (Non-cancerous colon tissue)

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Installation

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

3. **Frontend Setup** (New Terminal)
```bash
cd lung-cancer-detection-web
npm install
npm run dev
```

4. **Access the Application**
- Frontend: http://localhost:5173
- Backend API: http://localhost:5000

### Quick Demo
1. Open http://localhost:5173 in your browser
2. Navigate to "Model Testing" tab
3. Upload a lung/colon tissue image
4. View AI predictions with explanations

## 📁 Project Structure

```
Cancer-Image-Classification/
├── Model_Training
├── backend/
│   ├── app.py                 # Flask API with Grad-CAM explanations
│   ├── requirements.txt       # Python dependencies
│   └── lung_colon_cancer_nas_model_*.pth  # Trained NAS model
│   ├── src/
│   │   ├── components/        # React components
│   │   │   ├── Header/        # Application header
│   │   │   ├── Tabs/          # Tabbed interface
│   │   │   ├── InfoTab/       # Project information
│   │   │   ├── TestTab/       # Model testing interface
│   │   │   ├── UploadInterface/ # Image upload component
│   │   │   ├── Results/       # Prediction results
│   │   │   ├── ExplanationView/ # AI explanations
│   │   │   └── Footer/        # Project information
│   │   ├── App.jsx            # Main application component
│   │   └── main.jsx           # Application entry point
│   ├── package.json           # Node.js dependencies
│   └── vite.config.js         # Vite configuration
└── README.md                  # This file
```

## 💻 Usage

### Model Training
```python
# Run Neural Architecture Search
python training_code.py

# This will:
# 1. Perform NAS with 15 configurations
# 2. Select best architecture (99.04% accuracy)
# 3. Train final model
# 4. Generate performance plots and reports
```

### Web Interface Usage
1. **Project Information Tab**: Learn about NAS and model architecture
2. **Model Testing Tab**: Upload images for AI analysis
3. **Results View**: See predictions with confidence scores
4. **Explanation Panel**: Understand AI reasoning with heatmaps

### API Usage
```bash
# Health check
curl http://localhost:5000/health

# Model information
curl http://localhost:5000/model-info

# Prediction with explanation
curl -X POST -F "image=@path_to_image.jpg" http://localhost:5000/predict-with-explanation
```

## 🔧 Model Details

### Neural Architecture Search Configuration
```python
SEARCH_SPACE = {
    'num_conv_blocks': [3, 4, 5],
    'filters_multiplier': [1, 2, 4],
    'kernel_sizes_options': [[3,3,3], [3,5,3], [5,3,5], [3,3,5,3], [3,5,5,3]],
    'use_batch_norm': [True, False],
    'conv_dropout': [0.1, 0.2, 0.3],
    'dense_units_options': [[512,256], [256,128], [512], [256]],
    'dense_dropout': [0.3, 0.4, 0.5],
    'learning_rate': [1e-3, 1e-4, 1e-5]
}
```

### Best Model Architecture
- **Convolutional Blocks**: 4 layers with 4× filter multiplier
- **Kernel Sizes**: [3, 3, 5, 3] optimized pattern
- **Dense Layers**: [256, 128] units with 0.5 dropout
- **Learning Rate**: 0.0001 with ReduceLROnPlateau scheduler
- **Batch Size**: 32 with extensive data augmentation

## 🌐 API Documentation

### Endpoints

#### GET `/health`
Check API status and model loading status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "mode": "real",
  "explainable_ai": true
}
```

#### GET `/model-info`
Get detailed model information and configuration.

**Response:**
```json
{
  "model_name": "NAS-Optimized Cancer Detection Model",
  "validation_accuracy": "99.04%",
  "model_loaded": true,
  "classes": ["Colon Adenocarcinoma", "Colon Benign", ...],
  "features": ["Grad-CAM explanations", "Textual reasoning"]
}
```

#### POST `/predict-with-explanation`
Analyze image with AI explanations.

**Request:**
```bash
curl -X POST -F "image=@tissue_image.jpg" http://localhost:5000/predict-with-explanation
```

**Response:**
```json
{
  "success": true,
  "predictions": [...],
  "confidence": 0.9904,
  "diagnosis": "Lung Adenocarcinoma",
  "explanation": {
    "textual": "The model detected irregular glandular patterns...",
    "heatmap_image": "base64_encoded_image",
    "key_factors": ["Glandular architecture", "Nuclear features"]
  }
}
```

## 🤝 Contributing

We welcome contributions to enhance this medical AI system! Here's how you can help:

### Areas for Improvement
- **Model Performance**: Experiment with different NAS algorithms
- **Explainability**: Enhance Grad-CAM and add LIME/SHAP explanations
- **Frontend Features**: Add prediction history and comparison tools
- **Deployment**: Docker containers and cloud deployment scripts

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Reporting Issues
Please use the [GitHub issue tracker](https://github.com/Villwin007/Cancer-Image-Classification-with-AutoML-Driven-Neural-Architecture-Optimization/issues) for:
- Bug reports
- Feature requests
- Documentation improvements
- Security vulnerabilities

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Citation
If you use this work in your research, please cite:
```bibtex
@software{CancerClassificationNAS2025,
  author = {Dhanush Saravanan},
  title = {Cancer Image Classification with AutoML-Driven Neural Architecture Optimization},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Villwin007/Cancer-Image-Classification-with-AutoML-Driven-Neural-Architecture-Optimization}}
}
```

## ⚠️ Important Disclaimer

**Medical Disclaimer**: This tool is for research and educational purposes only. It should not be used for actual medical diagnosis, treatment, or clinical decision-making. Always consult qualified healthcare professionals for medical advice.

**Research Use**: The model achieves 99.04% accuracy on the LC25000 dataset but should be clinically validated before any medical application.

## 📞 Support

For questions, issues, or collaborations:
- **GitHub Issues**: [Create an issue](https://github.com/Villwin007/Cancer-Image-Classification-with-AutoML-Driven-Neural-Architecture-Optimization/issues)
- **Email**: s.dhanush1106@gmail.com
- **Documentation**: See code comments and this README

## 🎓 Academic Impact

This project demonstrates:
- **AutoML superiority** over manual architecture design in medical imaging
- **Explainable AI** importance in healthcare applications
- **Neural Architecture Search** practical implementation
- **End-to-end ML system** design from research to deployment

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

*Advancing medical AI through automated machine learning and explainable artificial intelligence.*

</div>
