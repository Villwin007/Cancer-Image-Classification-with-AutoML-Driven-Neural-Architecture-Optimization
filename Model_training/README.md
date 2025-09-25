# 🏥Cancer Image Classification with AutoML Driven Neural Architecture Optimization (NAS)

## 📖 Introduction

This repository presents an advanced deep learning solution for automated classification of lung and colon cancer histopathology images using Neural Architecture Search (NAS). The system achieves **99.72% validation accuracy** in distinguishing between five different cancer types and benign tissues through an automated architecture discovery process.

The project demonstrates how modern neural architecture search techniques can outperform manually designed architectures in medical image analysis, providing a robust framework for cancer diagnosis assistance.

## 🎯 Problem Statement

Cancer diagnosis through histopathology image analysis is a complex, time-consuming process that requires expert pathologists. The challenges include:

- **High diagnostic variability** between pathologists
- **Time-intensive manual analysis** of tissue samples
- **Limited access** to specialized medical expertise in remote areas
- **Increasing workload** on healthcare systems

This project addresses these challenges by developing an AI system that can:
- Automatically classify lung and colon cancer subtypes from histopathology images
- Provide rapid, consistent second opinions to pathologists
- Reduce diagnostic time from hours to seconds
- Maintain exceptional accuracy comparable to human experts

## 📊 Dataset Overview

The model classifies images into **5 distinct categories**:

| Class | Type | Description |
|-------|------|-------------|
| **Colon Adenocarcinoma** | Cancerous | Malignant tumor in glandular tissues |
| **Colon Benign** | Non-cancerous | Healthy colon tissue |
| **Lung Adenocarcinoma** | Cancerous | Most common type of lung cancer |
| **Lung Benign** | Non-cancerous | Healthy lung tissue |
| **Lung Squamous Cell Carcinoma** | Cancerous | Second most common lung cancer |

**Dataset Statistics:**
- **Total Images**: 25,000 histopathology images
- **Image Size**: 768×768 pixels (original), resized to 128×128 for training
- **Classes**: 5 balanced categories
- **Train/Validation Split**: 80%/20%

**Dataset Download:** [LC25000 DATASET](https://academictorrents.com/details/7a638ed187a6180fd6e464b3666a6ea0499af4af)

## 🏗️ Architecture & Methodology

### Neural Architecture Search (NAS) Approach

The system employs a sophisticated NAS strategy to automatically discover optimal neural network architectures:

#### 🔍 Search Space Design
```python
SEARCH_SPACE = {
    'num_conv_blocks': [3, 4, 5],           # Network depth
    'filters_multiplier': [1, 2, 4],        # Network width
    'kernel_sizes_options': [[3,3,3], [3,5,3], [5,3,5], ...],  # Receptive fields
    'use_batch_norm': [True, False],        # Normalization strategy
    'conv_dropout': [0.1, 0.2, 0.3],       # Convolutional regularization
    'dense_units_options': [[512,256], [256,128], [512], [256]], # Classifier design
    'dense_dropout': [0.3, 0.4, 0.5],      # Fully-connected regularization
    'learning_rate': [1e-3, 1e-4, 1e-5]    # Optimization parameters
}
```

#### ⚡ Search Algorithm
- **Random Search with Hyperband Pruning**: Tests 15 configurations efficiently
- **Multi-stage Training**: Short epochs for exploration → Full epochs for refinement
- **Performance-based Selection**: Chooses architectures with best validation accuracy

### 🏆 Best Discovered Architecture

The NAS discovered an optimal configuration achieving **99.72% accuracy**:

```json
{
  "num_conv_blocks": 4,
  "filters_multiplier": 4,
  "kernel_sizes": [3, 3, 5, 3],
  "use_batch_norm": false,
  "conv_dropout": 0.1,
  "dense_units": [256, 128],
  "dense_dropout": 0.5,
  "learning_rate": 0.0001
}
```

**Key Architectural Insights:**
- ✅ **4 convolutional blocks** with **4× filter multiplier** for optimal capacity
- ✅ **Mixed kernel sizes** [3,3,5,3] for diverse feature extraction
- ✅ **No batch normalization** - simpler architecture performed better
- ✅ **High dropout (0.5)** in classifier layers for robust regularization

## 🚀 Performance Results

### 📈 Training Performance
| Metric | Value | Description |
|--------|-------|-------------|
| **Final Validation Accuracy** | 99.72% | Near-perfect classification |
| **Training Accuracy** | 99.98% | Excellent learning capability |
| **Overfitting Gap** | 0.26% | Minimal, indicating great generalization |
| **Training Time** | 2.4 hours | Efficient convergence |
| **Total NAS Process** | 15.4 hours | Comprehensive architecture search |

### 🎯 Classification Performance
```
                              precision    recall  f1-score   support

        Colon_Adenocarcinoma       1.00      1.00      1.00      1025
         Colon_Benign_Tissue       1.00      1.00      1.00      1007
        Lung_Adenocarcinoma       0.99      0.99      0.99      1009
          Lung-Benign_Tissue       1.00      1.00      1.00       980
Lung_Squamous_Cell_Carcinoma       0.99      0.99      0.99       979

                    accuracy                           1.00      5000
                   macro avg       1.00      1.00      1.00      5000
                weighted avg       1.00      1.00      1.00      5000
```

### ⏱️ Computational Efficiency
| Phase | Time | Configurations Tested |
|-------|------|---------------------|
| **NAS Search** | 13.0 hours | 15 architectures |
| **Final Training** | 2.4 hours | 1 best architecture |
| **Average Epoch** | 4.8 minutes | 30 epochs total |

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- NVIDIA GPU with CUDA support (recommended)
- 8GB+ RAM
- 10GB+ free disk space

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/Villwin007/Cancer-Image-Classification-with-AutoML-Driven-Neural-Architecture-Optimization.git
```

2. **Create and activate virtual environment**
```bash
python -m venv cancer_env
source cancer_env/bin/activate  # Linux/Mac
cancer_env\Scripts\activate     # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Required Dependencies
```txt
torch==2.5.1+cu121
torchvision==0.20.1+cu121
scikit-learn==1.5.0
seaborn==0.13.2
matplotlib==3.8.0
numpy==1.26.0
tqdm==4.66.0
Pillow==10.2.0
```

## 💻 Usage Instructions

### 1. Data Preparation
Organize your dataset in the following structure:
```
Lung_and_Colon_Cancer/
├── Colon_Adenocarcinoma/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Colon_Benign_Tissue/
├── Lung_Adenocarcinoma/
├── Lung_Benign_Tissue/
└── Lung_Squamous_Cell_Carcinoma/
```

### 2. Run Neural Architecture Search
```bash
python train_nas_model_pytorch.py
```

This will:
- Automatically search through 15 architecture configurations
- Train each for 15 epochs to evaluate performance
- Select the best architecture based on validation accuracy
- Train the best architecture for 30 full epochs
- Generate performance plots and classification reports

### 3. Use Pre-trained Model for Inference
```python
import torch
from PIL import Image
import torchvision.transforms as transforms

# Load trained model
checkpoint = torch.load('lung_colon_cancer_nas_model_20250918_120553.pth')
model = CancerNASModel(checkpoint['config'], num_classes=5)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Predict
image = Image.open('path_to_image.jpg')
image_tensor = transform(image).unsqueeze(0)
with torch.no_grad():
    output = model(image_tensor)
    prediction = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(prediction).item()
```

## 📁 Project Structure

```
Model_Training/
├── train_nas_model_pytorch.py     # Main training script with NAS
├── requirements.txt               # Python dependencies
├── lung_colon_cancer_nas_model_*.pth      # Trained model weights
├── nas_training_plot_*.png        # Training history visualization
├── confusion_matrix_*.png         # Classification performance
├── nas_results_*.json            # NAS search results
├── data/                         # Dataset directory (not included)
│   └── Lung_and_Colon_Cancer/
├── results/                      # Generated results and plots
└── README.md                     # This file
```

## 🔬 Technical Innovations

### 1. **Automated Architecture Discovery**
- Eliminates manual architecture design bias
- Discovers optimal configurations specific to medical imaging
- Adapts to dataset characteristics automatically

### 2. **Advanced Regularization Strategy**
- Multi-level dropout (convolutional + dense layers)
- Smart data augmentation tailored for histopathology images
- Learning rate scheduling for stable convergence

### 3. **Efficient Search Algorithm**
- Hyperband-inspired pruning of poor performers
- Parallel configuration evaluation
- Memory-efficient model testing

## 📊 Results Interpretation

### Training Convergence Analysis
The model demonstrated excellent learning behavior:
- **Rapid initial convergence** (99%+ accuracy within 10 epochs)
- **Stable validation performance** throughout training
- **Minimal overfitting** despite high model capacity
- **Consistent improvement** across all cancer subtypes

### Generalization Capability
- **99.72% validation accuracy** indicates strong generalization
- **Balanced performance** across all 5 classes
- **Robust feature learning** unaffected by image variations
- **Clinical-grade reliability** suitable for medical applications

## 🎯 Potential Applications

### Medical Use Cases
1. **Pathologist Assistance**: Second opinion system for cancer diagnosis
2. **Telemedicine**: Remote cancer screening capabilities
3. **Medical Education**: Training tool for pathology students
4. **Research Acceleration**: High-throughput image analysis

### Technical Extensions
1. **Multi-modal Integration**: Combine with patient metadata
2. **Prognostic Prediction**: Survival rate estimation
3. **Treatment Response**: Monitor therapy effectiveness
4. **Early Detection**: Identify pre-cancerous conditions

## 🔮 Future Work

### Short-term Improvements
- [ ] Integration with DICOM standard for clinical compatibility
- [ ] Real-time inference optimization for clinical workflows
- [ ] Uncertainty quantification for prediction confidence
- [ ] Multi-center validation across different hospitals

### Long-term Vision
- [ ] Extension to other cancer types (breast, prostate, etc.)
- [ ] 3D histopathology volume analysis
- [ ] Integration with electronic health records
- [ ] Federated learning for privacy-preserving multi-institutional training

## 🤝 Contributing

We welcome contributions from researchers, developers, and medical professionals:

1. **Bug Reports**: Open an issue with detailed description
2. **Feature Requests**: Suggest new functionalities
3. **Code Contributions**: Submit pull requests with tests
4. **Dataset Contributions**: Help expand to other cancer types

## 📞 Contact & Support

- **Author**: Dhanush Saravanan
- **Email**: s.dhanush1106@gmail.com
- **Issues**: [GitHub Issues](https://github.com/Villwin007/Cancer-Image-Classification-with-AutoML-Driven-Neural-Architecture-Optimization/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Villwin007/Cancer-Image-Classification-with-AutoML-Driven-Neural-Architecture-Optimization/discussions)

## ⚠️ Disclaimer

This software is intended for **research purposes only**. It is not certified for clinical use and should not be used for actual medical diagnosis without proper validation and regulatory approval. Always consult qualified healthcare professionals for medical decisions.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**⭐ If you find this project useful, please consider starring the repository!**

---

*Last updated: September 2025*
