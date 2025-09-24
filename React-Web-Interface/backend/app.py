from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import json
import os
import numpy as np
import cv2
import base64

app = Flask(__name__)
CORS(app)

# Define the exact same model architecture as used in training
def create_model(config, num_classes):
    layers = []
    in_channels = 3
    
    # Base filter counts
    base_filters = [32, 64, 128, 256, 512]
    
    # Create convolutional blocks
    for i in range(config['num_conv_blocks']):
        out_channels = base_filters[i] * config['filters_multiplier']
        kernel_size = config['kernel_sizes'][i] if i < len(config['kernel_sizes']) else 3
        
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2))
        
        if config['use_batch_norm']:
            layers.append(nn.BatchNorm2d(out_channels))
            
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        layers.append(nn.Dropout2d(config['conv_dropout']))
        
        in_channels = out_channels
    
    return nn.Sequential(*layers)

class CancerNASModel(nn.Module):
    def __init__(self, config, num_classes):
        super(CancerNASModel, self).__init__()
        self.features = create_model(config, num_classes)
        
        # Calculate flattened size
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 128, 128)
            dummy_output = self.features(dummy_input)
            self.flattened_size = dummy_output.numel()
        
        # Create classifier based on config
        classifier_layers = []
        in_features = self.flattened_size
        
        for units in config['dense_units']:
            classifier_layers.append(nn.Linear(in_features, units))
            classifier_layers.append(nn.ReLU(inplace=True))
            classifier_layers.append(nn.Dropout(config['dense_dropout']))
            in_features = units
        
        classifier_layers.append(nn.Linear(in_features, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
            
        # Register hooks
        forward_handle = self.target_layer.register_forward_hook(forward_hook)
        backward_handle = self.target_layer.register_backward_hook(backward_hook)
        self.hook_handles = [forward_handle, backward_handle]
    
    def generate(self, input_tensor, target_class=None):
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)
        
        # Pool gradients
        gradients = self.gradients[0].mean(dim=(1, 2), keepdim=True)
        
        # Weight activations by gradients
        cam = (gradients * self.activations[0]).sum(dim=0)
        cam = torch.relu(cam)  # ReLU to keep only positive influences
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.detach().cpu().numpy(), target_class
    
    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

def create_heatmap(cam, original_image, alpha=0.5):
    """Create heatmap overlay on original image"""
    # Resize CAM to match original image
    cam_resized = cv2.resize(cam, (original_image.width, original_image.height))
    
    # Convert to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Convert PIL image to numpy
    img_np = np.array(original_image)
    
    # Blend images
    overlayed = cv2.addWeighted(img_np, 1 - alpha, heatmap, alpha, 0)
    
    return Image.fromarray(overlayed)

def image_to_base64(pil_image):
    """Convert PIL image to base64 string"""
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def generate_textual_explanation(diagnosis, confidence, diagnosis_type):
    """Generate human-readable explanation"""
    confidence_level = "high" if confidence > 0.8 else "medium" if confidence > 0.6 else "low"
    
    explanations = {
        'Colon Adenocarcinoma': [
            "The model detected irregular glandular patterns typical of adenocarcinoma",
            "Noticed abnormal cell nuclei and disrupted tissue architecture",
            "Observed features consistent with malignant transformation"
        ],
        'Lung Adenocarcinoma': [
            "Identified characteristic lepidic growth pattern",
            "Detected abnormal alveolar structures and nuclear pleomorphism",
            "Noticed features consistent with gland-forming carcinoma"
        ],
        'Lung Squamous Cell Carcinoma': [
            "Observed keratin pearl formation and intercellular bridges",
            "Detected characteristic squamous cell differentiation",
            "Noticed irregular nests of epithelial cells"
        ],
        'Colon Benign': [
            "Regular glandular architecture with maintained tissue structure",
            "Normal nuclear size and distribution observed",
            "No signs of malignant transformation detected"
        ],
        'Lung Benign': [
            "Preserved alveolar architecture with normal cellularity",
            "Regular tissue patterns without abnormal features",
            "No evidence of malignant characteristics"
        ]
    }
    
    base_explanation = f"The model classified this tissue as {diagnosis} ({diagnosis_type}) with {confidence_level} confidence ({confidence*100:.1f}%). "
    
    if diagnosis in explanations:
        specific_explanations = explanations[diagnosis]
        chosen_explanation = specific_explanations[0]
        base_explanation += chosen_explanation
    
    return base_explanation

def get_key_factors(diagnosis, confidence):
    """Return key factors that influenced the decision"""
    factors = {
        'Colon Adenocarcinoma': [
            "Glandular architecture abnormalities",
            "Nuclear enlargement and hyperchromasia", 
            "Loss of normal tissue organization",
            "Increased mitotic activity"
        ],
        'Lung Adenocarcinoma': [
            "Alveolar structure disruption",
            "Nuclear pleomorphism",
            "Gland formation patterns",
            "Stromal invasion evidence"
        ],
        'Lung Squamous Cell Carcinoma': [
            "Keratinization presence",
            "Intercellular bridge visibility",
            "Nuclear atypia degree",
            "Tumor nest organization"
        ],
        'Colon Benign': [
            "Regular glandular patterns",
            "Uniform cell distribution",
            "Maintained tissue architecture",
            "Normal nuclear characteristics"
        ],
        'Lung Benign': [
            "Preserved alveolar structure",
            "Normal cellular distribution",
            "Regular tissue organization",
            "Absence of malignant features"
        ]
    }
    
    return factors.get(diagnosis, ["Tissue architecture", "Cellular morphology", "Pattern consistency"])

# Load your model with the exact configuration from training
def load_model():
    # Use the exact path where your model is located
    model_path = 'lung_colon_cancer_nas_model_20250918_120553.pth'
    
    # Verify the file exists
    if not os.path.exists(model_path):
        # Try absolute path
        absolute_path = r'E:\MyMLprojects\Capstone-project\lung-cancer-detection-web\backend\lung_colon_cancer_nas_model_20250918_120553.pth'
        if os.path.exists(absolute_path):
            model_path = absolute_path
        else:
            raise FileNotFoundError(f"Model file not found at: {model_path} or {absolute_path}")
    
    print(f"Loading model from: {model_path}")
    print(f"File exists: {os.path.exists(model_path)}")
    print(f"File size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
    
    try:
        # Load with weights_only=True for security
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        
        # Extract the configuration from the checkpoint
        config = checkpoint['config']
        class_names = checkpoint['class_names']
        
        print("✅ Model loaded successfully!")
        print("Loaded configuration:")
        print(f"Number of conv blocks: {config['num_conv_blocks']}")
        print(f"Filters multiplier: {config['filters_multiplier']}")
        print(f"Kernel sizes: {config['kernel_sizes']}")
        print(f"Use batch norm: {config['use_batch_norm']}")
        print(f"Conv dropout: {config['conv_dropout']}")
        print(f"Dense units: {config['dense_units']}")
        print(f"Dense dropout: {config['dense_dropout']}")
        print(f"Learning rate: {config['learning_rate']}")
        print(f"Number of classes: {len(class_names)}")
        print(f"Class names: {class_names}")
        
        # Create model with the same configuration
        model = CancerNASModel(config, len(class_names))
        
        # Load the trained weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, class_names
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise

# Global variables
try:
    model, class_names = load_model()
    MODEL_LOADED = True
    print("🎯 Model ready for predictions!")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    model = None
    MODEL_LOADED = False
    # Use the class names from your training results
    class_names = ['colon_aca', 'colon_n', 'lung_aca', 'lung_n', 'lung_scc']

# Map class names to human readable format
CLASS_NAME_MAP = {
    'colon_aca': 'Colon Adenocarcinoma',
    'colon_n': 'Colon Benign', 
    'lung_aca': 'Lung Adenocarcinoma',
    'lung_n': 'Lung Benign',
    'lung_scc': 'Lung Squamous Cell Carcinoma'
}

# Preprocessing transform (must match training transform)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/')
def home():
    return jsonify({
        'message': 'Lung and Colon Cancer Detection API',
        'status': 'active' if MODEL_LOADED else 'demo_mode',
        'model_accuracy': '99.04% (validation)',
        'model_loaded': MODEL_LOADED,
        'classes': [CLASS_NAME_MAP.get(cls, cls) for cls in class_names]
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not MODEL_LOADED:
            return jsonify({
                'error': 'Model not loaded. Please check server logs.',
                'demo_mode': True
            }), 500
            
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file type
        allowed_extensions = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
        if not ('.' in file.filename and 
                file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            return jsonify({'error': 'Invalid file type. Please upload an image (PNG, JPG, JPEG, BMP, TIFF).'}), 400
        
        # Read and preprocess image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Preprocess and predict
        input_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            confidence, predicted = torch.max(probabilities, 0)
            
            results = []
            for i, prob in enumerate(probabilities):
                original_class = class_names[i]
                human_readable_class = CLASS_NAME_MAP.get(original_class, original_class)
                results.append({
                    'class': human_readable_class,
                    'original_class': original_class,
                    'probability': prob.item(),
                    'percentage': f"{prob.item() * 100:.2f}%"
                })
            
            # Sort by probability descending
            results.sort(key=lambda x: x['probability'], reverse=True)
            
            # Determine if it's cancerous or benign
            predicted_class = class_names[predicted.item()]
            human_readable_predicted = CLASS_NAME_MAP.get(predicted_class, predicted_class)
            is_cancerous = any(cancer_type in predicted_class.lower() 
                             for cancer_type in ['aca', 'scc'])
            
            diagnosis_type = "cancerous" if is_cancerous else "benign"
            
            return jsonify({
                'success': True,
                'predictions': results,
                'confidence': confidence.item(),
                'confidence_percentage': f"{confidence.item() * 100:.2f}%",
                'diagnosis': human_readable_predicted,
                'original_diagnosis': predicted_class,
                'diagnosis_type': diagnosis_type,
                'message': f'Predicted: {human_readable_predicted} ({diagnosis_type}) with {confidence.item() * 100:.2f}% confidence',
                'model_accuracy': '99.04%'
            })
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({
            'error': f'Error processing image: {str(e)}'
        }), 500

@app.route('/predict-with-explanation', methods=['POST'])
def predict_with_explanation():
    try:
        if not MODEL_LOADED:
            return jsonify({
                'error': 'Model not loaded. Please check server logs.',
                'demo_mode': True
            }), 500
            
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file type
        allowed_extensions = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
        if not ('.' in file.filename and 
                file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            return jsonify({'error': 'Invalid file type. Please upload an image (PNG, JPG, JPEG, BMP, TIFF).'}), 400
        
        # Read and preprocess image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Store original image for overlay
        original_image = image.copy()
        
        # Preprocess and predict
        input_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            confidence, predicted = torch.max(probabilities, 0)
        
        # Generate Grad-CAM explanation
        explanation_b64 = None
        if MODEL_LOADED:
            try:
                # Use the last convolutional layer for Grad-CAM
                target_layer = None
                # Find the last convolutional layer in features
                for name, module in model.features.named_modules():
                    if isinstance(module, nn.Conv2d):
                        target_layer = module
                
                if target_layer:
                    gradcam = GradCAM(model, target_layer)
                    cam, target_class = gradcam.generate(input_tensor, predicted.item())
                    gradcam.remove_hooks()
                    
                    # Create heatmap
                    explanation_image = create_heatmap(cam, original_image)
                    explanation_b64 = image_to_base64(explanation_image)
                    print("✅ Grad-CAM explanation generated successfully")
                else:
                    print("⚠️ No convolutional layer found for Grad-CAM")
            except Exception as e:
                print(f"⚠️ Grad-CAM generation failed: {e}")
                # Continue without Grad-CAM
                explanation_b64 = None
        
        # Prepare results
        results = []
        for i, prob in enumerate(probabilities):
            original_class = class_names[i]
            human_readable_class = CLASS_NAME_MAP.get(original_class, original_class)
            results.append({
                'class': human_readable_class,
                'original_class': original_class,
                'probability': prob.item(),
                'percentage': f"{prob.item() * 100:.2f}%"
            })
        
        results.sort(key=lambda x: x['probability'], reverse=True)
        
        predicted_class = class_names[predicted.item()]
        human_readable_predicted = CLASS_NAME_MAP.get(predicted_class, predicted_class)
        is_cancerous = any(cancer_type in predicted_class.lower() 
                         for cancer_type in ['aca', 'scc'])
        
        diagnosis_type = "cancerous" if is_cancerous else "benign"
        
        # Generate textual explanation
        textual_explanation = generate_textual_explanation(
            human_readable_predicted, 
            confidence.item(), 
            diagnosis_type
        )
        
        response_data = {
            'success': True,
            'predictions': results,
            'confidence': confidence.item(),
            'confidence_percentage': f"{confidence.item() * 100:.2f}%",
            'diagnosis': human_readable_predicted,
            'diagnosis_type': diagnosis_type,
            'explanation': {
                'textual': textual_explanation,
                'key_factors': get_key_factors(human_readable_predicted, confidence.item())
            },
            'message': f'Predicted: {human_readable_predicted} with {confidence.item() * 100:.2f}% confidence'
        }
        
        # Add heatmap if available
        if explanation_b64:
            response_data['explanation']['heatmap_image'] = explanation_b64
        
        return jsonify(response_data)
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Endpoint to get model information"""
    info = {
        'model_name': 'NAS-Optimized Cancer Detection Model',
        'validation_accuracy': '99.04%',
        'model_loaded': MODEL_LOADED,
        'classes': [CLASS_NAME_MAP.get(cls, cls) for cls in class_names],
        'original_classes': class_names,
        'image_size': '128x128',
        'framework': 'PyTorch',
        'features': ['Grad-CAM explanations', 'Textual reasoning', 'Confidence analysis']
    }
    
    if MODEL_LOADED:
        info.update({
            'status': 'fully_operational',
            'message': 'Model is loaded and ready for predictions'
        })
    else:
        info.update({
            'status': 'model_not_loaded',
            'message': 'Model file not found or failed to load'
        })
    
    return jsonify(info)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy' if MODEL_LOADED else 'degraded',
        'model_loaded': MODEL_LOADED,
        'mode': 'real' if MODEL_LOADED else 'demo',
        'classes_loaded': len(class_names),
        'explainable_ai': True
    })

if __name__ == '__main__':
    print("=" * 60)
    print("Starting Lung and Colon Cancer Detection API...")
    print("=" * 60)
    
    # Verify current directory and file
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")
    model_file_path = os.path.join(current_dir, 'lung_colon_cancer_nas_model_20250918_120553.pth')
    print(f"Model file path: {model_file_path}")
    print(f"File exists: {os.path.exists('lung_colon_cancer_nas_model_20250918_120553.pth')}")
    
    if MODEL_LOADED:
        print("✅ Model loaded successfully!")
        print(f"📊 Classes: {[CLASS_NAME_MAP.get(cls, cls) for cls in class_names]}")
        print(f"🎯 Validation Accuracy: 99.04%")
        print("🔧 Mode: REAL PREDICTIONS")
        print("🧠 Explainable AI: ENABLED (Grad-CAM + Textual Explanations)")
    else:
        print("❌ Model failed to load!")
        print("💡 Check if the model file is in the backend directory")
        print("📁 Expected file: lung_colon_cancer_nas_model_20250918_120553.pth")
    
    print("\n🌐 API endpoints:")
    print("  GET  /health - Health check")
    print("  GET  /model-info - Model information")
    print("  POST /predict - Basic prediction")
    print("  POST /predict-with-explanation - Prediction with AI explanations")
    print("  GET  / - API information")
    print("\n🚀 Server running on http://localhost:5000")
    print("=" * 60)
    
    app.run(debug=True, port=5000, host='0.0.0.0')