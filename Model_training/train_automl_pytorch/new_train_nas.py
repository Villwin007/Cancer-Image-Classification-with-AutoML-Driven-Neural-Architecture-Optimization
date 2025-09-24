import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from tqdm import tqdm
import itertools
import json
from datetime import datetime, timedelta
import random
import time  # Added for timing

# Try to import scikit-learn and seaborn
try:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import seaborn as sns
    SKLEARN_AVAILABLE = True
except ImportError:
    print("scikit-learn or seaborn not available. Installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn", "seaborn"])
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import seaborn as sns
    SKLEARN_AVAILABLE = True

# Check if CUDA is available and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Set seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)

# Parameters
DATASET_DIR = r"ENTER-PROJECT-DIRECTORY"
IMG_SIZE = 128
BATCH_SIZE = 32
NUM_EPOCHS_NAS = 15  # Shorter epochs for NAS search
NUM_EPOCHS_FINAL = 30  # Longer epochs for final training

# Data transformations with augmentation
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
try:
    full_dataset = datasets.ImageFolder(root=DATASET_DIR)
    
    # Split dataset (80% train, 20% validation)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # Apply transforms
    train_subset.dataset.transform = train_transform
    val_subset.dataset.transform = val_transform

    print(f"Number of classes: {len(full_dataset.classes)}")
    print(f"Class names: {full_dataset.classes}")
    print(f"Training samples: {len(train_subset)}")
    print(f"Validation samples: {len(val_subset)}")

except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Please check if the dataset path is correct and contains the proper folder structure")
    exit()

# Define search space for Neural Architecture Search
SEARCH_SPACE = {
    'num_conv_blocks': [3, 4, 5],
    'filters_multiplier': [1, 2, 4],  # Multiplier for base filter counts
    'kernel_sizes_options': [
        [3, 3, 3], 
        [3, 5, 3], 
        [5, 3, 5], 
        [3, 3, 5, 3], 
        [3, 5, 5, 3]
    ],
    'use_batch_norm': [True, False],
    'conv_dropout': [0.1, 0.2, 0.3],
    'dense_units_options': [
        [512, 256], 
        [256, 128], 
        [512], 
        [256]
    ],
    'dense_dropout': [0.3, 0.4, 0.5],
    'learning_rate': [1e-3, 1e-4, 1e-5]
}

# Flexible CNN model generator
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
    
    # Calculate flattened size
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        for layer in layers:
            dummy_input = layer(dummy_input)
        flattened_size = dummy_input.numel()
    
    # Create classifier
    classifier_layers = []
    in_features = flattened_size
    
    for units in config['dense_units']:
        classifier_layers.append(nn.Linear(in_features, units))
        classifier_layers.append(nn.ReLU(inplace=True))
        classifier_layers.append(nn.Dropout(config['dense_dropout']))
        in_features = units
    
    classifier_layers.append(nn.Linear(in_features, num_classes))
    
    return nn.Sequential(*layers), nn.Sequential(*classifier_layers)

# Complete model wrapper
class CancerNASModel(nn.Module):
    def __init__(self, config, num_classes):
        super(CancerNASModel, self).__init__()
        self.features, self.classifier = create_model(config, num_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

# Format time function
def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

# Training function with timing
def train_model(model, train_loader, val_loader, config, num_epochs, desc="Training"):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    best_model_state = None
    
    # Track epoch times
    epoch_times = []
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_bar = tqdm(train_loader, desc=f'{desc} Epoch {epoch+1}/{num_epochs}')
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            train_acc = 100. * correct / total
            train_bar.set_postfix({
                'Loss': f'{running_loss/(train_bar.n+1):.4f}',
                'Acc': f'{train_acc:.2f}%'
            })
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        
        # Calculate estimated time remaining
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = num_epochs - (epoch + 1)
        eta = avg_epoch_time * remaining_epochs
        
        print(f'Epoch {epoch+1}/{num_epochs}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | '
              f'Epoch Time: {format_time(epoch_time)} | '
              f'ETA: {format_time(eta)}')
    
    # Load best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    total_time = time.time() - start_time
    
    return model, train_losses, val_losses, train_accs, val_accs, best_val_acc, epoch_times, total_time

# Generate random configurations from search space
def generate_configurations(search_space, num_configs):
    configs = []
    for _ in range(num_configs):
        config = {}
        for key, values in search_space.items():
            if key.endswith('_options'):
                # For options that are lists of sequences
                config_key = key.replace('_options', '')
                config[config_key] = random.choice(values)
            else:
                # For regular parameters
                config[key] = random.choice(values)
        configs.append(config)
    return configs

# Neural Architecture Search with timing
def neural_architecture_search(search_space, num_configs=20):
    print("Starting Neural Architecture Search...")
    print(f"Testing {num_configs} configurations with {NUM_EPOCHS_NAS} epochs each")
    
    # Create data loaders for NAS (smaller batch size for faster experimentation)
    nas_train_loader = DataLoader(
        train_subset, 
        batch_size=16,  # Smaller batch for faster experimentation
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    nas_val_loader = DataLoader(
        val_subset, 
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Generate configurations
    configurations = generate_configurations(search_space, num_configs)
    results = []
    
    # Track configuration times
    config_times = []
    nas_start_time = time.time()
    
    for i, config in enumerate(configurations):
        config_start_time = time.time()
        print(f"\n{'='*80}")
        print(f"Testing configuration {i+1}/{num_configs}")
        print(f"Config: {json.dumps(config, indent=2, default=str)}")
        
        try:
            # Create and train model
            model = CancerNASModel(config, len(full_dataset.classes))
            model, _, _, _, _, val_acc, epoch_times, config_time = train_model(
                model, nas_train_loader, nas_val_loader, config, 
                NUM_EPOCHS_NAS, desc=f"NAS Config {i+1}"
            )
            
            results.append({
                'config': config,
                'val_accuracy': val_acc,
                'model': model,
                'epoch_times': epoch_times,
                'total_time': config_time
            })
            
            # Calculate estimated time for remaining configurations
            avg_config_time = sum(config_times + [config_time]) / (len(config_times) + 1)
            remaining_configs = num_configs - (i + 1)
            eta = avg_config_time * remaining_configs
            
            print(f"Configuration {i+1} completed in {format_time(config_time)} "
                  f"with validation accuracy: {val_acc:.2f}%")
            print(f"Average epoch time: {format_time(sum(epoch_times)/len(epoch_times))}")
            print(f"Estimated time remaining for NAS: {format_time(eta)}")
            
            config_times.append(config_time)
            
        except Exception as e:
            print(f"Error with configuration {i+1}: {e}")
            continue
    
    # Sort results by validation accuracy
    results.sort(key=lambda x: x['val_accuracy'], reverse=True)
    
    nas_total_time = time.time() - nas_start_time
    print(f"\nNAS completed in {format_time(nas_total_time)}")
    
    return results, nas_total_time

# Main execution
if __name__ == "__main__":
    # Start total timer
    total_start_time = time.time()
    
    # Perform NAS
    nas_results, nas_time = neural_architecture_search(SEARCH_SPACE, num_configs=15)
    
    # Display top configurations
    print("\n" + "="*80)
    print("TOP CONFIGURATIONS FROM NEURAL ARCHITECTURE SEARCH")
    print("="*80)
    
    for i, result in enumerate(nas_results[:5]):
        print(f"\nTop {i+1}: Validation Accuracy = {result['val_accuracy']:.2f}%")
        print(f"Training Time: {format_time(result['total_time'])}")
        print(f"Average Epoch Time: {format_time(sum(result['epoch_times'])/len(result['epoch_times']))}")
        print(f"Configuration: {json.dumps(result['config'], indent=2, default=str)}")
    
    # Get best configuration
    if nas_results:
        best_result = nas_results[0]
        best_config = best_result['config']
        best_model = best_result['model']
        
        print(f"\nBest configuration validation accuracy: {best_result['val_accuracy']:.2f}%")
        print(f"NAS training time: {format_time(best_result['total_time'])}")
        
        # Train best model with full epochs and regular batch size
        print("\n" + "="*80)
        print("TRAINING BEST MODEL WITH FULL EPOCHS")
        print("="*80)
        
        # Create full data loaders
        train_loader = DataLoader(
            train_subset, 
            batch_size=BATCH_SIZE, 
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_subset, 
            batch_size=BATCH_SIZE, 
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        # Train best model
        final_model, train_losses, val_losses, train_accs, val_accs, final_val_acc, epoch_times, final_time = train_model(
            best_model, train_loader, val_loader, best_config, 
            NUM_EPOCHS_FINAL, desc="Final Training"
        )
        
        # Save the best model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"lung_colon_cancer_nas_model_{timestamp}.pth"
        torch.save({
            'model_state_dict': final_model.state_dict(),
            'config': best_config,
            'val_accuracy': final_val_acc,
            'class_names': full_dataset.classes,
            'training_time': final_time,
            'epoch_times': epoch_times
        }, model_path)
        
        print(f"\nBest model saved as: {model_path}")
        print(f"Final validation accuracy: {final_val_acc:.2f}%")
        print(f"Final training time: {format_time(final_time)}")
        print(f"Average epoch time: {format_time(sum(epoch_times)/len(epoch_times))}")
        
        # Plot training history
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Train Accuracy')
        plt.plot(val_accs, label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plot_path = f"nas_training_plot_{timestamp}.png"
        plt.savefig(plot_path)
        print(f"Training plot saved as: {plot_path}")
        
        # Evaluate on validation set
        if SKLEARN_AVAILABLE:
            final_model.eval()
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = final_model(images)
                    _, preds = torch.max(outputs, 1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            # Generate classification report
            print("\n" + "="*80)
            print("CLASSIFICATION REPORT")
            print("="*80)
            print(classification_report(all_labels, all_preds, target_names=full_dataset.classes))
            
            # Confusion matrix
            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=full_dataset.classes, 
                        yticklabels=full_dataset.classes)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            cm_path = f"confusion_matrix_{timestamp}.png"
            plt.savefig(cm_path)
            print(f"Confusion matrix saved as: {cm_path}")
        
        # Save NAS results with timing information
        results_path = f"nas_results_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump([{
                'config': r['config'],
                'val_accuracy': r['val_accuracy'],
                'total_time': r['total_time'],
                'avg_epoch_time': sum(r['epoch_times'])/len(r['epoch_times']) if 'epoch_times' in r else 0
            } for r in nas_results], f, indent=2, default=str)
        
        print(f"NAS results saved as: {results_path}")
        
        # Calculate and display total time
        total_time = time.time() - total_start_time
        print(f"\nTotal process time: {format_time(total_time)}")
        print(f"NAS time: {format_time(nas_time)}")
        print(f"Final training time: {format_time(final_time)}")
        print("\nNeural Architecture Search completed successfully!")
    else:
        print("No successful configurations found during NAS.")
