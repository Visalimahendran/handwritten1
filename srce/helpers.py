import os
import random
import numpy as np
import torch
import yaml
import json
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageOps
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from torchvision import transforms
import torch.nn.functional as F
from datetime import datetime
import hashlib

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path):
    """Load YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def save_config(config, save_path):
    """Save configuration to YAML file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def pil_transform(image, config=None):
    """Transform PIL image to tensor with proper preprocessing."""
    if config is None:
        # Default transform
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        # Use config-based transform
        image_size = config['dataset'].get('image_size', [224, 224])
        grayscale = config.get('handwriting', {}).get('grayscale', False)
        
        if grayscale:
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    return transform(image)

def create_sample_handwriting(text="Sample Writing", size=(400, 200), style="normal"):
    """Create sample handwriting images for testing."""
    img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(img)
    
    try:
        # Try to use a handwriting-like font
        font_size = random.randint(20, 30)
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        try:
            font = ImageFont.load_default()
        except:
            font = None
    
    # Position text
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2
    
    # Add some randomness to simulate natural handwriting
    if style == "depressed":
        # Smaller, tighter writing
        font_size = max(15, font_size - 5)
        color = "darkgray"
        y_variation = 2
    elif style == "anxious":
        # Irregular, shaky writing
        color = "black"
        y_variation = 5
    elif style == "manic":
        # Large, expansive writing
        font_size = min(35, font_size + 8)
        color = "darkblue"
        y_variation = 1
    else:
        color = "black"
        y_variation = 1
    
    # Draw text with variations
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # Draw multiple times with slight offsets to create natural variation
    for i in range(3):
        offset_x = random.randint(-1, 1)
        offset_y = random.randint(-y_variation, y_variation)
        draw.text((x + offset_x, y + offset_y), text, fill=color, font=font)
    
    # Add some noise
    img_array = np.array(img)
    noise = np.random.randint(0, 10, img_array.shape, dtype=np.uint8)
    img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return Image.fromarray(img_array)

def create_sample_drawing(drawing_type="house", size=(224, 224), style="normal"):
    """Create sample drawings for testing with different mental health styles."""
    img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(img)
    
    if drawing_type == "house":
        if style == "depressed":
            # Small, simple house with dark colors
            color = "gray"
            house_color = "darkgray"
            # Small house
            draw.rectangle([80, 120, 140, 160], outline=color, width=2, fill=house_color)
            draw.polygon([70, 120, 150, 120, 110, 90], outline=color, width=2, fill=house_color)
            draw.rectangle([105, 140, 115, 160], outline=color, width=1)  # Small door
        elif style == "anxious":
            # Chaotic, uneven house
            color = "black"
            # Uneven house body
            draw.polygon([50, 100, 150, 110, 150, 180, 50, 170], outline=color, width=3)
            # Irregular roof
            draw.polygon([40, 100, 160, 110, 110, 70], outline=color, width=3)
            # Off-center door
            draw.rectangle([85, 140, 105, 180], outline=color, width=2)
        elif style == "manic":
            # Large, detailed house with multiple colors
            colors = ["red", "blue", "green", "yellow"]
            # Large house body
            draw.rectangle([30, 80, 190, 180], outline=colors[0], width=3, fill="lightyellow")
            # Elaborate roof
            draw.polygon([20, 80, 200, 80, 160, 40, 60, 40], outline=colors[1], width=3, fill="lightblue")
            # Multiple windows and details
            draw.rectangle([50, 100, 80, 130], outline=colors[2], width=2, fill="white")
            draw.rectangle([140, 100, 170, 130], outline=colors[2], width=2, fill="white")
            draw.rectangle([95, 140, 125, 180], outline=colors[3], width=2, fill="brown")
        else:
            # Normal house
            draw.rectangle([50, 100, 150, 180], outline='black', width=3, fill="lightblue")  # House body
            draw.polygon([40, 100, 160, 100, 100, 60], outline='black', width=3, fill="red")  # Roof
            draw.rectangle([80, 130, 120, 180], outline='black', width=3, fill="brown")  # Door
            draw.rectangle([60, 120, 80, 140], outline='black', width=2, fill="white")  # Window
            draw.rectangle([120, 120, 140, 140], outline='black', width=2, fill="white")  # Window
    
    elif drawing_type == "tree":
        if style == "depressed":
            # Bare tree
            draw.rectangle([95, 140, 105, 180], outline="brown", width=3)  # Trunk
            # Sparse leaves
            draw.ellipse([80, 110, 120, 150], outline="darkgreen", width=2, fill="lightgreen")
        elif style == "anxious":
            # Messy, overlapping tree
            draw.rectangle([90, 140, 110, 180], outline="black", width=3)
            # Multiple overlapping circles for leaves
            for i in range(5):
                x_offset = random.randint(-15, 15)
                y_offset = random.randint(-10, 10)
                draw.ellipse([70+x_offset, 80+y_offset, 130+x_offset, 140+y_offset], 
                           outline="green", width=2)
        else:
            # Normal tree
            draw.rectangle([100, 140, 110, 180], outline="brown", width=3, fill="brown")  # Trunk
            draw.ellipse([70, 80, 140, 150], outline="green", width=3, fill="lightgreen")  # Leaves
    
    elif drawing_type == "person":
        if style == "depressed":
            # Small, simplified person
            draw.ellipse([95, 50, 105, 60], outline="black", width=1, fill="lightblue")  # Head
            draw.line([100, 60, 100, 80], fill="black", width=2)  # Body
            draw.line([100, 65, 90, 75], fill="black", width=1)  # Arms
            draw.line([100, 65, 110, 75], fill="black", width=1)
            draw.line([100, 80, 95, 95], fill="black", width=1)  # Legs
            draw.line([100, 80, 105, 95], fill="black", width=1)
        elif style == "manic":
            # Large, detailed person
            draw.ellipse([90, 40, 110, 60], outline="black", width=2, fill="yellow")  # Head
            draw.line([100, 60, 100, 100], fill="red", width=3)  # Body
            draw.line([100, 70, 80, 90], fill="blue", width=2)  # Arms
            draw.line([100, 70, 120, 90], fill="blue", width=2)
            draw.line([100, 100, 85, 130], fill="green", width=2)  # Legs
            draw.line([100, 100, 115, 130], fill="green", width=2)
            # Facial features
            draw.ellipse([95, 47, 97, 49], fill="black")  # Eyes
            draw.ellipse([103, 47, 105, 49], fill="black")
            draw.arc([95, 50, 105, 55], 0, 180, fill="black", width=1)  # Smile
        else:
            # Normal person
            draw.ellipse([95, 50, 105, 60], outline="black", width=2, fill="peachpuff")  # Head
            draw.line([100, 60, 100, 100], fill="black", width=2)  # Body
            draw.line([100, 70, 80, 90], fill="black", width=2)  # Arms
            draw.line([100, 70, 120, 90], fill="black", width=2)
            draw.line([100, 100, 85, 130], fill="black", width=2)  # Legs
            draw.line([100, 100, 115, 130], fill="black", width=2)
    
    elif drawing_type == "abstract":
        # Abstract drawing that can show emotional state
        if style == "depressed":
            # Minimal, dark abstract
            colors = ["gray", "darkblue", "black"]
            for i in range(3):
                x = random.randint(20, size[0]-20)
                y = random.randint(20, size[1]-20)
                radius = random.randint(5, 15)
                draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                           outline=colors[i], width=1)
        elif style == "anxious":
            # Chaotic lines
            for i in range(10):
                x1, y1 = random.randint(10, size[0]-10), random.randint(10, size[1]-10)
                x2, y2 = random.randint(10, size[0]-10), random.randint(10, size[1]-10)
                draw.line([x1, y1, x2, y2], fill="black", width=1)
        elif style == "manic":
            # Colorful, energetic abstract
            colors = ["red", "yellow", "blue", "green", "orange", "purple"]
            for i in range(15):
                shape_type = random.choice(["circle", "rectangle", "line"])
                color = random.choice(colors)
                if shape_type == "circle":
                    x, y = random.randint(20, size[0]-20), random.randint(20, size[1]-20)
                    radius = random.randint(10, 30)
                    draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                               outline=color, width=2, fill=color if random.random() > 0.5 else None)
                elif shape_type == "rectangle":
                    x1, y1 = random.randint(10, size[0]-30), random.randint(10, size[1]-30)
                    x2, y2 = x1 + random.randint(20, 50), y1 + random.randint(20, 50)
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=2, 
                                 fill=color if random.random() > 0.5 else None)
    
    return img

def plot_training_history(history, save_path=None):
    """Plot training history with multiple metrics."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plot
    ax1.plot(history.get('train_loss', []), label='Training Loss', linewidth=2)
    ax1.plot(history.get('val_loss', []), label='Validation Loss', linewidth=2)
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(history.get('train_acc', []), label='Training Accuracy', linewidth=2)
    ax2.plot(history.get('val_acc', []), label='Validation Accuracy', linewidth=2)
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Learning rate plot
    if 'learning_rates' in history:
        ax3.plot(history['learning_rates'], label='Learning Rate', color='green', linewidth=2)
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Additional metrics if available
    if 'train_f1' in history or 'val_f1' in history:
        if 'train_f1' in history:
            ax4.plot(history['train_f1'], label='Training F1-Score', linewidth=2)
        if 'val_f1' in history:
            ax4.plot(history['val_f1'], label='Validation F1-Score', linewidth=2)
        ax4.set_title('F1-Score')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('F1-Score')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """Plot confusion matrix with better visualization."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'shrink': 0.8})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # Print detailed classification report
    print("\n" + "="*60)
    print("📊 CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=class_names))

def plot_roc_curve(y_true, y_scores, class_names, save_path=None):
    """Plot ROC curves for multi-class classification."""
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot all ROC curves
    plt.figure(figsize=(10, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
    
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Multi-class ROC Curves', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def create_sample_dataset(config, output_dir="data/samples", num_samples_per_class=50):
    """Create a sample dataset for testing and demonstration."""
    os.makedirs(output_dir, exist_ok=True)
    
    classes = config['classes']
    samples_created = []
    
    print(f"🎨 Creating sample dataset in {output_dir}...")
    
    for class_id, class_name in classes.items():
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        print(f"   Creating samples for {class_name}...")
        
        for i in range(num_samples_per_class):
            # Create both handwriting and drawing samples
            handwriting_img = create_sample_handwriting(
                f"Sample {i+1}", 
                style=class_name.lower()
            )
            drawing_img = create_sample_drawing(
                random.choice(["house", "tree", "person", "abstract"]),
                style=class_name.lower()
            )
            
            # Save samples
            hw_path = os.path.join(class_dir, f"handwriting_{i+1:03d}.png")
            drawing_path = os.path.join(class_dir, f"drawing_{i+1:03d}.png")
            
            handwriting_img.save(hw_path)
            drawing_img.save(drawing_path)
            
            samples_created.extend([hw_path, drawing_path])
    
    print(f"✅ Created {len(samples_created)} sample images")
    return samples_created

def calculate_model_metrics(model, dataloader, device, num_classes):
    """Calculate comprehensive model metrics."""
    model.eval()
    all_preds = []
    all_targets = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            targets = batch['label'].to(device)
            
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate metrics
    accuracy = np.mean(all_preds == all_targets)
    
    metrics = {
        'accuracy': accuracy,
        'predictions': all_preds,
        'targets': all_targets,
        'probabilities': all_probabilities
    }
    
    return metrics

def save_checkpoint(model, optimizer, scheduler, epoch, metrics, filepath):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(checkpoint, filepath)
    print(f"💾 Checkpoint saved: {filepath}")

def load_checkpoint(filepath, model, optimizer=None, scheduler=None):
    """Load training checkpoint."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"📥 Checkpoint loaded: {filepath} (epoch {checkpoint['epoch']})")
    return checkpoint

def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def create_experiment_name(config):
    """Create a unique experiment name based on config."""
    model_name = config['model'].get('backbone', 'unknown')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a hash of important config parameters
    config_str = f"{model_name}_{config['training']['epochs']}_{config['training']['learning_rate']}"
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    return f"exp_{model_name}_{timestamp}_{config_hash}"

def visualize_data_samples(dataloader, class_names, num_samples=8, save_path=None):
    """Visualize samples from dataloader."""
    # Get a batch of samples
    batch = next(iter(dataloader))
    images = batch['image']
    labels = batch['label']
    
    # Denormalize images
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    images = images * std + mean
    images = torch.clamp(images, 0, 1)
    
    # Plot samples
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(min(num_samples, len(images))):
        img = images[i].permute(1, 2, 0).numpy()
        label = labels[i].item()
        
        axes[i].imshow(img)
        axes[i].set_title(f'{class_names[label]}', fontsize=10)
        axes[i].axis('off')
    
    plt.suptitle('Data Samples', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def analyze_predictions_distribution(predictions, class_names, save_path=None):
    """Analyze and visualize prediction distribution."""
    pred_counts = {name: 0 for name in class_names}
    
    for pred in predictions:
        if pred < len(class_names):
            pred_counts[class_names[pred]] += 1
    
    # Plot distribution
    plt.figure(figsize=(10, 6))
    names = list(pred_counts.keys())
    counts = list(pred_counts.values())
    
    bars = plt.bar(names, counts, color=plt.cm.Set3(np.linspace(0, 1, len(names))))
    plt.title('Prediction Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Classes', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def create_feature_importance_plot(feature_names, importance_scores, top_k=15, save_path=None):
    """Create feature importance plot."""
    # Sort features by importance
    indices = np.argsort(importance_scores)[-top_k:]
    
    plt.figure(figsize=(10, 8))
    y_pos = np.arange(len(indices))
    
    plt.barh(y_pos, importance_scores[indices], color=plt.cm.viridis(np.linspace(0, 1, len(indices))))
    plt.yticks(y_pos, [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title(f'Top {top_k} Most Important Features', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def print_model_summary(model, input_size=(1, 3, 224, 224)):
    """Print model summary similar to torchsummary."""
    try:
        from torchsummary import summary
        summary(model, input_size=input_size[1:])
    except ImportError:
        print("📋 Model Architecture:")
        print(model)
        print(f"\n🔢 Trainable parameters: {count_parameters(model):,}")

def create_progress_tracker(total_epochs):
    """Create a progress tracker for training."""
    from tqdm import tqdm
    
    class ProgressTracker:
        def __init__(self, total_epochs):
            self.total_epochs = total_epochs
            self.current_epoch = 0
            self.best_accuracy = 0.0
            self.history = {
                'train_loss': [], 'train_acc': [],
                'val_loss': [], 'val_acc': []
            }
        
        def update(self, train_loss, train_acc, val_loss, val_acc):
            self.current_epoch += 1
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
            
            # Print progress
            print(f"📈 Epoch {self.current_epoch}/{self.total_epochs}")
            print(f"   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"   Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"   Best Val Acc: {self.best_accuracy:.2f}%")
            print("-" * 50)
    
    return ProgressTracker(total_epochs)

def format_time(seconds):
    """Format seconds into human readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

def check_data_leakage(train_files, val_files, test_files):
    """Check for data leakage between splits."""
    train_set = set(train_files)
    val_set = set(val_files)
    test_set = set(test_files)
    
    # Check intersections
    train_val_leakage = train_set.intersection(val_set)
    train_test_leakage = train_set.intersection(test_set)
    val_test_leakage = val_set.intersection(test_set)
    
    leakage_found = False
    
    if train_val_leakage:
        print(f"⚠️  Data leakage found: {len(train_val_leakage)} files in both train and val sets")
        leakage_found = True
    
    if train_test_leakage:
        print(f"⚠️  Data leakage found: {len(train_test_leakage)} files in both train and test sets")
        leakage_found = True
    
    if val_test_leakage:
        print(f"⚠️  Data leakage found: {len(val_test_leakage)} files in both val and test sets")
        leakage_found = True
    
    if not leakage_found:
        print("✅ No data leakage detected")
    
    return not leakage_found

# Label binarize function for ROC curve
def label_binarize(y, classes):
    """Binarize labels for multi-class ROC."""
    n_classes = len(classes)
    y_bin = np.zeros((len(y), n_classes))
    for i, label in enumerate(y):
        y_bin[i, label] = 1
    return y_bin

# Color maps for different mental health conditions
MENTAL_HEALTH_COLORS = {
    'Healthy': '#2E8B57',        # Sea Green
    'Depression': '#4682B4',     # Steel Blue
    'Anxiety': '#FF8C00',        # Dark Orange
    'Bipolar_Disorder': '#8A2BE2', # Blue Violet
    'OCD': '#DC143C',            # Crimson
    'PTSD': '#FFD700',           # Gold
    'ADHD': '#20B2AA',           # Light Sea Green
    'Schizophrenia': '#8B4513'   # Saddle Brown
}

def get_mental_health_color(condition):
    """Get color for mental health condition."""
    return MENTAL_HEALTH_COLORS.get(condition, '#666666')

if __name__ == "__main__":
    # Test the helper functions
    config = {
        'classes': {
            '0': 'Healthy',
            '1': 'Depression', 
            '2': 'Anxiety',
            '3': 'Bipolar_Disorder'
        }
    }
    
    # Create sample images
    handwriting = create_sample_handwriting("Test Writing", style="normal")
    drawing = create_sample_drawing("house", style="normal")
    
    print("✅ Helper functions tested successfully!")
    print(f"   Handwriting sample size: {handwriting.size}")
    print(f"   Drawing sample size: {drawing.size}")