import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import time
import json
from datetime import datetime

import yaml

# Fix import issues - add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    # Try absolute imports first
    from srce.model import build_model
    from srce.dataset import create_data_loaders
    from srce.helpers import set_seed, plot_training_history, plot_confusion_matrix, load_config
except ImportError:
    # Fallback to direct imports
    try:
        from model import build_model
        from dataset import create_data_loaders
        from helpers import set_seed, plot_training_history, plot_confusion_matrix, load_config
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("📁 Current directory:", current_dir)
        print("📁 Parent directory:", parent_dir)
        print("🔍 Python path:")
        for path in sys.path:
            print(f"   - {path}")
        raise

class Trainer:
    """Training class for mental health classification model."""
    
    def __init__(self, config, device=None):
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        print(f"✅ Trainer initialized with device: {self.device}")
        
        # Create directories
        os.makedirs(config['paths']['log_dir'], exist_ok=True)
        os.makedirs(config['paths']['model_save_dir'], exist_ok=True)
        
        # Set seed
        set_seed(42)
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # Training history
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'learning_rates': []
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_model_path = None

    def setup_training(self):
        """Setup model, optimizer, criterion, and data loaders."""
        # Build model
        self.model = build_model(self.config).to(self.device)
        print(f"✅ Model built: {self.config['model']['backbone']}")
        
        # Data loaders
        try:
            self.train_loader, self.val_loader, self.test_loader, self.class_names = \
                create_data_loaders(self.config)
            print(f"✅ Data loaders created")
            print(f"   Classes: {self.class_names}")
            print(f"   Train: {len(self.train_loader.dataset)} samples")
            print(f"   Val: {len(self.val_loader.dataset)} samples")
            print(f"   Test: {len(self.test_loader.dataset)} samples")
        except Exception as e:
            print(f"❌ Error creating data loaders: {e}")
            print("📝 Creating sample dataset...")
            self.create_sample_dataset()
            self.train_loader, self.val_loader, self.test_loader, self.class_names = \
                create_data_loaders(self.config)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=0.0001
        )
        
        # Scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
           
        )
        
        # Criterion
        self.criterion = nn.CrossEntropyLoss()
        
        print("✅ Training setup completed")

    def create_sample_dataset(self):
        """Create sample dataset if no data exists."""
        from srce.helpers import create_sample_drawing, create_sample_handwriting
        
        data_dir = self.config['paths']['raw_data_dir']
        os.makedirs(data_dir, exist_ok=True)
        
        classes = self.config['classes']
        print(f"🎨 Creating sample dataset in {data_dir}...")
        
        for class_id, class_name in classes.items():
            class_dir = os.path.join(data_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            for i in range(10):  # Create 10 samples per class
                # Create drawing sample
                drawing_img = create_sample_drawing(
                    drawing_type="house",
                    style=class_name.lower()
                )
                drawing_path = os.path.join(class_dir, f"drawing_{i+1:03d}.png")
                drawing_img.save(drawing_path)
                
                # Create handwriting sample  
                handwriting_img = create_sample_handwriting(
                    f"Sample {i+1}",
                    style=class_name.lower()
                )
                handwriting_path = os.path.join(class_dir, f"handwriting_{i+1:03d}.png")
                handwriting_img.save(handwriting_path)
        
        print(f"✅ Sample dataset created with {len(classes) * 10 * 2} images")

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} Training')
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        self.history['train_loss'].append(epoch_loss)
        self.history['train_acc'].append(epoch_acc)
        
        return epoch_loss, epoch_acc

    def validate_epoch(self, epoch):
        """Validate for one epoch."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1} Validation')
            
            for batch in pbar:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        self.history['val_loss'].append(epoch_loss)
        self.history['val_acc'].append(epoch_acc)
        
        return epoch_loss, epoch_acc

    def train(self):
        """Main training loop."""
        print("🚀 Starting training...")
        self.setup_training()
        
        start_time = time.time()
        patience_counter = 0
        
        for epoch in range(self.config['training']['epochs']):
            print(f"\n📚 Epoch {epoch+1}/{self.config['training']['epochs']}")
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate_epoch(epoch)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Save learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rates'].append(current_lr)
            
            # Print epoch summary
            print(f"📊 Epoch {epoch+1} Summary:")
            print(f"   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"   Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"   Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                best_model_path = os.path.join(
                    self.config['paths']['model_save_dir'],
                    'model_best.pth'
                )
                torch.save(self.model.state_dict(), best_model_path)
                self.best_model_path = best_model_path
                print(f"💾 New best model saved with val_acc: {self.best_val_acc:.2f}%")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config['training'].get('patience', 10):
                print(f"🛑 Early stopping at epoch {epoch+1}")
                break
        
        # Training completed
        training_time = time.time() - start_time
        print(f"\n✅ Training completed in {training_time:.2f} seconds")
        print(f"🏆 Best validation accuracy: {self.best_val_acc:.2f}%")
        
        # Save final model
        final_model_path = os.path.join(
            self.config['paths']['model_save_dir'],
            'model_final.pth'
        )
        torch.save(self.model.state_dict(), final_model_path)
        print(f"💾 Final model saved: {final_model_path}")
        
        # Save training history
        history_path = os.path.join(
            self.config['paths']['log_dir'],
            'training_history.json'
        )
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"💾 Training history saved: {history_path}")
        
        return self.history

def main():
    """Main training function."""
    # First, create necessary directories
    os.makedirs("configs", exist_ok=True)
    os.makedirs("data/images", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    config_path = "configs/default.yaml"
    
    # Create default config if it doesn't exist
    if not os.path.exists(config_path):
        print("⚠️  Config file not found, creating default config...")
        
        default_config = {
            'paths': {
                'raw_data_dir': 'data/images',
                'processed_dir': 'data/processed', 
                'model_save_dir': 'models',
                'log_dir': 'logs'
            },
            'dataset': {
                'image_size': [224, 224],
                'channels': 3,
                'train_ratio': 0.8,
                'val_ratio': 0.1,
                'test_ratio': 0.1,
                'augmentation': True
            },
            'classes': {
                '0': 'Healthy',
                '1': 'Depression',
                '2': 'Anxiety', 
                '3': 'Bipolar_Disorder',
                '4': 'OCD',
                '5': 'PTSD'
            },
            'model': {
                'backbone': 'resnet18',
                'num_classes': 6,
                'pretrained': True,
                'dropout_rate': 0.3,
                'feature_dim': 512
            },
            'training': {
                'epochs': 10,
                'batch_size': 8,
                'learning_rate': 0.001,
                'patience': 5
            },
            'optimizer': {
                'name': 'adam',
                'weight_decay': 0.0001
            },
            'scheduler': {
                'name': 'reduce_on_plateau',
                'factor': 0.5,
                'patience': 3
            },
            'logging': {
                'log_interval': 10,
                'save_interval': 5
            }
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        print(f"📁 Default config created: {config_path}")
    
    # Load config
    try:
        config = load_config(config_path)
        print("✅ Config loaded successfully")
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return
    
    # Train model
    try:
        trainer = Trainer(config)
        history = trainer.train()
        print("🎉 Training completed successfully!")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()