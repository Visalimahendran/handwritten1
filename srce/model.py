import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet50_Weights, EfficientNet_B0_Weights

class MentalHealthClassifier(nn.Module):
    """Deep learning model for mental health classification from drawings and handwriting."""
    
    def __init__(self, config):
        super(MentalHealthClassifier, self).__init__()
        self.config = config
        self.backbone_name = config['model']['backbone']
        
        # Backbone model
        if self.backbone_name == 'resnet18':
            if config['model']['pretrained']:
                self.backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            else:
                self.backbone = models.resnet18(weights=None)
            
            # Modify final layers
            in_features = self.backbone.fc.in_features
            
            # Custom classifier head
            self.backbone.fc = nn.Sequential(
                nn.Dropout(config['model']['dropout_rate']),
                nn.Linear(in_features, config['model']['feature_dim']),
                nn.ReLU(inplace=True),
                nn.Dropout(config['model']['dropout_rate'] / 2),
                nn.Linear(config['model']['feature_dim'], config['model']['num_classes'])
            )
        
        elif self.backbone_name == 'resnet50':
            if config['model']['pretrained']:
                self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            else:
                self.backbone = models.resnet50(weights=None)
            
            in_features = self.backbone.fc.in_features
            
            # Custom classifier head
            self.backbone.fc = nn.Sequential(
                nn.Dropout(config['model']['dropout_rate']),
                nn.Linear(in_features, config['model']['feature_dim']),
                nn.ReLU(inplace=True),
                nn.Dropout(config['model']['dropout_rate'] / 2),
                nn.Linear(config['model']['feature_dim'], config['model']['num_classes'])
            )
        
        elif self.backbone_name == 'efficientnet_b0':
            if config['model']['pretrained']:
                self.backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            else:
                self.backbone = models.efficientnet_b0(weights=None)
            
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(config['model']['dropout_rate']),
                nn.Linear(in_features, config['model']['feature_dim']),
                nn.ReLU(inplace=True),
                nn.Dropout(config['model']['dropout_rate'] / 2),
                nn.Linear(config['model']['feature_dim'], config['model']['num_classes'])
            )
        
        elif self.backbone_name == 'simple_cnn':
            # Simple CNN for smaller datasets
            self.backbone = nn.Sequential(
                # First conv block
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                
                # Second conv block
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                
                # Third conv block
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                
                # Fourth conv block
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                
                nn.Flatten()
            )
            
            # Calculate the input features for classifier
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)
                in_features = self.backbone(dummy_input).shape[1]
            
            self.classifier = nn.Sequential(
                nn.Dropout(config['model']['dropout_rate']),
                nn.Linear(in_features, config['model']['feature_dim']),
                nn.ReLU(inplace=True),
                nn.Dropout(config['model']['dropout_rate'] / 2),
                nn.Linear(config['model']['feature_dim'], config['model']['num_classes'])
            )
        
        else:
            raise ValueError(f"Unsupported backbone: {config['model']['backbone']}. "
                           f"Supported backbones: resnet18, resnet50, efficientnet_b0, simple_cnn")
    
    def forward(self, x):
        if self.backbone_name == 'simple_cnn':
            features = self.backbone(x)
            return self.classifier(features)
        else:
            return self.backbone(x)
    
    def get_features(self, x):
        """Extract features from intermediate layers."""
        if self.backbone_name == 'resnet18' or self.backbone_name == 'resnet50':
            # For ResNet models
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
            
            return x
        
        elif self.backbone_name == 'efficientnet_b0':
            # For EfficientNet
            features = self.backbone.features(x)
            features = self.backbone.avgpool(features)
            features = torch.flatten(features, 1)
            return features
        
        elif self.backbone_name == 'simple_cnn':
            # For simple CNN
            return self.backbone(x)
        
        else:
            return None

class MultiTaskMentalHealthModel(nn.Module):
    """Multi-task model for predicting multiple mental health aspects."""
    
    def __init__(self, config):
        super(MultiTaskMentalHealthModel, self).__init__()
        self.config = config
        
        # Shared backbone
        backbone_name = config['model']['backbone']
        
        if backbone_name == 'resnet18':
            if config['model']['pretrained']:
                backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            else:
                backbone = models.resnet18(weights=None)
            in_features = backbone.fc.in_features
            self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # Remove avgpool and fc
        
        elif backbone_name == 'resnet50':
            if config['model']['pretrained']:
                backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            else:
                backbone = models.resnet50(weights=None)
            in_features = backbone.fc.in_features
            self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        
        elif backbone_name == 'efficientnet_b0':
            if config['model']['pretrained']:
                backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            else:
                backbone = models.efficientnet_b0(weights=None)
            in_features = backbone.classifier[1].in_features
            self.backbone = backbone.features
        
        else:
            raise ValueError(f"Unsupported backbone for multi-task: {backbone_name}")
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Task-specific heads
        self.primary_classifier = nn.Sequential(
            nn.Dropout(config['model']['dropout_rate']),
            nn.Linear(in_features, config['model']['feature_dim']),
            nn.ReLU(inplace=True),
            nn.Linear(config['model']['feature_dim'], config['model']['num_classes'])
        )
        
        # Additional heads for severity, confidence, etc.
        self.severity_head = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Severity score 0-1
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Confidence score 0-1
        )
        
        self.anxiety_head = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Anxiety level 0-1
        )
        
        self.depression_head = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Depression level 0-1
        )
    
    def forward(self, x):
        features = self.backbone(x)
        features = self.global_pool(features)
        features = features.view(features.size(0), -1)
        
        primary_output = self.primary_classifier(features)
        severity_output = self.severity_head(features)
        confidence_output = self.confidence_head(features)
        anxiety_output = self.anxiety_head(features)
        depression_output = self.depression_head(features)
        
        return {
            'primary': primary_output,
            'severity': severity_output,
            'confidence': confidence_output,
            'anxiety_level': anxiety_output,
            'depression_level': depression_output
        }

class HandwritingSpecificModel(nn.Module):
    """Model specifically designed for handwriting analysis."""
    
    def __init__(self, config):
        super(HandwritingSpecificModel, self).__init__()
        self.config = config
        
        # Use grayscale for handwriting
        in_channels = 1 if config.get('handwriting', {}).get('grayscale', True) else 3
        
        # Feature extractor for handwriting patterns
        self.feature_extractor = nn.Sequential(
            # First block
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Second block - focus on stroke patterns
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Third block - focus on letter shapes
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Fourth block - focus on spatial arrangement
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            
            nn.Flatten()
        )
        
        # Calculate feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, 224, 224)
            feature_dim = self.feature_extractor(dummy_input).shape[1]
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, config['model']['num_classes'])
        )
        
        # Handwriting-specific feature heads
        self.pressure_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.tremor_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        classification = self.classifier(features)
        pressure_level = self.pressure_head(features)
        tremor_level = self.tremor_head(features)
        
        return {
            'classification': classification,
            'pressure_level': pressure_level,
            'tremor_level': tremor_level,
            'features': features
        }

class DrawingSpecificModel(nn.Module):
    """Model specifically designed for drawing analysis."""
    
    def __init__(self, config):
        super(DrawingSpecificModel, self).__init__()
        self.config = config
        
        # Feature extractor for drawing patterns
        self.feature_extractor = nn.Sequential(
            # First block - color and basic shapes
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Second block - complex shapes
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Third block - composition
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Fourth block - fine details
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            
            nn.Flatten()
        )
        
        # Calculate feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            feature_dim = self.feature_extractor(dummy_input).shape[1]
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, config['model']['num_classes'])
        )
        
        # Drawing-specific feature heads
        self.complexity_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.color_variety_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.symmetry_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        classification = self.classifier(features)
        complexity = self.complexity_head(features)
        color_variety = self.color_variety_head(features)
        symmetry = self.symmetry_head(features)
        
        return {
            'classification': classification,
            'complexity': complexity,
            'color_variety': color_variety,
            'symmetry': symmetry,
            'features': features
        }

def build_model(config):
    """Build model based on configuration."""
    model_type = config['model'].get('type', 'standard')
    
    if model_type == 'standard':
        model = MentalHealthClassifier(config)
    elif model_type == 'multi_task':
        model = MultiTaskMentalHealthModel(config)
    elif model_type == 'handwriting':
        model = HandwritingSpecificModel(config)
    elif model_type == 'drawing':
        model = DrawingSpecificModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print(f"✅ Model built: {model_type} with backbone {config['model']['backbone']}")
    print(f"🔢 Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model

def test_model():
    """Test function to verify model builds correctly."""
    test_config = {
        'model': {
            'backbone': 'resnet18',
            'num_classes': 6,
            'pretrained': True,
            'dropout_rate': 0.3,
            'feature_dim': 512,
            'type': 'standard'
        }
    }
    
    print("🧪 Testing model building...")
    
    # Test different backbones
    backbones = ['resnet18', 'resnet50', 'efficientnet_b0', 'simple_cnn']
    
    for backbone in backbones:
        try:
            test_config['model']['backbone'] = backbone
            model = build_model(test_config)
            
            # Test forward pass
            dummy_input = torch.randn(2, 3, 224, 224)
            output = model(dummy_input)
            
            print(f"✅ {backbone}: Forward pass successful, output shape: {output.shape}")
            
            # Test feature extraction
            features = model.get_features(dummy_input)
            if features is not None:
                print(f"   Feature extraction successful, shape: {features.shape}")
            
        except Exception as e:
            print(f"❌ {backbone}: {e}")
    
    # Test specialized models
    print("\n🧪 Testing specialized models...")
    
    try:
        handwriting_config = test_config.copy()
        handwriting_config['model']['type'] = 'handwriting'
        handwriting_model = HandwritingSpecificModel(handwriting_config)
        dummy_input = torch.randn(2, 1, 224, 224)  # Grayscale for handwriting
        output = handwriting_model(dummy_input)
        print(f"✅ Handwriting model: Forward pass successful")
        print(f"   Output keys: {list(output.keys())}")
    except Exception as e:
        print(f"❌ Handwriting model: {e}")
    
    try:
        drawing_config = test_config.copy()
        drawing_config['model']['type'] = 'drawing'
        drawing_model = DrawingSpecificModel(drawing_config)
        dummy_input = torch.randn(2, 3, 224, 224)  # RGB for drawings
        output = drawing_model(dummy_input)
        print(f"✅ Drawing model: Forward pass successful")
        print(f"   Output keys: {list(output.keys())}")
    except Exception as e:
        print(f"❌ Drawing model: {e}")

if __name__ == "__main__":
    test_model()