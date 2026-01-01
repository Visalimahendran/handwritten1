import torch
import torch.nn.functional as F
from PIL import Image
import sys
import os

sys.path.insert(0, os.getcwd())

from srce.model import build_model
from srce.helpers import pil_transform, load_config

config = load_config('configs/default.yaml')
model = build_model(config)
model.load_state_dict(torch.load('models/model_best.pth', map_location='cpu'))
model.eval()

image_path = "data/images/Healthy/drawing_001.png"
image = Image.open(image_path).convert('RGB')
tensor = pil_transform(image, config).unsqueeze(0)

with torch.no_grad():
    output = model(tensor)
    probabilities = F.softmax(output, dim=1)
    confidence, predicted_class = torch.max(probabilities, 1)

class_names = list(config['classes'].values())
print(f"🎯 Prediction: {class_names[predicted_class.item()]}")
print(f"📊 Confidence: {confidence.item():.2%}")