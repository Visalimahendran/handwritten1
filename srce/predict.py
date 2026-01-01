import torch
import torch.nn.functional as F
from PIL import Image
import sys
import os

# Fix imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from srce.model import build_model
from srce.helpers import pil_transform, load_config

def predict_image(image_path):
    config = load_config('configs/default.yaml')
    model = build_model(config)
    model.load_state_dict(torch.load('models/model_best.pth', map_location='cpu'))
    model.eval()

    image = Image.open(image_path).convert('RGB')
    tensor = pil_transform(image, config).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    class_names = list(config['classes'].values())
    
    print(f"🖼️  Image: {image_path}")
    print(f"🎯 Prediction: {class_names[predicted_class.item()]} (Class {predicted_class.item()})")
    print(f"📊 Confidence: {confidence.item():.2%}")
    
    # Show top 3 predictions
    top_probs, top_indices = torch.topk(probabilities, 3)
    print("\n🏆 Top Predictions:")
    for i in range(3):
        print(f"   {i+1}. {class_names[top_indices[0][i].item()]:<15} {top_probs[0][i].item():.2%}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        predict_image(sys.argv[1])
    else:
        print("Usage: python -m srce.predict <image_path>")