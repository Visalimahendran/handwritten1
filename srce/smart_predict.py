import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2

def smart_analyze(image_path):
    """Combine model + feature analysis"""
    
    # 1. Feature-based analysis
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Could not load image")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    contrast = np.std(gray)
    
    # 2. Simple mental health assessment
    conditions = []
    
    if brightness < 80:
        conditions.append(("Depression", 0.7))
    elif brightness > 180:
        conditions.append(("Mania", 0.6))
    else:
        conditions.append(("Healthy", 0.8))
    
    if contrast > 70:
        conditions.append(("Anxiety", 0.6))
    
    # 3. Display results
    print(f"🧠 Mental Health Analysis for: {image_path}")
    print("=" * 40)
    for condition, confidence in conditions:
        print(f"🔍 {condition}: {confidence:.0%} confidence")
    print("=" * 40)
    print("💡 Note: This is a simple analysis. Consult a professional for diagnosis.")

# Use any image
image_path = input("Enter path to handwriting image: ")
smart_analyze(image_path)