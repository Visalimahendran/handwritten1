import cv2
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def analyze_handwriting_features(image_path):
    """Analyze handwriting without AI model"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Basic analysis
    brightness = np.mean(gray)
    contrast = np.std(gray)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    print(f"📊 Image Analysis:")
    print(f"   Brightness: {brightness:.1f}")
    print(f"   Contrast: {contrast:.1f}")
    print(f"   Edge Density: {edge_density:.3f}")
    
    # Simple rule-based assessment
    if brightness < 100:
        print("🎯 Possible Depression: Dark writing")
    if contrast > 60:
        print("🎯 Possible Anxiety: High contrast/variation")
    if edge_density > 0.1:
        print("🎯 Possible Stress: Tremors in writing")
    if brightness > 150 and contrast < 30:
        print("🎯 Possible Healthy: Balanced writing")
    
    return {
        'brightness': brightness,
        'contrast': contrast, 
        'edge_density': edge_density
    }

# Test with any image
image_path = input("Enter image path: ")
analyze_handwriting_features(image_path)