import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure, color, filters
import torch

class DrawingAnalyzer:
    """Analyze drawing features for mental health assessment."""
    
    def __init__(self, config):
        self.config = config
    
    def analyze_drawing(self, image_path):
        """Complete analysis of a drawing sample."""
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Extract features
        structural_features = self.extract_structural_features(image)
        color_features = self.extract_color_features(image)
        composition_features = self.extract_composition_features(image)
        
        # Combine features
        all_features = {**structural_features, **color_features, **composition_features}
        
        # Mental health indicators
        indicators = self.assess_mental_health_indicators(all_features)
        
        return {
            'features': all_features,
            'indicators': indicators,
            'image': image
        }
    
    def extract_structural_features(self, image):
        """Extract structural features from drawing."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Contour analysis
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        features = {}
        
        if contours:
            # Complexity metrics
            total_area = sum(cv2.contourArea(cnt) for cnt in contours)
            total_perimeter = sum(cv2.arcLength(cnt, True) for cnt in contours)
            
            features['complexity'] = total_perimeter / total_area if total_area > 0 else 0
            features['num_contours'] = len(contours)
            
            # Size distribution
            areas = [cv2.contourArea(cnt) for cnt in contours]
            features['size_variation'] = np.std(areas) / np.mean(areas) if np.mean(areas) > 0 else 0
            
            # Symmetry analysis
            features['symmetry_score'] = self.analyze_symmetry(image)
            
            # Detail level
            features['detail_density'] = np.sum(edges) / (image.shape[0] * image.shape[1])
        
        return features
    
    def analyze_symmetry(self, image):
        """Analyze symmetry of the drawing."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Vertical symmetry
        height, width = gray.shape
        left_half = gray[:, :width//2]
        right_half = gray[:, width//2:]
        
        # Flip right half for comparison
        right_half_flipped = cv2.flip(right_half, 1)
        
        # Resize to same dimensions if needed
        min_height = min(left_half.shape[0], right_half_flipped.shape[0])
        min_width = min(left_half.shape[1], right_half_flipped.shape[1])
        
        left_cropped = left_half[:min_height, :min_width]
        right_cropped = right_half_flipped[:min_height, :min_width]
        
        # Calculate symmetry score
        if left_cropped.size > 0 and right_cropped.size > 0:
            mse = np.mean((left_cropped.astype(float) - right_cropped.astype(float)) ** 2)
            max_pixel = 255.0
            symmetry = 1 - (mse / (max_pixel ** 2))
            return max(0, symmetry)
        
        return 0
    
    def extract_color_features(self, image):
        """Extract color-related features."""
        features = {}
        
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Color diversity
        unique_colors = len(np.unique(image.reshape(-1, image.shape[2]), axis=0))
        features['color_diversity'] = unique_colors / 1000  # Normalize
        
        # Color intensity
        features['average_brightness'] = np.mean(hsv[:, :, 2]) / 255.0
        
        # Color contrast
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        features['contrast'] = np.std(gray) / 255.0
        
        # Dominant colors analysis
        dominant_colors = self.analyze_dominant_colors(image)
        features.update(dominant_colors)
        
        return features
    
    def analyze_dominant_colors(self, image, n_colors=5):
        """Analyze dominant colors in the drawing."""
        # Reshape image to be a list of pixels
        pixels = image.reshape(-1, 3)
        
        # Convert to float32 for k-means
        pixels = np.float32(pixels)
        
        # Define criteria and apply k-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(pixels, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert back to uint8
        centers = np.uint8(centers)
        
        # Count labels for each center
        unique, counts = np.unique(labels, return_counts=True)
        
        # Get the most dominant color
        dominant_idx = np.argmax(counts)
        dominant_color = centers[dominant_idx]
        
        # Convert to HSV for emotional analysis
        dominant_color_rgb = np.uint8([[dominant_color]])
        dominant_color_hsv = cv2.cvtColor(dominant_color_rgb, cv2.COLOR_RGB2HSV)[0][0]
        
        return {
            'dominant_color_hue': dominant_color_hsv[0] / 179.0,  # Normalize to 0-1
            'dominant_color_saturation': dominant_color_hsv[1] / 255.0,
            'dominant_color_value': dominant_color_hsv[2] / 255.0,
            'color_uniformity': np.max(counts) / np.sum(counts)
        }
    
    def extract_composition_features(self, image):
        """Extract composition and layout features."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        height, width = gray.shape
        
        features = {}
        
        # Center of mass
        y, x = np.indices(gray.shape)
        total_intensity = np.sum(gray)
        
        if total_intensity > 0:
            center_x = np.sum(x * gray) / total_intensity
            center_y = np.sum(y * gray) / total_intensity
            
            # Normalize to image dimensions
            features['center_x'] = center_x / width
            features['center_y'] = center_y / height
            
            # Distance from center
            features['center_deviation'] = np.sqrt(
                (features['center_x'] - 0.5) ** 2 + 
                (features['center_y'] - 0.5) ** 2
            )
        else:
            features['center_deviation'] = 0
        
        # Usage of space
        binary = gray > np.mean(gray)  # Simple threshold
        features['space_usage'] = np.sum(binary) / (height * width)
        
        # Balance analysis
        left_half = binary[:, :width//2]
        right_half = binary[:, width//2:]
        top_half = binary[:height//2, :]
        bottom_half = binary[height//2:, :]
        
        features['horizontal_balance'] = abs(np.sum(left_half) - np.sum(right_half)) / (height * width)
        features['vertical_balance'] = abs(np.sum(top_half) - np.sum(bottom_half)) / (height * width)
        
        return features
    
    def assess_mental_health_indicators(self, features):
        """Assess mental health indicators from drawing features."""
        indicators = {}
        
        # Depression indicators (simplified, dark colors, minimal detail)
        indicators['depression_risk'] = (
            (1 - features.get('average_brightness', 0)) * 0.4 +
            (1 - features.get('color_diversity', 0)) * 0.3 +
            (1 - features.get('detail_density', 0)) * 0.3
        )
        
        # Anxiety indicators (chaotic, asymmetrical)
        indicators['anxiety_risk'] = (
            (1 - features.get('symmetry_score', 0)) * 0.4 +
            features.get('complexity', 0) * 0.3 +
            features.get('center_deviation', 0) * 0.3
        )
        
        # Mania indicators (vibrant colors, busy composition)
        indicators['mania_risk'] = (
            features.get('color_diversity', 0) * 0.4 +
            features.get('detail_density', 0) * 0.3 +
            features.get('complexity', 0) * 0.3
        )
        
        return indicators
    
    def visualize_analysis(self, analysis_result, save_path=None):
        """Visualize drawing analysis results."""
        image = analysis_result['image']
        features = analysis_result['features']
        indicators = analysis_result['indicators']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original drawing
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Drawing')
        axes[0, 0].axis('off')
        
        # Color analysis
        color_features = ['color_diversity', 'average_brightness', 'contrast']
        color_values = [features.get(name, 0) for name in color_features]
        color_labels = ['Color Diversity', 'Brightness', 'Contrast']
        
        axes[0, 1].bar(color_labels, color_values, color=['red', 'yellow', 'blue'])
        axes[0, 1].set_title('Color Features')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Structural features
        structural_features = ['complexity', 'symmetry_score', 'detail_density']
        structural_values = [features.get(name, 0) for name in structural_features]
        structural_labels = ['Complexity', 'Symmetry', 'Detail Density']
        
        axes[0, 2].bar(structural_labels, structural_values, color=['green', 'purple', 'orange'])
        axes[0, 2].set_title('Structural Features')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Composition features
        composition_features = ['space_usage', 'center_deviation', 'horizontal_balance']
        composition_values = [features.get(name, 0) for name in composition_features]
        composition_labels = ['Space Usage', 'Center Deviation', 'Horizontal Balance']
        
        axes[1, 0].bar(composition_labels, composition_values, color=['brown', 'pink', 'gray'])
        axes[1, 0].set_title('Composition Features')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Mental health indicators
        indicator_names = list(indicators.keys())
        indicator_values = [indicators[name] for name in indicator_names]
        
        colors = ['blue', 'red', 'orange']
        axes[1, 1].bar(indicator_names, indicator_values, color=colors)
        axes[1, 1].set_title('Mental Health Indicators')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Text summary
        axes[1, 2].axis('off')
        summary_text = f"""
        Drawing Analysis Summary:
        
        Color Analysis:
        - Color Diversity: {features.get('color_diversity', 0):.3f}
        - Average Brightness: {features.get('average_brightness', 0):.3f}
        - Contrast: {features.get('contrast', 0):.3f}
        
        Structural Analysis:
        - Complexity: {features.get('complexity', 0):.3f}
        - Symmetry: {features.get('symmetry_score', 0):.3f}
        - Detail Density: {features.get('detail_density', 0):.3f}
        
        Mental Health Indicators:
        - Depression Risk: {indicators.get('depression_risk', 0):.3f}
        - Anxiety Risk: {indicators.get('anxiety_risk', 0):.3f}
        - Mania Risk: {indicators.get('mania_risk', 0):.3f}
        """
        axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes, 
                       verticalalignment='top', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()