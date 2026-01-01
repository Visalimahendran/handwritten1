import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure, morphology
import pandas as pd

class HandwritingAnalyzer:
    """Analyze handwriting features for mental health assessment."""
    
    def __init__(self, config):
        self.config = config
        self.features = {}
    
    def preprocess_handwriting(self, image):
        """Preprocess handwriting image for analysis."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply noise reduction
        denoised = cv2.medianBlur(gray, 3)
        
        # Binarization
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Skeletonization
        skeleton = morphology.skeletonize(binary > 0)
        skeleton = skeleton.astype(np.uint8) * 255
        
        return {
            'original': image,
            'grayscale': gray,
            'binary': binary,
            'skeleton': skeleton
        }
    
    def extract_graphomotor_features(self, processed):
        """Extract graphomotor features from handwriting."""
        binary = processed['binary']
        skeleton = processed['skeleton']
        
        features = {}
        
        # 1. Stroke width variation (pressure indicator)
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        features['stroke_width_mean'] = np.mean(dist_transform[dist_transform > 0])
        features['stroke_width_std'] = np.std(dist_transform[dist_transform > 0])
        
        # 2. Tremor analysis
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(largest_contour, True)
            area = cv2.contourArea(largest_contour)
            
            # Smoothness metric (tremor indicator)
            features['smoothness'] = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        
        # 3. Writing speed estimation (from skeleton complexity)
        branch_points = self.detect_branch_points(skeleton)
        features['branch_points_count'] = len(branch_points)
        
        return features
    
    def detect_branch_points(self, skeleton):
        """Detect branch points in skeletonized writing."""
        kernel = np.array([[1, 1, 1],
                          [1, 10, 1],
                          [1, 1, 1]], dtype=np.uint8)
        
        filtered = cv2.filter2D(skeleton, -1, kernel)
        branch_points = np.where(filtered >= 13)
        
        return list(zip(branch_points[1], branch_points[0]))
    
    def extract_spatial_features(self, processed):
        """Extract spatial arrangement features."""
        binary = processed['binary']
        height, width = binary.shape
        
        features = {}
        
        # 1. Margins analysis
        horizontal_projection = np.sum(binary, axis=0)
        vertical_projection = np.sum(binary, axis=1)
        
        # Left/right margins
        left_margin = np.argmax(horizontal_projection > 0) if np.any(horizontal_projection > 0) else 0
        right_margin = width - np.argmax(horizontal_projection[::-1] > 0) if np.any(horizontal_projection > 0) else width
        
        features['left_margin'] = left_margin / width
        features['right_margin'] = (width - right_margin) / width
        features['margin_balance'] = abs(features['left_margin'] - features['right_margin'])
        
        # 2. Slant analysis
        moments = cv2.moments(binary)
        if moments['mu02'] != 0:
            slant = moments['mu11'] / moments['mu02']
            features['slant_angle'] = np.arctan(slant) * (180 / np.pi)
        else:
            features['slant_angle'] = 0
        
        # 3. Baseline analysis
        row_sums = np.sum(binary, axis=1)
        if np.any(row_sums > 0):
            writing_region = np.where(row_sums > 0)[0]
            features['baseline_stability'] = np.std(writing_region) / height
        else:
            features['baseline_stability'] = 0
        
        return features
    
    def extract_temporal_features(self, processed):
        """Estimate temporal features from static image."""
        binary = processed['binary']
        skeleton = processed['skeleton']
        
        features = {}
        
        # 1. Writing continuity (pen lifts)
        labeled_image, num_labels = measure.label(binary > 0, return_num=True)
        features['stroke_fragmentation'] = num_labels / (binary.shape[0] * binary.shape[1]) * 1000
        
        # 2. Complexity metric (related to writing speed)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            total_curvature = 0
            for contour in contours:
                if len(contour) > 2:
                    total_curvature += self.calculate_curvature(contour)
            features['writing_complexity'] = total_curvature / len(contours)
        else:
            features['writing_complexity'] = 0
        
        return features
    
    def calculate_curvature(self, contour):
        """Calculate curvature of a contour."""
        if len(contour) < 3:
            return 0
        
        curvature = 0
        for i in range(1, len(contour)-1):
            p1 = contour[i-1][0]
            p2 = contour[i][0]
            p3 = contour[i+1][0]
            
            # Calculate angle between vectors
            v1 = p1 - p2
            v2 = p3 - p2
            
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            
            if norm_v1 > 0 and norm_v2 > 0:
                cosine_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
                cosine_angle = np.clip(cosine_angle, -1, 1)
                angle = np.arccos(cosine_angle)
                curvature += angle
        
        return curvature / (len(contour) - 2) if len(contour) > 2 else 0
    
    def analyze_handwriting_sample(self, image_path):
        """Complete analysis of a handwriting sample."""
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        processed = self.preprocess_handwriting(image)
        
        # Extract features
        graphomotor_features = self.extract_graphomotor_features(processed)
        spatial_features = self.extract_spatial_features(processed)
        temporal_features = self.extract_temporal_features(processed)
        
        # Combine all features
        all_features = {**graphomotor_features, **spatial_features, **temporal_features}
        
        # Mental health indicators
        indicators = self.assess_mental_health_indicators(all_features)
        
        return {
            'features': all_features,
            'indicators': indicators,
            'processed_images': processed
        }
    
    def assess_mental_health_indicators(self, features):
        """Assess potential mental health indicators from handwriting features."""
        indicators = {}
        
        # Depression indicators
        indicators['depression_risk'] = (
            features.get('stroke_width_std', 0) * 0.3 +  # Pressure variation
            features.get('smoothness', 0) * 0.4 +        # Tremor
            features.get('writing_complexity', 0) * 0.3   # Reduced complexity
        )
        
        # Anxiety indicators
        indicators['anxiety_risk'] = (
            features.get('branch_points_count', 0) * 0.4 +  # Restlessness in writing
            features.get('baseline_stability', 0) * 0.3 +   # Unstable baseline
            features.get('margin_balance', 0) * 0.3         # Irregular margins
        )
        
        # Mania/hypomania indicators (Bipolar)
        indicators['mania_risk'] = (
            features.get('stroke_width_mean', 0) * 0.4 +   # Heavy pressure
            features.get('slant_angle', 0) * 0.3 +         # Rightward slant
            features.get('writing_complexity', 0) * 0.3     # Increased complexity
        )
        
        return indicators
    
    def visualize_analysis(self, analysis_result, save_path=None):
        """Visualize handwriting analysis results."""
        processed = analysis_result['processed_images']
        features = analysis_result['features']
        indicators = analysis_result['indicators']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(processed['original'])
        axes[0, 0].set_title('Original Handwriting')
        axes[0, 0].axis('off')
        
        # Binary image
        axes[0, 1].imshow(processed['binary'], cmap='gray')
        axes[0, 1].set_title('Binary Image')
        axes[0, 1].axis('off')
        
        # Skeleton
        axes[0, 2].imshow(processed['skeleton'], cmap='gray')
        axes[0, 2].set_title('Skeletonized')
        axes[0, 2].axis('off')
        
        # Feature visualization
        feature_names = ['stroke_width_std', 'smoothness', 'slant_angle']
        feature_values = [features.get(name, 0) for name in feature_names]
        feature_labels = ['Stroke Variation', 'Smoothness', 'Slant Angle']
        
        axes[1, 0].bar(feature_labels, feature_values)
        axes[1, 0].set_title('Key Handwriting Features')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Mental health indicators
        indicator_names = list(indicators.keys())
        indicator_values = [indicators[name] for name in indicator_names]
        
        axes[1, 1].bar(indicator_names, indicator_values)
        axes[1, 1].set_title('Mental Health Indicators')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Text summary
        axes[1, 2].axis('off')
        summary_text = f"""
        Handwriting Analysis Summary:
        
        Graphomotor Features:
        - Stroke Variation: {features.get('stroke_width_std', 0):.3f}
        - Smoothness: {features.get('smoothness', 0):.3f}
        - Branch Points: {features.get('branch_points_count', 0)}
        
        Spatial Features:
        - Slant Angle: {features.get('slant_angle', 0):.1f}°
        - Margin Balance: {features.get('margin_balance', 0):.3f}
        - Baseline Stability: {features.get('baseline_stability', 0):.3f}
        
        Dominant Indicators:
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
        