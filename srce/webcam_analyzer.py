import cv2
import numpy as np
import os

def analyze_image(image):
    """Analyze handwriting from image"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    brightness = np.mean(gray)
    contrast = np.std(gray)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # Mental health analysis
    conditions = []
    
    if brightness < 80:
        conditions.append(("Depression", 0.7, "Dark writing"))
    elif brightness > 180:
        conditions.append(("Mania", 0.6, "Very bright"))
    else:
        conditions.append(("Healthy", 0.8, "Normal brightness"))
    
    if contrast > 70:
        conditions.append(("Anxiety", 0.6, "High variation"))
    
    if edge_density > 0.1:
        conditions.append(("Stress", 0.5, "Shaky writing"))
    
    if contrast < 30:
        conditions.append(("OCD", 0.4, "Very uniform"))
    
    return conditions, brightness, contrast, edge_density

def print_results(conditions, bright, contr, edges, source):
    print(f"\n🎯 ANALYSIS: {source}")
    print("="*50)
    print(f"📊 Brightness: {bright:.1f}")
    print(f"📊 Contrast: {contr:.1f}") 
    print(f"📊 Tremor Level: {edges:.3f}")
    print("-"*50)
    
    for condition, confidence, reason in conditions:
        print(f"🔍 {condition}: {confidence:.0%} - {reason}")
    
    print("="*50)

def webcam_mode():
    """Analyze via webcam"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot access webcam")
        return
    
    print("📷 WEBCAM MODE - Show handwriting to camera")
    print("🛑 Press 's' to analyze, 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        cv2.putText(frame, "Show handwriting - Press 's' to analyze", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow('Handwriting Analysis - Press s to analyze', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            conditions, bright, contr, edges = analyze_image(frame)
            print_results(conditions, bright, contr, edges, "WEBCAM")
            cv2.imwrite("webcam_capture.jpg", frame)
            print("💾 Saved as 'webcam_capture.jpg'")
            
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def upload_mode():
    """Analyze uploaded image"""
    path = input("📁 Enter image path: ").strip('"')
    
    if not os.path.exists(path):
        print("❌ File not found")
        return
    
    image = cv2.imread(path)
    if image is None:
        print("❌ Cannot load image")
        return
    
    conditions, bright, contr, edges = analyze_image(image)
    print_results(conditions, bright, contr, edges, f"UPLOAD: {os.path.basename(path)}")
    
    # Show the image
    cv2.imshow("Uploaded Image", image)
    print("👀 Image displayed - Press any key to close")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    print("🧠 HANDWRITING MENTAL HEALTH ANALYZER")
    print("="*40)
    print("1. 📷 Webcam Analysis")
    print("2. 📁 Upload Image")
    print("3. ❌ Exit")
    
    choice = input("\nChoose option (1/2/3): ")
    
    if choice == "1":
        webcam_mode()
    elif choice == "2":
        upload_mode()
    elif choice == "3":
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()