import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
import os

class PlantDiseaseDetector:
    def __init__(self):
        self.image = None
        self.processed_image = None
        self.leaf_mask = None
        self.features = {}
        
    def load_image(self, image_path):
        """Load an image from the specified path."""
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise FileNotFoundError(f"Could not load image from {image_path}")
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.processed_image = self.image.copy()
        print(f"Image loaded: {image_path}, Shape: {self.image.shape}")
        return self.image
    
    def display_image(self, img=None, title="Image"):
        """Display the image using matplotlib."""
        if img is None:
            img = self.processed_image
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
        plt.show()
        
    # 1. POINT PROCESSING TECHNIQUES
    
    def adjust_brightness(self, beta=30):
        """Adjust brightness by adding a constant value."""
        self.processed_image = np.clip(self.image + beta, 0, 255).astype(np.uint8)
        print(f"Brightness adjusted with value: {beta}")
        return self.processed_image
    
    def adjust_contrast(self, alpha=1.5):
        """Adjust contrast by multiplying a constant value."""
        self.processed_image = np.clip(alpha * self.image, 0, 255).astype(np.uint8)
        print(f"Contrast adjusted with factor: {alpha}")
        return self.processed_image
    
    def gamma_correction(self, gamma=0.5):
        """Apply gamma correction to the image."""
        gamma_inv = 1.0 / gamma
        table = np.array([((i / 255.0) ** gamma_inv) * 255 for i in range(256)]).astype(np.uint8)
        self.processed_image = cv2.LUT(self.image, table)
        print(f"Gamma correction applied with gamma: {gamma}")
        return self.processed_image
    
    def threshold(self, thresh=127, max_val=255):
        """Apply binary thresholding to the image."""
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, thresh, max_val, cv2.THRESH_BINARY)
        self.processed_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        print(f"Thresholding applied with threshold: {thresh}")
        return self.processed_image
    
    # 2. MASK PROCESSING TECHNIQUES
    
    def create_leaf_mask(self):
        """Create a mask to isolate the leaf from the background."""
        # Convert to HSV color space
        hsv = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV)
        
        # Define range of green color in HSV
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([85, 255, 255])
        
        # Threshold the HSV image to get only green colors
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        self.leaf_mask = mask
        print("Leaf mask created")
        return mask
    
    def apply_mask(self, mask=None):
        """Apply the mask to isolate the leaf in the image."""
        if mask is None:
            if self.leaf_mask is None:
                self.create_leaf_mask()
            mask = self.leaf_mask
            
        # Apply the mask to the original image
        masked_img = cv2.bitwise_and(self.image, self.image, mask=mask)
        self.processed_image = masked_img
        print("Mask applied to isolate leaf")
        return masked_img
    
    def detect_disease_spots(self):
        """Detect potential disease spots on the leaf."""
        if self.leaf_mask is None:
            self.create_leaf_mask()
            
        # Convert to HSV color space
        hsv = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV)
        
        # Define range for brown/yellow spots (common disease symptoms)
        lower_brown = np.array([10, 100, 20])
        upper_brown = np.array([30, 255, 255])
        
        # Threshold the HSV image to get brown/yellow spots
        disease_mask = cv2.inRange(hsv, lower_brown, upper_brown)
        
        # Apply leaf mask to only consider spots on the leaf
        disease_mask = cv2.bitwise_and(disease_mask, self.leaf_mask)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_OPEN, kernel)
        
        # Create a visual representation of the disease spots
        disease_highlight = self.image.copy()
        disease_highlight[disease_mask > 0] = [255, 0, 0]  # Mark disease spots in red
        
        self.processed_image = disease_highlight
        self.features['disease_ratio'] = np.sum(disease_mask) / np.sum(self.leaf_mask) if np.sum(self.leaf_mask) > 0 else 0
        print(f"Disease spots detected. Disease ratio: {self.features['disease_ratio']:.4f}")
        return disease_highlight
    
    def create_disease_heatmap(self):
        """Create a heatmap visualization of disease severity."""
        if self.leaf_mask is None:
            self.create_leaf_mask()
            
        # Convert to HSV color space
        hsv = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV)
        
        # Define range for brown/yellow spots (common disease symptoms)
        lower_brown = np.array([10, 100, 20])
        upper_brown = np.array([30, 255, 255])
        
        # Threshold the HSV image to get brown/yellow spots
        disease_mask = cv2.inRange(hsv, lower_brown, upper_brown)
        
        # Apply leaf mask to only consider spots on the leaf
        disease_mask = cv2.bitwise_and(disease_mask, self.leaf_mask)
        
        # Apply a distance transform to create a gradient effect
        dist_transform = cv2.distanceTransform(disease_mask, cv2.DIST_L2, 5)
        
        # Normalize the distance transform to 0-255 range
        cv2.normalize(dist_transform, dist_transform, 0, 255, cv2.NORM_MINMAX)
        
        # Create a heatmap visualization
        heatmap = cv2.applyColorMap(dist_transform.astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Create a blended visualization
        alpha = 0.7
        leaf_only = self.apply_mask(self.leaf_mask)
        blended = cv2.addWeighted(leaf_only, 1-alpha, heatmap, alpha, 0)
        
        self.processed_image = blended
        print("Disease heatmap created")
        return blended
    
    # 3. HISTOGRAM PROCESSING
    
    def plot_histogram(self):
        """Plot the histogram of the image."""
        colors = ('r', 'g', 'b')
        plt.figure(figsize=(12, 6))
        
        for i, color in enumerate(colors):
            hist = cv2.calcHist([self.image], [i], None, [256], [0, 256])
            plt.plot(hist, color=color)
            plt.xlim([0, 256])
            
        plt.title('Image Histogram')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.show()
        print("Histogram plotted")
    
    def equalize_histogram(self):
        """Apply histogram equalization to enhance contrast."""
        image_yuv = cv2.cvtColor(self.image, cv2.COLOR_RGB2YUV)
        
        # Equalize the Y channel
        image_yuv[:,:,0] = cv2.equalizeHist(image_yuv[:,:,0])
        
        # Convert back to RGB
        self.processed_image = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)
        print("Histogram equalization applied")
        return self.processed_image
    
    # 4. EDGE DETECTION
    
    def detect_edges_sobel(self):
        """Detect edges using Sobel operator."""
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        
        # Apply Sobel operator
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate the gradient magnitude
        sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel_mag = np.uint8(np.clip(sobel_mag, 0, 255))
        
        self.processed_image = cv2.cvtColor(sobel_mag, cv2.COLOR_GRAY2RGB)
        print("Sobel edge detection applied")
        return self.processed_image
    
    def detect_edges_prewitt(self):
        """Detect edges using Prewitt operator."""
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        
        # Define Prewitt kernels
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        
        # Apply Prewitt operator
        prewitt_x = cv2.filter2D(gray, -1, kernel_x)
        prewitt_y = cv2.filter2D(gray, -1, kernel_y)
        
        # Calculate the gradient magnitude
        prewitt_mag = np.sqrt(prewitt_x**2 + prewitt_y**2)
        prewitt_mag = np.uint8(np.clip(prewitt_mag, 0, 255))
        
        self.processed_image = cv2.cvtColor(prewitt_mag, cv2.COLOR_GRAY2RGB)
        print("Prewitt edge detection applied")
        return self.processed_image
    
    def detect_edges_roberts(self):
        """Detect edges using Roberts operator."""
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        
        # Define Roberts kernels
        kernel_x = np.array([[1, 0], [0, -1]])
        kernel_y = np.array([[0, 1], [-1, 0]])
        
        # Apply Roberts operator
        roberts_x = cv2.filter2D(gray, -1, kernel_x)
        roberts_y = cv2.filter2D(gray, -1, kernel_y)
        
        # Calculate the gradient magnitude
        roberts_mag = np.sqrt(roberts_x**2 + roberts_y**2)
        roberts_mag = np.uint8(np.clip(roberts_mag, 0, 255))
        
        self.processed_image = cv2.cvtColor(roberts_mag, cv2.COLOR_GRAY2RGB)
        print("Roberts edge detection applied")
        return self.processed_image
    
    # 5. IMAGE COMPRESSION USING DCT
    
    def compress_dct(self, quality=10):
        """Compress the image using Discrete Cosine Transform."""
        # Convert to YCrCb color space
        ycrcb = cv2.cvtColor(self.image, cv2.COLOR_RGB2YCrCb)
        
        # Process each channel separately
        compressed = np.zeros_like(ycrcb, dtype=np.float32)
        
        for i in range(3):
            channel = ycrcb[:,:,i].astype(np.float32)
            
            # Apply DCT on 8x8 blocks
            h, w = channel.shape
            h_blocks = h // 8
            w_blocks = w // 8
            
            for y in range(h_blocks):
                for x in range(w_blocks):
                    # Extract the 8x8 block
                    block = channel[y*8:(y+1)*8, x*8:(x+1)*8]
                    
                    # Apply DCT
                    dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                    
                    # Quantization (zeroing out high-frequency components)
                    # Higher quality = fewer zeroed components
                    mask = np.zeros((8, 8))
                    mask[:quality, :quality] = 1
                    dct_block = dct_block * mask
                    
                    # Apply inverse DCT
                    block = idct(idct(dct_block.T, norm='ortho').T, norm='ortho')
                    compressed[y*8:(y+1)*8, x*8:(x+1)*8, i] = block
        
        # Convert back to RGB
        compressed = np.clip(compressed, 0, 255).astype(np.uint8)
        self.processed_image = cv2.cvtColor(compressed, cv2.COLOR_YCrCb2RGB)
        print(f"DCT compression applied with quality factor: {quality}")
        return self.processed_image
    
    # 6. FEATURE EXTRACTION
    
    def extract_features(self):
        """Extract features for disease classification."""
        if self.leaf_mask is None:
            self.create_leaf_mask()
            
        # Color features
        hsv = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV)
        masked_hsv = cv2.bitwise_and(hsv, hsv, mask=self.leaf_mask)
        
        # Calculate mean and std of hue and saturation in the leaf area
        nonzero_mask = self.leaf_mask > 0
        if np.sum(nonzero_mask) > 0:
            self.features['mean_hue'] = np.mean(masked_hsv[:,:,0][nonzero_mask])
            self.features['mean_saturation'] = np.mean(masked_hsv[:,:,1][nonzero_mask])
            self.features['std_hue'] = np.std(masked_hsv[:,:,0][nonzero_mask])
            self.features['std_saturation'] = np.std(masked_hsv[:,:,1][nonzero_mask])
        
        # Texture features (using Sobel edges)
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
        
        masked_sobel = cv2.bitwise_and(sobel_mag.astype(np.uint8), sobel_mag.astype(np.uint8), mask=self.leaf_mask)
        if np.sum(nonzero_mask) > 0:
            self.features['mean_edge_intensity'] = np.mean(masked_sobel[nonzero_mask])
            self.features['std_edge_intensity'] = np.std(masked_sobel[nonzero_mask])
        
        # Disease spot detection
        hsv = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV)
        lower_brown = np.array([10, 100, 20])
        upper_brown = np.array([30, 255, 255])
        disease_mask = cv2.inRange(hsv, lower_brown, upper_brown)
        disease_mask = cv2.bitwise_and(disease_mask, self.leaf_mask)
        
        self.features['disease_ratio'] = np.sum(disease_mask) / np.sum(self.leaf_mask) if np.sum(self.leaf_mask) > 0 else 0
        
        # Calculate moment features of the disease spots
        M = cv2.moments(disease_mask)
        if M["m00"] != 0:
            self.features['centroid_x'] = M["m10"] / M["m00"]
            self.features['centroid_y'] = M["m01"] / M["m00"]
            self.features['disease_spread'] = np.sqrt(M["mu20"] + M["mu02"]) / M["m00"]
        
        print("Features extracted:", self.features)
        return self.features
    
    # 7. DISEASE CLASSIFICATION
    
    def classify_disease(self):
        """Classify the disease based on extracted features."""
        if not self.features:
            self.extract_features()
        
        # Simple rule-based classification
        disease_ratio = self.features.get('disease_ratio', 0)
        mean_hue = self.features.get('mean_hue', 0)
        
        # Define classification rules
        if disease_ratio < 0.05:
            disease = "Healthy"
            confidence = 0.95 - disease_ratio * 10
        elif disease_ratio < 0.15:
            disease = "Early Blight"
            confidence = min(0.7, disease_ratio * 4)
        elif disease_ratio < 0.30:
            disease = "Late Blight"
            confidence = min(0.85, disease_ratio * 2)
        else:
            disease = "Severe Infection"
            confidence = min(0.95, disease_ratio)
        
        # Adjust based on color features
        if 15 <= mean_hue <= 25 and disease_ratio > 0.1:
            disease = "Yellow Leaf Curl Virus"
            confidence = min(0.9, disease_ratio * 2)
        
        result = {
            "disease": disease,
            "confidence": confidence,
            "disease_ratio": disease_ratio,
            "severity": "Low" if disease_ratio < 0.1 else "Medium" if disease_ratio < 0.3 else "High"
        }
        
        print(f"Disease classification: {result['disease']} (Confidence: {result['confidence']:.2f}, Severity: {result['severity']})")
        return result
    
    # 8. VIDEO PROCESSING
    
    def process_video(self, video_path, output_path, frame_processor):
        """
        Process a video file by applying image processing to each frame.
        
        Parameters:
        - video_path: Path to the input video file
        - output_path: Path to save the processed video
        - frame_processor: Function to apply to each frame
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process each frame
        frame_number = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB for processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            self.image = frame_rgb
            self.processed_image = frame_rgb.copy()
            processed_frame = frame_processor(self)
            
            # Convert back to BGR for saving
            processed_frame_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
            out.write(processed_frame_bgr)
            
            frame_number += 1
            if frame_number % 10 == 0:
                print(f"Processed {frame_number}/{frame_count} frames ({frame_number/frame_count*100:.1f}%)")
        
        # Release resources
        cap.release()
        out.release()
        print(f"Video processing complete. Output saved to: {output_path}")


# Demo Usage Function
def run_demo():
    """Run a demonstration of the plant disease detection system."""
    # Create instance of the detector
    detector = PlantDiseaseDetector()
    
    # Sample plant leaf images with diseases (paths to be replaced with actual image paths)
    sample_image_path = "path_to_sample_leaf_image.jpg"
    
    # Replace with an actual image path or use sample data
    # For demonstration purposes, let's create a simple synthetic image
    synthetic_image_path = "synthetic_leaf.jpg"
    
    # Create a synthetic leaf image for demonstration
    height, width = 400, 600
    synthetic_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Green background for leaf
    synthetic_image[:,:] = [100, 180, 100]  # Dark green
    
    # Add some brown spots to simulate disease
    for _ in range(50):
        x = np.random.randint(width)
        y = np.random.randint(height)
        cv2.circle(synthetic_image, (x, y), np.random.randint(5, 15), [50, 70, 30], -1)
    
    # Add leaf shape
    leaf_points = np.array([
        [width//2, 20], 
        [width-100, height//2],
        [width//2, height-20],
        [100, height//2]
    ])
    cv2.fillPoly(synthetic_image, [leaf_points], [120, 200, 80])
    
    # Add some disease spots
    for _ in range(30):
        x = np.random.randint(width//4, 3*width//4)
        y = np.random.randint(height//4, 3*height//4)
        cv2.circle(synthetic_image, (x, y), np.random.randint(3, 10), [60, 100, 50], -1)
    
    cv2.imwrite(synthetic_image_path, cv2.cvtColor(synthetic_image, cv2.COLOR_RGB2BGR))
    print(f"Created synthetic image for demonstration: {synthetic_image_path}")
    
    # 1. Load and display the image
    image = detector.load_image(synthetic_image_path)
    detector.display_image(title="Original Image")
    
    # 2. Apply point processing techniques
    brightened = detector.adjust_brightness(beta=20)
    detector.display_image(title="Brightness Adjusted Image")
    
    contrasted = detector.adjust_contrast(alpha=1.3)
    detector.display_image(title="Contrast Adjusted Image")
    
    gamma_corrected = detector.gamma_correction(gamma=0.7)
    detector.display_image(title="Gamma Corrected Image")
    
    # 3. Apply mask processing techniques
    leaf_mask = detector.create_leaf_mask()
    detector.display_image(cv2.cvtColor(leaf_mask, cv2.COLOR_GRAY2RGB), title="Leaf Mask")
    
    masked_leaf = detector.apply_mask()
    detector.display_image(title="Masked Leaf")
    
    disease_spots = detector.detect_disease_spots()
    detector.display_image(title="Disease Spots Highlighted")
    
    disease_heatmap = detector.create_disease_heatmap()
    detector.display_image(title="Disease Heatmap")
    
    # 4. Histogram processing
    detector.plot_histogram()
    
    equalized = detector.equalize_histogram()
    detector.display_image(title="Histogram Equalized Image")
    detector.plot_histogram()
    
    # 5. Edge detection
    sobel_edges = detector.detect_edges_sobel()
    detector.display_image(title="Sobel Edge Detection")
    
    prewitt_edges = detector.detect_edges_prewitt()
    detector.display_image(title="Prewitt Edge Detection")
    
    roberts_edges = detector.detect_edges_roberts()
    detector.display_image(title="Roberts Edge Detection")
    
    # 6. DCT compression
    compressed = detector.compress_dct(quality=5)
    detector.display_image(title="DCT Compressed Image (Quality=5)")
    
    # 7. Feature extraction and classification
    features = detector.extract_features()
    print("Extracted Features:", features)
    
    classification = detector.classify_disease()
    print("Disease Classification:", classification)
    
    print("\nDemonstration complete!")


if __name__ == "__main__":
    run_demo()