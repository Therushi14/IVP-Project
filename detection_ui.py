
import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import threading
from PIL import Image, ImageTk

# Import our disease detection system
# Assuming the main code is saved as plant_disease_detector.py
from plant_disease_detector import PlantDiseaseDetector

class DiseaseDetectionUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Plant Disease Detection System")
        self.root.geometry("1200x800")
        
        # Initialize detector
        self.detector = PlantDiseaseDetector()
        self.current_image_path = None
        self.processing_thread = None
        
        # Create UI elements
        self.create_widgets()
        
    def create_widgets(self):
        # Create main frames
        self.control_frame = ttk.Frame(self.root, padding="10")
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        self.image_frame = ttk.Frame(self.root)
        self.image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Control panel elements
        ttk.Label(self.control_frame, text="Plant Disease Detection", font=("Arial", 16)).pack(pady=10)
        
        # Image loading section
        ttk.Label(self.control_frame, text="1. Load Image", font=("Arial", 12, "bold")).pack(pady=(10, 5), anchor="w")
        self.load_button = ttk.Button(self.control_frame, text="Load Image", command=self.load_image)
        self.load_button.pack(pady=5, fill=tk.X)
        
        self.capture_button = ttk.Button(self.control_frame, text="Capture from Camera", command=self.capture_image)
        self.capture_button.pack(pady=5, fill=tk.X)
        
        # Image processing section
        ttk.Separator(self.control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        ttk.Label(self.control_frame, text="2. Image Processing", font=("Arial", 12, "bold")).pack(pady=(10, 5), anchor="w")
        
        # Point processing
        ttk.Label(self.control_frame, text="Point Processing:").pack(anchor="w", pady=(10, 0))
        self.point_processing_var = tk.StringVar(value="original")
        point_options = [
            ("Original", "original"),
            ("Brightness", "brightness"),
            ("Contrast", "contrast"),
            ("Gamma", "gamma")
        ]
        
        for text, value in point_options:
            ttk.Radiobutton(self.control_frame, text=text, value=value, 
                            variable=self.point_processing_var, command=self.apply_processing).pack(anchor="w")
        
        # Mask processing
        ttk.Label(self.control_frame, text="Mask Processing:").pack(anchor="w", pady=(10, 0))
        self.mask_processing_var = tk.StringVar(value="none")
        mask_options = [
            ("None", "none"),
            ("Leaf Mask", "leaf_mask"),
            ("Disease Spots", "disease_spots"),
            ("Disease Heatmap", "heatmap")
        ]
        
        for text, value in mask_options:
            ttk.Radiobutton(self.control_frame, text=text, value=value, 
                            variable=self.mask_processing_var, command=self.apply_processing).pack(anchor="w")
        
        # Edge detection
        ttk.Label(self.control_frame, text="Edge Detection:").pack(anchor="w", pady=(10, 0))
        self.edge_detection_var = tk.StringVar(value="none")
        edge_options = [
            ("None", "none"),
            ("Sobel", "sobel"),
            ("Prewitt", "prewitt"),
            ("Roberts", "roberts")
        ]
        
        for text, value in edge_options:
            ttk.Radiobutton(self.control_frame, text=text, value=value, 
                            variable=self.edge_detection_var, command=self.apply_processing).pack(anchor="w")
        
        # Analysis section
        ttk.Separator(self.control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        ttk.Label(self.control_frame, text="3. Analysis", font=("Arial", 12, "bold")).pack(pady=(10, 5), anchor="w")
        
        self.analyze_button = ttk.Button(self.control_frame, text="Analyze Disease", command=self.analyze_disease)
        self.analyze_button.pack(pady=5, fill=tk.X)
        
        self.histogram_button = ttk.Button(self.control_frame, text="Show Histogram", command=self.show_histogram)
        self.histogram_button.pack(pady=5, fill=tk.X)
        
        self.equalize_button = ttk.Button(self.control_frame, text="Equalize Histogram", command=self.equalize_histogram)
        self.equalize_button.pack(pady=5, fill=tk.X)
        
        self.compress_button = ttk.Button(self.control_frame, text="DCT Compression", command=self.compress_image)
        self.compress_button.pack(pady=5, fill=tk.X)
        
        # Results section
        ttk.Separator(self.control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        ttk.Label(self.control_frame, text="4. Results", font=("Arial", 12, "bold")).pack(pady=(10, 5), anchor="w")
        
        self.results_text = tk.Text(self.control_frame, height=10, width=30, wrap=tk.WORD)
        self.results_text.pack(pady=5, fill=tk.X)
        self.results_text.config(state=tk.DISABLED)
        
        # Save results
        self.save_button = ttk.Button(self.control_frame, text="Save Results", command=self.save_results)
        self.save_button.pack(pady=5, fill=tk.X)
        
        # Image display area
        self.canvas_frame = ttk.Frame(self.image_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Initial image display (placeholder)
        self.fig = plt.Figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Load an image to begin")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def load_image(self):
        """Load image from file"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.status_var.set(f"Loaded: {os.path.basename(file_path)}")
            
            try:
                self.detector.load_image(file_path)
                self.update_image_display(self.detector.image, "Original Image")
                
                # Reset processing options
                self.point_processing_var.set("original")
                self.mask_processing_var.set("none")
                self.edge_detection_var.set("none")
                
                # Clear results
                self.results_text.config(state=tk.NORMAL)
                self.results_text.delete(1.0, tk.END)
                self.results_text.config(state=tk.DISABLED)
                
            except Exception as e:
                self.status_var.set(f"Error: {str(e)}")
    
    def capture_image(self):
        """Capture image from camera"""
        self.status_var.set("Accessing camera...")
        
        # Open camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.status_var.set("Error: Could not open camera")
            return
        
        # Create a simple camera window
        camera_window = tk.Toplevel(self.root)
        camera_window.title("Camera Capture")
        camera_window.geometry("800x600")
        
        # Create canvas for camera feed
        camera_canvas = tk.Canvas(camera_window, width=640, height=480)
        camera_canvas.pack(pady=10)
        
        # Create capture button
        capture_btn = ttk.Button(camera_window, text="Capture", width=20)
        capture_btn.pack(pady=10)
        
        # Variables for storing captured image
        captured_image = [None]
        
        def update_camera_feed():
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                img = ImageTk.PhotoImage(image=img)
                camera_canvas.create_image(0, 0, anchor=tk.NW, image=img)
                camera_canvas.image = img
                if camera_window.winfo_exists():
                    camera_window.after(10, update_camera_feed)
            else:
                cap.release()
                camera_window.destroy()
                self.status_var.set("Camera disconnected")
        
        def capture_and_use():
            ret, frame = cap.read()
            if ret:
                captured_image[0] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Save to temp file
                temp_path = "temp_capture.jpg"
                cv2.imwrite(temp_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                self.current_image_path = temp_path
                # Load into detector
                self.detector.image = captured_image[0]
                self.detector.processed_image = captured_image[0].copy()
                # Update display
                self.update_image_display(self.detector.image, "Captured Image")
                # Close camera window
                cap.release()
                camera_window.destroy()
                self.status_var.set("Image captured from camera")
        
        # Bind capture button
        capture_btn.config(command=capture_and_use)
        
        # Start camera feed
        update_camera_feed()
        
        # Handle window close
        def on_close():
            cap.release()
            camera_window.destroy()
            self.status_var.set("Camera capture cancelled")
        
        camera_window.protocol("WM_DELETE_WINDOW", on_close)
    
    def update_image_display(self, image, title="Image"):
        """Update the image display with the given image"""
        self.ax.clear()
        self.ax.imshow(image)
        self.ax.set_title(title)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.canvas.draw()
    
    def apply_processing(self):
        """Apply selected processing methods to the image"""
        if self.detector.image is None:
            self.status_var.set("Please load an image first")
            return
        
        # Reset to original image
        self.detector.processed_image = self.detector.image.copy()
        
        # Apply point processing
        point_method = self.point_processing_var.get()
        if point_method == "brightness":
            self.detector.adjust_brightness(beta=30)
        elif point_method == "contrast":
            self.detector.adjust_contrast(alpha=1.5)
        elif point_method == "gamma":
            self.detector.gamma_correction(gamma=0.7)
        
        # Apply mask processing
        mask_method = self.mask_processing_var.get()
        if mask_method == "leaf_mask":
            mask = self.detector.create_leaf_mask()
            self.update_image_display(cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB), "Leaf Mask")
            return
        elif mask_method == "disease_spots":
            self.detector.detect_disease_spots()
        elif mask_method == "heatmap":
            self.detector.create_disease_heatmap()
        
        # Apply edge detection
        edge_method = self.edge_detection_var.get()
        if edge_method == "sobel":
            self.detector.detect_edges_sobel()
        elif edge_method == "prewitt":
            self.detector.detect_edges_prewitt()
        elif edge_method == "roberts":
            self.detector.detect_edges_roberts()
        
        # Update display
        self.update_image_display(self.detector.processed_image, "Processed Image")
        self.status_var.set("Processing applied")
    
    def show_histogram(self):
        """Show histogram of the current image"""
        if self.detector.image is None:
            self.status_var.set("Please load an image first")
            return
        
        # Create a new window for the histogram
        hist_window = tk.Toplevel(self.root)
        hist_window.title("Image Histogram")
        hist_window.geometry("800x600")
        
        # Create figure for histogram
        hist_fig = plt.Figure(figsize=(8, 6))
        hist_ax = hist_fig.add_subplot(111)
        
        # Plot histogram
        colors = ('r', 'g', 'b')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([self.detector.processed_image], [i], None, [256], [0, 256])
            hist_ax.plot(hist, color=color)
        
        hist_ax.set_xlim([0, 256])
        hist_ax.set_title("Image Histogram")
        hist_ax.set_xlabel("Pixel Value")
        hist_ax.set_ylabel("Frequency")
        
        # Add canvas to window
        hist_canvas = FigureCanvasTkAgg(hist_fig, master=hist_window)
        hist_canvas.draw()
        hist_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.status_var.set("Histogram displayed")
    
    def equalize_histogram(self):
        """Apply histogram equalization to the image"""
        if self.detector.image is None:
            self.status_var.set("Please load an image first")
            return
        
        self.detector.equalize_histogram()
        self.update_image_display(self.detector.processed_image, "Histogram Equalized")
        self.status_var.set("Histogram equalization applied")
    
    def compress_image(self):
        """Apply DCT compression to the image"""
        if self.detector.image is None:
            self.status_var.set("Please load an image first")
            return
        
        # Create a dialog to set compression quality
        dialog = tk.Toplevel(self.root)
        dialog.title("DCT Compression")
        dialog.geometry("300x150")
        dialog.resizable(False, False)
        
        # Add quality slider
        ttk.Label(dialog, text="Compression Quality:").pack(pady=(10, 0))
        quality_var = tk.IntVar(value=10)
        quality_slider = ttk.Scale(dialog, from_=1, to=20, variable=quality_var, orient=tk.HORIZONTAL, length=200)
        quality_slider.pack(pady=10)
        ttk.Label(dialog, text="1 = High Compression, 20 = High Quality").pack()
        
        def apply_compression():
            quality = quality_var.get()
            self.detector.compress_dct(quality=quality)
            self.update_image_display(self.detector.processed_image, f"DCT Compressed (Quality={quality})")
            self.status_var.set(f"DCT compression applied with quality {quality}")
            dialog.destroy()
        
        ttk.Button(dialog, text="Apply", command=apply_compression).pack(pady=10)
    
    def analyze_disease(self):
        """Analyze the image for plant disease"""
        if self.detector.image is None:
            self.status_var.set("Please load an image first")
            return
        
        self.status_var.set("Analyzing image for disease...")
        
        # Run analysis in a separate thread to keep UI responsive
        def run_analysis():
            try:
                # Extract features
                self.detector.create_leaf_mask()
                features = self.detector.extract_features()
                
                # Classify disease
                result = self.detector.classify_disease()
                
                # Create disease visualization
                disease_visual = self.detector.detect_disease_spots()
                self.update_image_display(disease_visual, "Disease Detection")
                
                # Update results text
                self.results_text.config(state=tk.NORMAL)
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, f"Disease: {result['disease']}\n")
                self.results_text.insert(tk.END, f"Confidence: {result['confidence']:.2f}\n")
                self.results_text.insert(tk.END, f"Severity: {result['severity']}\n\n")
                
                self.results_text.insert(tk.END, "Features:\n")
                for key, value in features.items():
                    if isinstance(value, float):
                        self.results_text.insert(tk.END, f"{key}: {value:.4f}\n")
                    else:
                        self.results_text.insert(tk.END, f"{key}: {value}\n")
                
                self.results_text.config(state=tk.DISABLED)
                
                # Update status
                self.status_var.set(f"Analysis complete: {result['disease']} detected")
                
            except Exception as e:
                self.status_var.set(f"Error during analysis: {str(e)}")
        
        # Start analysis thread
        self.processing_thread = threading.Thread(target=run_analysis)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def save_results(self):
        """Save the processed image and analysis results"""
        if self.detector.processed_image is None:
            self.status_var.set("No processed image to save")
            return
        
        # Ask for directory to save results
        save_dir = filedialog.askdirectory(title="Select Directory to Save Results")
        if not save_dir:
            return
        
        # Generate filename based on original image
        if self.current_image_path:
            base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
        else:
            base_name = "plant_analysis"
        
        # Save processed image
        img_path = os.path.join(save_dir, f"{base_name}_processed.jpg")
        cv2.imwrite(img_path, cv2.cvtColor(self.detector.processed_image, cv2.COLOR_RGB2BGR))
        
        # Save analysis results if available
        results_text = self.results_text.get(1.0, tk.END)
        if results_text.strip():
            txt_path = os.path.join(save_dir, f"{base_name}_results.txt")
            with open(txt_path, 'w') as f:
                f.write(results_text)
        
        # Save disease mask if available
        if self.detector.leaf_mask is not None:
            mask_path = os.path.join(save_dir, f"{base_name}_mask.jpg")
            cv2.imwrite(mask_path, self.detector.leaf_mask)
        
        self.status_var.set(f"Results saved to {save_dir}")


# Main function to run the application
def main():
    root = tk.Tk()
    app = DiseaseDetectionUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()