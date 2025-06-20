import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Button
import os
from typing import Tuple, List, Dict
import json
from scipy import signal
from scipy.ndimage import median_filter

class PatternSymmetryAnalyzer:
    def __init__(self, threshold_percentage: float = 15.0, smoothing_level: str = 'medium'):
        """
        Initialize the Pattern Symmetry Analyzer
        
        Args:
            threshold_percentage: Percentage threshold for intensity variation
                                to consider regions as symmetric
            smoothing_level: Level of smoothing to apply ('light', 'medium', 'heavy')
        """
        self.threshold_percentage = threshold_percentage
        self.smoothing_level = smoothing_level
        self.center = None
        self.current_image = None
        self.current_image_path = None
        self.gray_image = None
        self.preprocessed_image = None
        
    def load_image(self, image_path: str) -> np.ndarray:
        """Load and return image in RGB format"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert BGR to RGB for matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.current_image = image_rgb
        self.current_image_path = image_path
        
        # Convert to grayscale for analysis
        self.gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply preprocessing
        self.preprocessed_image = self.preprocess_image(self.gray_image)
        
        return image_rgb
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply sophisticated preprocessing to remove spikes and smooth the image
        
        Args:
            image: Grayscale image
            
        Returns:
            Preprocessed image
        """
        processed = image.copy()
        
        # Step 1: Median filter to remove salt-and-pepper noise and spikes
        if self.smoothing_level in ['medium', 'heavy']:
            kernel_size = 5 if self.smoothing_level == 'medium' else 7
            processed = median_filter(processed, size=kernel_size)
            print(f"Applied median filter (kernel size: {kernel_size}) to remove spikes")
        
        # Step 2: Bilateral filter for edge-preserving smoothing
        # This maintains edges while smoothing uniform areas
        if self.smoothing_level == 'light':
            processed = cv2.bilateralFilter(processed, d=5, sigmaColor=50, sigmaSpace=50)
        elif self.smoothing_level == 'medium':
            processed = cv2.bilateralFilter(processed, d=9, sigmaColor=75, sigmaSpace=75)
        elif self.smoothing_level == 'heavy':
            processed = cv2.bilateralFilter(processed, d=13, sigmaColor=100, sigmaSpace=100)
        print(f"Applied bilateral filter ({self.smoothing_level} smoothing)")
        
        # Step 3: Gaussian blur for final smoothing
        if self.smoothing_level == 'medium':
            processed = cv2.GaussianBlur(processed, (5, 5), 1.0)
        elif self.smoothing_level == 'heavy':
            processed = cv2.GaussianBlur(processed, (7, 7), 1.5)
        
        # Optional Step 4: Morphological operations for very noisy images
        if self.smoothing_level == 'heavy':
            # Opening (erosion followed by dilation) to remove small noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
            print("Applied morphological opening to remove small noise regions")
        
        return processed
    
    def select_center(self, image: np.ndarray) -> Tuple[int, int]:
        """
        Interactive center selection using matplotlib
        Returns (x, y) coordinates of selected center
        """
        self.center = None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Show original image
        ax1.imshow(image)
        ax1.set_title("Click to select the center of the pattern")
        
        # Show preprocessed grayscale image
        ax2.imshow(self.preprocessed_image, cmap='gray')
        ax2.set_title(f"Preprocessed image ({self.smoothing_level} smoothing)")
        
        def onclick(event):
            if event.inaxes == ax1:
                self.center = (int(event.xdata), int(event.ydata))
                # Draw a marker at the selected point on both images
                ax1.plot(event.xdata, event.ydata, 'r+', markersize=15, markeredgewidth=3)
                ax2.plot(event.xdata, event.ydata, 'r+', markersize=15, markeredgewidth=3)
                ax1.set_title(f"Center selected at ({self.center[0]}, {self.center[1]})")
                plt.draw()
        
        # Connect the click event
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        
        # Add a continue button
        ax_button = plt.axes([0.7, 0.02, 0.15, 0.04])
        btn_continue = Button(ax_button, 'Analyze')
        
        def on_continue(event):
            plt.close(fig)
        
        btn_continue.on_clicked(on_continue)
        
        plt.tight_layout()
        plt.show()
        
        if self.center is None:
            raise ValueError("No center point was selected")
        
        return self.center
    
    def analyze_radial_symmetry(self, center: Tuple[int, int], 
                               num_angles: int = 36,
                               num_radii: int = 20) -> Dict:
        """
        Analyze radial symmetry around the selected center
        
        Args:
            center: (x, y) coordinates of center
            num_angles: Number of angular divisions (e.g., 36 = 10-degree increments)
            num_radii: Number of concentric circles to analyze
            
        Returns:
            Dictionary containing symmetry analysis results
        """
        if self.preprocessed_image is None:
            raise ValueError("No image loaded")
        
        height, width = self.preprocessed_image.shape
        cx, cy = center
        
        # Use the preprocessed image for analysis
        analysis_image = self.preprocessed_image
        
        # Determine maximum radius
        max_radius = min(cx, cy, width - cx, height - cy)
        radii = np.linspace(10, max_radius * 0.9, num_radii)
        
        # Angular divisions
        angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
        
        # Store intensity values for each ring
        ring_asymmetries = []
        detailed_analysis = []
        
        for r_idx, radius in enumerate(radii):
            intensities = []
            
            for angle in angles:
                # Calculate point coordinates
                x = int(cx + radius * np.cos(angle))
                y = int(cy + radius * np.sin(angle))
                
                # Ensure coordinates are within image bounds
                if 0 <= x < width and 0 <= y < height:
                    # Sample a small region around the point for robustness
                    x_min = max(0, x - 2)
                    x_max = min(width, x + 3)
                    y_min = max(0, y - 2)
                    y_max = min(height, y + 3)
                    
                    region = analysis_image[y_min:y_max, x_min:x_max]
                    if region.size > 0:
                        intensities.append(np.mean(region))
            
            if len(intensities) > num_angles * 0.8:  # Need at least 80% of points
                # Smooth the intensity profile to reduce noise impact
                intensities_array = np.array(intensities)
                
                # Apply circular smoothing with adaptive window size
                window_size = 3 if self.smoothing_level == 'light' else 5
                smoothed_intensities = signal.convolve(
                    np.pad(intensities_array, window_size, mode='wrap'),
                    np.ones(window_size) / window_size,
                    mode='valid'
                )[window_size:-window_size]
                
                # Calculate statistics on smoothed data
                mean_intensity = np.mean(smoothed_intensities)
                std_intensity = np.std(smoothed_intensities)
                
                # Calculate multiple metrics for robustness
                cv = (std_intensity / mean_intensity * 100) if mean_intensity > 0 else 0
                
                # Calculate range-based metric (more robust to outliers)
                intensity_range = np.max(smoothed_intensities) - np.min(smoothed_intensities)
                range_ratio = (intensity_range / mean_intensity * 100) if mean_intensity > 0 else 0
                
                # Use percentile-based metric for additional robustness
                p25 = np.percentile(smoothed_intensities, 25)
                p75 = np.percentile(smoothed_intensities, 75)
                iqr_ratio = ((p75 - p25) / mean_intensity * 100) if mean_intensity > 0 else 0
                
                # Combine metrics with weights
                # CV is sensitive to small variations, range_ratio to large deviations
                combined_metric = 0.4 * cv + 0.4 * range_ratio + 0.2 * iqr_ratio
                
                # Adaptive threshold based on mean intensity
                # Darker regions may have more relative noise
                adaptive_threshold = self.threshold_percentage
                if mean_intensity < 50:  # Dark regions
                    adaptive_threshold *= 1.5
                elif mean_intensity > 200:  # Very bright regions
                    adaptive_threshold *= 0.8
                
                # Additional adjustment based on smoothing level
                if self.smoothing_level == 'heavy':
                    # With heavy smoothing, we can be slightly stricter
                    adaptive_threshold *= 0.9
                
                # Check if variation exceeds threshold
                is_asymmetric = combined_metric > adaptive_threshold
                
                ring_asymmetries.append(is_asymmetric)
                detailed_analysis.append({
                    'radius': float(radius),
                    'mean_intensity': float(mean_intensity),
                    'std_intensity': float(std_intensity),
                    'coefficient_of_variation': float(cv),
                    'range_ratio': float(range_ratio),
                    'iqr_ratio': float(iqr_ratio),
                    'combined_metric': float(combined_metric),
                    'adaptive_threshold': float(adaptive_threshold),
                    'is_asymmetric': bool(is_asymmetric),
                    'intensities': [float(x) for x in smoothed_intensities]
                })
        
        # Overall assessment with additional criteria
        asymmetric_rings = sum(ring_asymmetries)
        total_rings = len(ring_asymmetries)
        
        # Pattern is asymmetric if:
        # 1. More than 20% of rings are asymmetric, OR
        # 2. Any 3 consecutive rings are asymmetric (indicates localized asymmetry)
        consecutive_asymmetric = 0
        max_consecutive = 0
        for is_asym in ring_asymmetries:
            if is_asym:
                consecutive_asymmetric += 1
                max_consecutive = max(max_consecutive, consecutive_asymmetric)
            else:
                consecutive_asymmetric = 0
        
        is_pattern_symmetric = not (
            (asymmetric_rings / total_rings > 0.2) if total_rings > 0 else False or 
            (max_consecutive >= 3)
        )
        
        return {
            'is_symmetric': bool(is_pattern_symmetric),
            'asymmetric_rings': int(asymmetric_rings),
            'total_rings': int(total_rings),
            'asymmetry_percentage': float((asymmetric_rings / total_rings * 100) if total_rings > 0 else 0),
            'max_consecutive_asymmetric': int(max_consecutive),
            'detailed_analysis': detailed_analysis,
            'center': center,
            'threshold_percentage': float(self.threshold_percentage),
            'smoothing_level': self.smoothing_level
        }
    
    def visualize_analysis(self, analysis_results: Dict):
        """Create comprehensive visualization of the symmetry analysis"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Original image with center and rings
        ax1 = axes[0, 0]
        ax1.imshow(self.current_image)
        ax1.set_title("Original Image with Analysis Overlay")
        
        # Draw center
        cx, cy = analysis_results['center']
        ax1.plot(cx, cy, 'r+', markersize=15, markeredgewidth=3)
        
        # Draw concentric circles
        for ring_data in analysis_results['detailed_analysis']:
            radius = ring_data['radius']
            color = 'red' if ring_data['is_asymmetric'] else 'green'
            circle = Circle((cx, cy), radius, fill=False, 
                          edgecolor=color, linewidth=2, alpha=0.6)
            ax1.add_patch(circle)
        
        # 2. Original grayscale
        ax2 = axes[0, 1]
        ax2.imshow(self.gray_image, cmap='gray')
        ax2.set_title("Original Grayscale")
        ax2.plot(cx, cy, 'r+', markersize=15, markeredgewidth=3)
        
        # 3. Preprocessed image
        ax3 = axes[0, 2]
        ax3.imshow(self.preprocessed_image, cmap='gray')
        ax3.set_title(f"Preprocessed ({self.smoothing_level} smoothing)")
        ax3.plot(cx, cy, 'r+', markersize=15, markeredgewidth=3)
        
        # 4. Combined metric by radius
        ax4 = axes[1, 0]
        radii = [r['radius'] for r in analysis_results['detailed_analysis']]
        combined_metrics = [r['combined_metric'] for r in analysis_results['detailed_analysis']]
        adaptive_thresholds = [r['adaptive_threshold'] for r in analysis_results['detailed_analysis']]
        
        ax4.plot(radii, combined_metrics, 'b-', linewidth=2, label='Combined Metric')
        ax4.plot(radii, adaptive_thresholds, 'r--', linewidth=2, label='Adaptive Threshold')
        ax4.set_xlabel('Radius (pixels)')
        ax4.set_ylabel('Asymmetry Metric (%)')
        ax4.set_title('Asymmetry Analysis by Radius')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Difference image (showing preprocessing effect)
        ax5 = axes[1, 1]
        diff_image = np.abs(self.gray_image.astype(float) - self.preprocessed_image.astype(float))
        im = ax5.imshow(diff_image, cmap='hot')
        ax5.set_title("Preprocessing Effect (removed noise/spikes)")
        plt.colorbar(im, ax=ax5)
        
        # 6. Summary text
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        summary_text = f"""
SYMMETRY ANALYSIS RESULTS
{'='*35}

Image: {os.path.basename(self.current_image_path)}
Center: ({cx}, {cy})

Overall Assessment: {'SYMMETRIC' if analysis_results['is_symmetric'] else 'ASYMMETRIC'}

Asymmetric Rings: {analysis_results['asymmetric_rings']} / {analysis_results['total_rings']}
Asymmetry Rate: {analysis_results['asymmetry_percentage']:.1f}%
Max Consecutive: {analysis_results['max_consecutive_asymmetric']}

Settings:
- Base Threshold: {self.threshold_percentage}%
- Smoothing: {self.smoothing_level}

{'='*35}

Analysis Criteria:
- Ring asymmetric if metric > threshold
- Pattern asymmetric if:
  • >20% rings asymmetric OR
  • 3+ consecutive rings asymmetric

Legend:
- Green circles: Symmetric regions
- Red circles: Asymmetric regions
        """
        
        ax6.text(0.05, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center', transform=ax6.transAxes)
        
        plt.tight_layout()
        
        # Save the figure
        output_name = os.path.basename(self.current_image_path).split('.')[0]
        output_path = f"analysis_{output_name}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Analysis visualization saved to: {output_path}")
        
        plt.show()
        
        return fig
    
    def save_results(self, analysis_results: Dict, output_path: str = None):
        """Save analysis results to JSON file"""
        if output_path is None:
            base_name = os.path.basename(self.current_image_path).split('.')[0]
            output_path = f"results_{base_name}.json"
        
        # The results should already have all numpy types converted
        # Double-check and convert any remaining numpy types
        def convert_numpy_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int_)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float_)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(val) for key, val in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Convert all numpy types recursively
        results_to_save = convert_numpy_types(analysis_results)
        
        with open(output_path, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"Results saved to: {output_path}")
    
    def analyze_image(self, image_path: str):
        """Complete analysis workflow for a single image"""
        print(f"\nAnalyzing: {image_path}")
        print(f"Preprocessing with {self.smoothing_level} smoothing...")
        
        # Load image
        image = self.load_image(image_path)
        
        # Select center
        center = self.select_center(image)
        
        # Analyze symmetry
        results = self.analyze_radial_symmetry(center)
        
        # Visualize results
        self.visualize_analysis(results)
        
        # Save results
        self.save_results(results)
        
        return results


# Simple usage example
if __name__ == "__main__":
    import sys
    
    print("Pattern Symmetry Analyzer")
    print("========================")
    print("\nUsage: python pattern_symmetry_analyzer.py <image_file> [smoothing_level]")
    print("Smoothing levels: light, medium (default), heavy")
    print("\nOr use analyze_single.py for more options")
    
    if len(sys.argv) > 1:
        smoothing = sys.argv[2] if len(sys.argv) > 2 else 'medium'
        analyzer = PatternSymmetryAnalyzer(smoothing_level=smoothing)
        analyzer.analyze_image(sys.argv[1])
    else:
        print("\nNo image file specified. Use:")
        print("  python analyze_single.py <image_file>") 