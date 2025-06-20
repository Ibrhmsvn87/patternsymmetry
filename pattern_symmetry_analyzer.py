import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Button
import os
from typing import Tuple, List, Dict
import json

class PatternSymmetryAnalyzer:
    def __init__(self, threshold_percentage: float = 10.0):
        """
        Initialize the Pattern Symmetry Analyzer
        
        Args:
            threshold_percentage: Percentage threshold for intensity variation
                                to consider regions as symmetric
        """
        self.threshold_percentage = threshold_percentage
        self.center = None
        self.current_image = None
        self.current_image_path = None
        self.gray_image = None
        
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
        
        return image_rgb
    
    def select_center(self, image: np.ndarray) -> Tuple[int, int]:
        """
        Interactive center selection using matplotlib
        Returns (x, y) coordinates of selected center
        """
        self.center = None
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(image)
        ax.set_title("Click to select the center of the pattern")
        
        def onclick(event):
            if event.inaxes == ax:
                self.center = (int(event.xdata), int(event.ydata))
                # Draw a marker at the selected point
                ax.plot(event.xdata, event.ydata, 'r+', markersize=15, markeredgewidth=3)
                ax.set_title(f"Center selected at ({self.center[0]}, {self.center[1]})")
                plt.draw()
        
        # Connect the click event
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        
        # Add a continue button
        ax_button = plt.axes([0.7, 0.05, 0.15, 0.04])
        btn_continue = Button(ax_button, 'Analyze')
        
        def on_continue(event):
            plt.close(fig)
        
        btn_continue.on_clicked(on_continue)
        
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
        if self.gray_image is None:
            raise ValueError("No image loaded")
        
        height, width = self.gray_image.shape
        cx, cy = center
        
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
                    x_min = max(0, x - 1)
                    x_max = min(width, x + 2)
                    y_min = max(0, y - 1)
                    y_max = min(height, y + 2)
                    
                    region = self.gray_image[y_min:y_max, x_min:x_max]
                    if region.size > 0:
                        intensities.append(np.mean(region))
            
            if len(intensities) > 0:
                # Calculate statistics for this ring
                mean_intensity = np.mean(intensities)
                std_intensity = np.std(intensities)
                
                # Calculate coefficient of variation (CV) as a measure of asymmetry
                cv = (std_intensity / mean_intensity * 100) if mean_intensity > 0 else 0
                
                # Check if variation exceeds threshold
                is_asymmetric = cv > self.threshold_percentage
                
                ring_asymmetries.append(is_asymmetric)
                detailed_analysis.append({
                    'radius': radius,
                    'mean_intensity': mean_intensity,
                    'std_intensity': std_intensity,
                    'coefficient_of_variation': cv,
                    'is_asymmetric': is_asymmetric,
                    'intensities': intensities
                })
        
        # Overall assessment
        asymmetric_rings = sum(ring_asymmetries)
        total_rings = len(ring_asymmetries)
        
        # Pattern is asymmetric if ANY ring shows asymmetry
        is_pattern_symmetric = asymmetric_rings == 0
        
        return {
            'is_symmetric': is_pattern_symmetric,
            'asymmetric_rings': asymmetric_rings,
            'total_rings': total_rings,
            'asymmetry_percentage': (asymmetric_rings / total_rings * 100) if total_rings > 0 else 0,
            'detailed_analysis': detailed_analysis,
            'center': center,
            'threshold_percentage': self.threshold_percentage
        }
    
    def visualize_analysis(self, analysis_results: Dict):
        """Create comprehensive visualization of the symmetry analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        
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
        
        # 2. Grayscale intensity image
        ax2 = axes[0, 1]
        ax2.imshow(self.gray_image, cmap='gray')
        ax2.set_title("Grayscale Intensity")
        ax2.plot(cx, cy, 'r+', markersize=15, markeredgewidth=3)
        
        # 3. Coefficient of Variation by radius
        ax3 = axes[1, 0]
        radii = [r['radius'] for r in analysis_results['detailed_analysis']]
        cvs = [r['coefficient_of_variation'] for r in analysis_results['detailed_analysis']]
        
        ax3.plot(radii, cvs, 'b-', linewidth=2)
        ax3.axhline(y=self.threshold_percentage, color='r', linestyle='--', 
                   label=f'Threshold ({self.threshold_percentage}%)')
        ax3.set_xlabel('Radius (pixels)')
        ax3.set_ylabel('Coefficient of Variation (%)')
        ax3.set_title('Intensity Variation by Radius')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Summary text
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_text = f"""
SYMMETRY ANALYSIS RESULTS
{'='*30}

Image: {os.path.basename(self.current_image_path)}
Center: ({cx}, {cy})

Overall Assessment: {'SYMMETRIC' if analysis_results['is_symmetric'] else 'ASYMMETRIC'}

Asymmetric Rings: {analysis_results['asymmetric_rings']} / {analysis_results['total_rings']}
Asymmetry Rate: {analysis_results['asymmetry_percentage']:.1f}%

Threshold: {self.threshold_percentage}% intensity variation

{'='*30}

Legend:
- Green circles: Symmetric regions
- Red circles: Asymmetric regions
- Red cross: Selected center point
        """
        
        ax4.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
                verticalalignment='center', transform=ax4.transAxes)
        
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
        
        # Convert numpy arrays to lists for JSON serialization
        results_copy = analysis_results.copy()
        for ring_data in results_copy['detailed_analysis']:
            ring_data['intensities'] = [float(x) for x in ring_data['intensities']]
        
        with open(output_path, 'w') as f:
            json.dump(results_copy, f, indent=2)
        
        print(f"Results saved to: {output_path}")
    
    def analyze_image(self, image_path: str):
        """Complete analysis workflow for a single image"""
        print(f"\nAnalyzing: {image_path}")
        
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


def batch_analyze_directory(directory_path: str = ".", 
                           threshold_percentage: float = 10.0):
    """Analyze all images in a directory"""
    analyzer = PatternSymmetryAnalyzer(threshold_percentage=threshold_percentage)
    
    # Find all image files
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    image_files = []
    
    for file in os.listdir(directory_path):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            # Skip analysis output files
            if not file.startswith('analysis_'):
                image_files.append(os.path.join(directory_path, file))
    
    print(f"Found {len(image_files)} images to analyze")
    
    results_summary = []
    
    for image_path in image_files:
        try:
            results = analyzer.analyze_image(image_path)
            results_summary.append({
                'image': os.path.basename(image_path),
                'is_symmetric': results['is_symmetric'],
                'asymmetry_percentage': results['asymmetry_percentage']
            })
        except Exception as e:
            print(f"Error analyzing {image_path}: {str(e)}")
    
    # Print summary
    print("\n" + "="*50)
    print("BATCH ANALYSIS SUMMARY")
    print("="*50)
    
    for result in results_summary:
        status = "SYMMETRIC" if result['is_symmetric'] else "ASYMMETRIC"
        print(f"{result['image']}: {status} ({result['asymmetry_percentage']:.1f}% asymmetry)")
    
    return results_summary


if __name__ == "__main__":
    # Example usage
    print("Pattern Symmetry Analyzer")
    print("========================")
    print("\nThis tool analyzes radial symmetry in patterns.")
    print("You will be asked to click on the center of each pattern.")
    print("\nStarting batch analysis of current directory...")
    
    # Analyze all images in current directory
    batch_analyze_directory(".", threshold_percentage=10.0) 