# Pattern Symmetry Analyzer

A Python tool for analyzing radial symmetry in pattern images. This tool allows users to interactively select the center of a pattern and automatically assesses whether the pattern is symmetric or asymmetric based on intensity variations in concentric circles.

## Features

- **Interactive Center Selection**: Click on the image to select the pattern center
- **Radial Symmetry Analysis**: Analyzes intensity variations in concentric circles
- **Automatic Assessment**: Determines if patterns are symmetric or asymmetric
- **Comprehensive Visualization**: Shows analysis results with color-coded rings
- **Batch Processing**: Analyze multiple images in a directory
- **Flexible Pattern Support**: Works with circular, doughnut-shaped, and multifocal patterns

## Installation

1. Clone this repository:
```bash
git clone https://github.com/Ibrhmsvn87/patternsymmetry.git
cd patternsymmetry
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Analyzing a Single Image

```python
from pattern_symmetry_analyzer import PatternSymmetryAnalyzer

# Create analyzer with custom threshold (default is 10%)
analyzer = PatternSymmetryAnalyzer(threshold_percentage=10.0)

# Analyze a single image
results = analyzer.analyze_image("path/to/your/image.png")
```

### Batch Analysis

To analyze all images in the current directory:

```bash
python pattern_symmetry_analyzer.py
```

Or from Python:

```python
from pattern_symmetry_analyzer import batch_analyze_directory

# Analyze all images in a directory
results = batch_analyze_directory("path/to/directory", threshold_percentage=10.0)
```

## How It Works

1. **Center Selection**: User clicks on the image to identify the pattern center
2. **Radial Sampling**: The tool samples intensity values at multiple points along concentric circles
3. **Statistical Analysis**: For each ring, it calculates:
   - Mean intensity
   - Standard deviation
   - Coefficient of variation (CV)
4. **Symmetry Assessment**: 
   - A ring is considered asymmetric if CV > threshold
   - Pattern is asymmetric if ANY ring shows asymmetry
   - This ensures asymmetry is not masked by symmetric regions

## Output

For each analyzed image, the tool generates:

1. **Analysis Visualization** (`analysis_<imagename>.png`):
   - Original image with overlay (green = symmetric rings, red = asymmetric)
   - Grayscale intensity view
   - Graph showing intensity variation by radius
   - Summary statistics

2. **JSON Results** (`results_<imagename>.json`):
   - Detailed numerical results
   - Ring-by-ring analysis
   - Overall symmetry assessment

## Parameters

- **threshold_percentage**: Percentage of intensity variation allowed for a ring to be considered symmetric (default: 10%)
- **num_angles**: Number of angular divisions for sampling (default: 36 = 10° increments)
- **num_radii**: Number of concentric circles to analyze (default: 20)

## Example Results

- **Symmetric Pattern**: All rings show intensity variation below threshold
- **Asymmetric Pattern**: One or more rings exceed the intensity variation threshold

## Notes

- The tool converts images to grayscale for intensity analysis
- Works best with patterns that have clear intensity variations
- Supports common image formats: PNG, JPG, JPEG, BMP, TIFF 