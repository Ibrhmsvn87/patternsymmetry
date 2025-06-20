#!/usr/bin/env python3
"""
Simple script to analyze a single image for pattern symmetry.
Usage: python analyze_single.py <image_file>
"""

import sys
from pattern_symmetry_analyzer import PatternSymmetryAnalyzer

def main():
    # Check if image file was provided
    if len(sys.argv) < 2:
        print("Usage: python analyze_single.py <image_file> [threshold] [smoothing]")
        print("\nExample: python analyze_single.py Sym1.png")
        print("         python analyze_single.py NotSym1.png 12.0 heavy")
        print("\nAvailable images:")
        import os
        images = [f for f in os.listdir('.') if f.endswith('.png') and not f.startswith('analysis_')]
        for img in sorted(images):
            print(f"  - {img}")
        print("\nSmoothing levels: light, medium (default), heavy")
        return
    
    image_file = sys.argv[1]
    
    # Optional: threshold can be provided as second argument
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0
    
    # Optional: smoothing level can be provided as third argument
    smoothing = sys.argv[3] if len(sys.argv) > 3 else 'medium'
    
    print(f"\nPattern Symmetry Analysis")
    print(f"========================")
    print(f"\nImage: {image_file}")
    print(f"Threshold: {threshold}%")
    print(f"Smoothing: {smoothing}")
    print(f"\nInstructions:")
    print(f"1. Click on the CENTER of the pattern (where the circular pattern originates)")
    print(f"2. Click the 'Analyze' button to proceed")
    print(f"\nNote: The center of the pattern may not be the center of the image!")
    
    # Create analyzer and analyze the image
    analyzer = PatternSymmetryAnalyzer(threshold_percentage=threshold, smoothing_level=smoothing)
    
    try:
        results = analyzer.analyze_image(image_file)
        
        print(f"\n{'='*50}")
        print(f"RESULTS")
        print(f"{'='*50}")
        print(f"Assessment: {'SYMMETRIC' if results['is_symmetric'] else 'ASYMMETRIC'}")
        print(f"Center: {results['center']}")
        print(f"Asymmetric rings: {results['asymmetric_rings']} / {results['total_rings']}")
        print(f"Asymmetry rate: {results['asymmetry_percentage']:.1f}%")
        print(f"\nOutput files:")
        print(f"  - analysis_{image_file.split('.')[0]}.png (visualization)")
        print(f"  - results_{image_file.split('.')[0]}.json (detailed data)")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 