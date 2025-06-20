"""
Example usage of the Pattern Symmetry Analyzer

This script demonstrates various ways to use the pattern symmetry analyzer
for single image analysis with different configurations.
"""

from pattern_symmetry_analyzer import PatternSymmetryAnalyzer
import os

def analyze_single_image_example():
    """Example: Analyze a single image with custom threshold"""
    print("\n" + "="*50)
    print("EXAMPLE 1: Single Image Analysis")
    print("="*50)
    
    # Create analyzer with 15% threshold (default)
    analyzer = PatternSymmetryAnalyzer(threshold_percentage=15.0)
    
    # Check if example image exists
    if os.path.exists("Sym1.png"):
        results = analyzer.analyze_image("Sym1.png")
        
        print(f"\nAnalysis complete!")
        print(f"Pattern is: {'SYMMETRIC' if results['is_symmetric'] else 'ASYMMETRIC'}")
        print(f"Asymmetric rings: {results['asymmetric_rings']} out of {results['total_rings']}")
    else:
        print("Example image 'Sym1.png' not found")

def analyze_with_strict_threshold():
    """Example: Use stricter threshold for more sensitive detection"""
    print("\n" + "="*50)
    print("EXAMPLE 2: Strict Threshold Analysis")
    print("="*50)
    
    # Create analyzer with 8% threshold (very strict)
    analyzer = PatternSymmetryAnalyzer(threshold_percentage=8.0)
    
    print("Using strict 8% threshold - will detect even small asymmetries")
    
    if os.path.exists("NotSym1.png"):
        results = analyzer.analyze_image("NotSym1.png")
        print(f"\nWith strict threshold, found {results['asymmetric_rings']} asymmetric rings")

def analyze_with_tolerant_threshold():
    """Example: Use more tolerant threshold for noisy images"""
    print("\n" + "="*50)
    print("EXAMPLE 3: Tolerant Threshold Analysis")
    print("="*50)
    
    # Create analyzer with 20% threshold (very tolerant)
    analyzer = PatternSymmetryAnalyzer(threshold_percentage=20.0)
    
    print("Using tolerant 20% threshold - good for noisy or imperfect images")
    
    if os.path.exists("Sym2.png"):
        results = analyzer.analyze_image("Sym2.png")
        print(f"\nWith tolerant threshold: {'SYMMETRIC' if results['is_symmetric'] else 'ASYMMETRIC'}")

def custom_analysis_parameters():
    """Example: Use custom analysis parameters"""
    print("\n" + "="*50)
    print("EXAMPLE 4: Custom Analysis Parameters")
    print("="*50)
    
    analyzer = PatternSymmetryAnalyzer(threshold_percentage=12.0)
    
    if os.path.exists("Sym1.png"):
        # Load image first
        image = analyzer.load_image("Sym1.png")
        
        # Select center interactively
        print("Please click on the center of the pattern...")
        center = analyzer.select_center(image)
        
        # Analyze with custom parameters
        # More angles = finer angular resolution
        # More radii = more detailed radial analysis
        results = analyzer.analyze_radial_symmetry(
            center=center,
            num_angles=72,  # 5-degree increments instead of 10
            num_radii=30    # 30 concentric circles instead of 20
        )
        
        print(f"\nCustom analysis with high resolution:")
        print(f"Angular resolution: 5 degrees (72 samples)")
        print(f"Radial resolution: 30 rings")
        print(f"Result: {'SYMMETRIC' if results['is_symmetric'] else 'ASYMMETRIC'}")
        
        # Visualize and save
        analyzer.visualize_analysis(results)
        analyzer.save_results(results, "custom_analysis_results.json")

if __name__ == "__main__":
    print("Pattern Symmetry Analyzer - Examples")
    print("====================================")
    print("\nThese examples demonstrate different ways to analyze single images.")
    print("You will need to interact with matplotlib windows to select centers.")
    print("\nPress Ctrl+C to skip to the next example.")
    
    # Run examples
    try:
        # Example 1: Default threshold
        analyze_single_image_example()
        
        # Uncomment to run other examples:
        # analyze_with_strict_threshold()
        # analyze_with_tolerant_threshold()
        # custom_analysis_parameters()
        
        print("\n" + "="*50)
        print("Examples complete! Check the generated files:")
        print("- analysis_*.png : Visual results")
        print("- results_*.json : Numerical data")
        print("="*50)
        
    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user.")
    except Exception as e:
        print(f"\n\nError running examples: {str(e)}") 