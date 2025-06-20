"""
Example usage of the Pattern Symmetry Analyzer

This script demonstrates various ways to use the pattern symmetry analyzer
with different configurations and use cases.
"""

from pattern_symmetry_analyzer import PatternSymmetryAnalyzer, batch_analyze_directory
import os

def analyze_single_image_example():
    """Example: Analyze a single image with custom threshold"""
    print("\n" + "="*50)
    print("EXAMPLE 1: Single Image Analysis")
    print("="*50)
    
    # Create analyzer with 15% threshold (more tolerant to variations)
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
    
    # Create analyzer with 5% threshold (very strict)
    analyzer = PatternSymmetryAnalyzer(threshold_percentage=5.0)
    
    print("Using strict 5% threshold - will detect even small asymmetries")
    
    if os.path.exists("NotSym1.png"):
        results = analyzer.analyze_image("NotSym1.png")
        print(f"\nWith strict threshold, found {results['asymmetric_rings']} asymmetric rings")

def batch_analysis_example():
    """Example: Batch analyze directory with custom settings"""
    print("\n" + "="*50)
    print("EXAMPLE 3: Batch Directory Analysis")
    print("="*50)
    
    # Analyze current directory with 10% threshold
    results_summary = batch_analyze_directory(".", threshold_percentage=10.0)
    
    # Count symmetric vs asymmetric
    symmetric_count = sum(1 for r in results_summary if r['is_symmetric'])
    asymmetric_count = len(results_summary) - symmetric_count
    
    print(f"\nBatch analysis complete:")
    print(f"Total images analyzed: {len(results_summary)}")
    print(f"Symmetric patterns: {symmetric_count}")
    print(f"Asymmetric patterns: {asymmetric_count}")

def analyze_specific_pattern_types():
    """Example: Analyze different pattern types"""
    print("\n" + "="*50)
    print("EXAMPLE 4: Different Pattern Types")
    print("="*50)
    
    analyzer = PatternSymmetryAnalyzer(threshold_percentage=10.0)
    
    # Define pattern categories
    pattern_types = {
        "Symmetric": ["Sym1.png", "Sym2.png", "Sym3.png", "Sym4.png"],
        "Asymmetric": ["NotSym1.png", "NotSym2.png", "NotSym3.png"]
    }
    
    for pattern_type, files in pattern_types.items():
        print(f"\nAnalyzing {pattern_type} patterns:")
        for file in files:
            if os.path.exists(file):
                try:
                    results = analyzer.analyze_image(file)
                    assessment = "SYMMETRIC" if results['is_symmetric'] else "ASYMMETRIC"
                    match = "✓" if assessment == pattern_type.upper() else "✗"
                    print(f"  {file}: {assessment} {match}")
                except Exception as e:
                    print(f"  {file}: Error - {str(e)}")

def custom_analysis_parameters():
    """Example: Use custom analysis parameters"""
    print("\n" + "="*50)
    print("EXAMPLE 5: Custom Analysis Parameters")
    print("="*50)
    
    analyzer = PatternSymmetryAnalyzer(threshold_percentage=12.0)
    
    if os.path.exists("Sym1.png"):
        # Load image first
        image = analyzer.load_image("Sym1.png")
        
        # Select center interactively
        center = analyzer.select_center(image)
        
        # Analyze with custom parameters
        # More angles = finer angular resolution
        # More radii = more detailed radial analysis
        results = analyzer.analyze_radial_symmetry(
            center=center,
            num_angles=72,  # 5-degree increments
            num_radii=30    # 30 concentric circles
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
    print("\nThese examples demonstrate different ways to use the analyzer.")
    print("You will need to interact with matplotlib windows to select centers.")
    
    # Run examples
    try:
        # Example 1: Single image
        analyze_single_image_example()
        
        # Example 2: Strict threshold
        # analyze_with_strict_threshold()
        
        # Example 3: Batch analysis
        # analyze_batch_analysis_example()
        
        # Example 4: Pattern types
        # analyze_specific_pattern_types()
        
        # Example 5: Custom parameters
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