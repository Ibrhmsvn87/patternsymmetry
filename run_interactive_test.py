from pattern_symmetry_analyzer import PatternSymmetryAnalyzer

print("Interactive Pattern Symmetry Analysis")
print("====================================")
print("\nThis will analyze 2 sample images - one symmetric and one asymmetric.")
print("For each image:")
print("1. Click on the CENTER of the pattern (not the center of the image!)")
print("2. Click the 'Analyze' button")
print("\nThe pattern center is where the circular/radial pattern originates from.")

# Create analyzer
analyzer = PatternSymmetryAnalyzer(threshold_percentage=10.0)

# Test images
test_images = ["Sym1.png", "NotSym1.png"]

for image_path in test_images:
    print(f"\n{'='*50}")
    print(f"Analyzing: {image_path}")
    print(f"{'='*50}")
    
    try:
        results = analyzer.analyze_image(image_path)
        
        print(f"\nResults for {image_path}:")
        print(f"  Assessment: {'SYMMETRIC' if results['is_symmetric'] else 'ASYMMETRIC'}")
        print(f"  Asymmetric rings: {results['asymmetric_rings']} / {results['total_rings']}")
        print(f"  Asymmetry rate: {results['asymmetry_percentage']:.1f}%")
        print(f"  Files created: analysis_{image_path.split('.')[0]}.png and results_{image_path.split('.')[0]}.json")
        
    except Exception as e:
        print(f"Error: {str(e)}")

print("\n" + "="*50)
print("Analysis complete! Check the output files for detailed visualization.") 