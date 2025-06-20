# Pattern Symmetry Analyzer - Algorithm Improvements

## Problem
The original algorithm was too strict, flagging symmetric patterns as asymmetric due to minor variations in intensity that could be caused by:
- Image noise
- Slight imperfections in the pattern
- Compression artifacts
- Natural variations in real-world patterns

## Solutions Implemented

### 1. **Noise Reduction**
- Added Gaussian blur preprocessing (`5x5` kernel) to smooth the image before analysis
- Increased sampling region from `3x3` to `5x5` pixels for more robust intensity measurements
- Applied circular smoothing to intensity profiles around each ring

### 2. **More Robust Metrics**
Instead of relying solely on coefficient of variation (CV), we now use a **combined metric**:
- **40%** Coefficient of Variation (sensitive to small variations)
- **40%** Range Ratio (max-min intensity difference)
- **20%** Interquartile Range Ratio (robust to outliers)

### 3. **Adaptive Thresholds**
The threshold now adapts based on the mean intensity of each ring:
- **Dark regions** (intensity < 50): Threshold × 1.5 (more tolerant)
- **Normal regions**: Base threshold
- **Bright regions** (intensity > 200): Threshold × 0.8 (slightly stricter)

### 4. **Improved Symmetry Criteria**
Pattern is considered **asymmetric** only if:
- More than **20%** of rings are asymmetric, OR
- **3 or more consecutive** rings are asymmetric

This prevents isolated noisy rings from causing false positives.

### 5. **Default Threshold Change**
- Increased from **10%** to **15%** to be more tolerant of natural variations

## Benefits
- **Better handling of real-world images** with natural imperfections
- **Reduced false positives** on symmetric patterns
- **Still sensitive to genuine asymmetries** through the consecutive ring criterion
- **More robust to noise** and image quality variations

## Usage Tips
- For high-quality, clean images: Use threshold 10-12%
- For typical images: Use default 15%
- For noisy or compressed images: Use threshold 18-20%
- For very strict analysis: Use threshold 5-8% 