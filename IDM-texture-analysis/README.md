# Image Description & Modeling — Texture Analysis (HLBP)

This project implements **texture-based image description and classification** using Histogram Local Binary Patterns (HLBP) on histopathology image tiles.

The pipeline focuses on **classical, interpretable texture analysis** rather than deep learning, emphasizing robustness to illumination variation and noise using tolerance-based binary encoding.

## Overview

Texture plays a critical role in medical image analysis. This project investigates Histogram Local Binary Patterns (HLBP) as a handcrafted texture descriptor for distinguishing histopathology tissue patterns.

The work was completed as part of an **Image Description and Modeling** academic project.

## Methodology

The pipeline includes:

- Dataset loading and automatic label extraction
- Texture feature extraction using:
  - Standard LBP (3×3)
  - Histogram LBP (HLBP) with tolerance parameter *(m)*
  - Multi-scale HLBP (R = 1, 2, 3)
- Grayscale and Color (Lab space) feature comparison
- Feature vector construction from normalized histograms
- Classification using:
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM, RBF kernel)
- Parameter analysis:
  - Effect of tolerance *(m)*
  - Effect of neighborhood size *(k)*
- Visualization of LBP vs HLBP representations and histogram distributions

## Experiments

The following experiments were conducted:

1. Grayscale HLBP texture classification  
2. Color HLBP (Lab space) feature analysis  
3. Effect of tolerance parameter *(m)* on robustness  
4. Effect of K in KNN classification  
5. Comparison of LBP vs HLBP texture encoding  

## Results

- Color HLBP features improved classification performance over grayscale.
- Moderate tolerance *(m ≈ 0.05)* produced the most stable results.
- HLBP showed improved robustness to illumination variation compared to standard LBP.
- Classical texture features demonstrated strong interpretability and competitive performance without deep learning.

(See notebook and report for detailed metrics and confusion matrices.)

## Visualization

The project includes visualization comparing:

- Original image vs grayscale
- LBP vs HLBP texture maps
- Histogram comparison
- Pixel distribution differences

Saved visualization: `figures/hlbp_visualization_report.png`

## Project Structure

```
IDM-texture-analysis/
│── notebook.ipynb
│── figures/
│   └── hlbp_visualization_report.png
│── README.md
│── report.pdf
```

## Requirements

- Python 3.x
- NumPy
- scikit-image
- scikit-learn
- matplotlib

## Notes

This project is for **academic and research purposes only**.  
All images belong to their respective dataset owners and are used strictly for non-commercial research.

## Author

**Gagandeep Kaur**  
Master’s Student — Computer Vision
