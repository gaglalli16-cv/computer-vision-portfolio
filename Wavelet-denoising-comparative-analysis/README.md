# Wavelet Multiscale Image Denoising — Comparative Analysis

This project implements and analyzes **multiscale wavelet-based image denoising** techniques for different noise types and datasets. The work follows the *Image Description and Modelling (IDM)* research project guidelines.

The objective is to evaluate how different wavelet families and thresholding strategies perform in removing noise while preserving image structure.

---

## Overview

Wavelet transforms provide a multiresolution representation of images in both spatial and frequency domains. Because most image energy is concentrated in a small number of coefficients, noise can be removed by thresholding small wavelet coefficients.

This project explores:

- Wavelet-based denoising using multiple wavelet families  
- Hard and soft thresholding strategies  
- Comparison with Gaussian blur baseline  
- Performance across multiple noise types and datasets  

The implementation follows the standard wavelet denoising pipeline:

1. Wavelet decomposition of noisy image  
2. Thresholding of wavelet coefficients  
3. Reconstruction of denoised image  

---

## Datasets

The project evaluates denoising performance on three image domains:

- **Histology images**
- **Radiology images**
- **Products / dense object images**

---

## Noise Types

The following noise models are applied:

- Additive Gaussian noise  
- Speckle noise (multiplicative)  
- Salt & Pepper noise  

---

## Methods

### Wavelet Families
- Daubechies (db4, db7)
- Haar
- Symlets (sym4)
- Coiflets (coif2)

### Thresholding Modes
- Hard thresholding  
- Soft thresholding  

### Baseline
- Gaussian blur filtering

---

## Evaluation Metrics

Performance is evaluated using:

- **MSE** — Mean Squared Error  
- **PSNR** — Peak Signal-to-Noise Ratio  
- **SSIM** — Structural Similarity Index  

---

## Results

The project performs a comparative evaluation across:

- Wavelet families
- Noise types
- Datasets

Outputs include:

- Best-performing wavelet per dataset and noise type  
- IEEE-style transposed performance tables  
- PSNR comparison plots across wavelet families  
- CSV result summaries  

Wavelet-based denoising consistently outperforms Gaussian blur in Gaussian noise scenarios, while performance varies for Speckle and Salt & Pepper noise depending on wavelet choice.

---

## Project Structure

images/
│
├── histology/          # Clean histology images
├── products/           # Clean product images
├── radiology/          # Clean radiology images
├── noisy_images/       # Generated noisy images
└── denoised_images/    # Outputs from denoising methods


results/
│
├── metrics_wavelet_comparison_*.csv   # Raw evaluation metrics
├── Table_Best_*.csv                   # Best-performing methods (formatted)
└── visual_*.png                       # Visualization outputs


Scripts
│
├── wavelet_denoising.py      # Main denoising pipeline
├── comparison.py             # Generates summary comparison tables
├── analyze_results.py        # Analysis + plots
└── Wavelet_Denoising_Analysis.ipynb   # Notebook version (visual + experiments)
