## Gagandeep Kaur — Computer Vision Portfolio

MSc Computer Vision student with a background in Computer Science and experience in industrial automation systems. This repository contains selected projects from academic coursework and independent study.

---
## Projects

### 1. Wildlife Species Image Classification  
**PyTorch · ResNet50 · Transfer Learning · Data Augmentation**

Deep learning–based wildlife species classification using camera trap images. The project explores transfer learning, data augmentation strategies, and performance evaluation on real-world ecological data.

→ `wildlife-image-classification/`

---

### 2. Tear Film Break-Up Detection (FIPA Project)  
**Classical Image Processing · ROI Extraction · Thresholding · Temporal Analysis**

A semi-automatic classical image processing pipeline for tear film break-up detection in fluorescein eye image sequences. Developed as part of the *Fundamentals of Image Processing and Analysis (FIPA)* course, with a strong focus on interpretability and deterministic processing.

→ `FIPA-roi-extraction/`

---

### 3. Image Description & Modeling — Texture Analysis (HLBP)  
**Handcrafted Texture Features · Classical Classification**

*Context:* Academic project (Image Description and Modeling)  
*Type:* Individual project

This project focuses on **texture-based image description** using **Histogram Local Binary Patterns (HLBP)** for medical image tiles.

Key highlights:
- Manual feature extraction (no deep learning)
- Texture descriptors (LBP / HLBP)
- Statistical analysis of texture distributions
- Classical classifier evaluation
- Strong emphasis on interpretability and feature design

→ `IDM-texture-analysis/`

---

### 4. Wavelet-Based Image Denoising  
**Wavelet Transform · Signal Processing · Image Restoration · Quantitative Evaluation**

This project investigates classical **wavelet-based image denoising** techniques across multiple datasets and noise conditions.

Key highlights:
- Evaluated multiple wavelet families (Daubechies, Symlets, Coiflets, etc.)
- Tested under Gaussian, Speckle, and Salt & Pepper noise
- Quantitative comparison using **PSNR, SSIM, and MSE**
- Visual and statistical performance analysis
- Focus on classical, interpretable image restoration methods

→ `Wavelet-denoising-comparison/`

---

### 5. Facial Emotion Recognition — Fine-tuned ResNet-18 on FER-2013  
**PyTorch · ResNet-18 · Transfer Learning · Class Imbalance · FER-2013**

*Context:* Academic project (Human Affective Response)  
*Type:* Individual project

Facial expression recognition using two-phase transfer learning on the FER-2013 dataset (35,887 grayscale images, 7 emotion classes). A ResNet-18 backbone pretrained on ImageNet is first adapted with a frozen backbone, then fully fine-tuned at a lower learning rate to preserve pretrained representations.

Key highlights:
- Two-phase training strategy: head-only (5 epochs) → full fine-tuning (10 epochs)
- Inverse-frequency class weighting to handle severe class imbalance (disgust: 436 samples vs. happy: 7,215)
- **69.3% test accuracy** across 7 emotion categories
- Strong performance on happy (F1=0.881) and surprise (F1=0.813)
- Cosine annealing scheduler and early stopping with best-checkpoint recovery
- Full evaluation: per-class precision/recall/F1, confusion matrix, softmax visualisations

→ `facial-emotion-recognition/`

---

## Technical Skills
- **Languages:** Python, MATLAB  
- **Computer Vision:** OpenCV, classical image processing, ROI extraction, texture analysis  
- **Deep Learning:** PyTorch, CNNs, transfer learning  
- **Image Analysis:** Thresholding, morphology, denoising, temporal analysis  
- **Tools:** Git, Jupyter, NumPy, Matplotlib
