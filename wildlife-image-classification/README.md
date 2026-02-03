# Wildlife Species Image Classification from Camera Trap Images

## Overview
This project focuses on multi-class image classification of wildlife species using camera trap images. The objective is to automatically identify animal species captured in natural environments, supporting scalable wildlife monitoring and conservation efforts.

The project was developed as part of a Master’s-level Computer Vision course and is based on a DrivenData wildlife conservation challenge.

## Problem Statement
Camera traps generate large volumes of images, making manual species identification time-consuming and impractical. Automated classification systems must handle real-world challenges such as class imbalance, occlusion, background clutter, and varying lighting conditions, while generalizing to unseen locations.

This project addresses these challenges by designing a robust deep learning pipeline for wildlife species classification.

## Dataset
- DrivenData Wildlife Conservation Camera Trap Dataset  
- Images collected from Taï National Park  
- 16,488 training images from 148 camera trap sites  
- 4,464 test images from 51 unseen sites  
- 8 classes: 7 wildlife species + blank (no animal)  
- No site overlap between training and validation sets  

## Methodology
- Site-aware train/validation split to prevent location-based data leakage  
- Transfer learning using a frozen ResNet50 backbone pretrained on ImageNet  
- Lightweight classification head with Dropout regularization  
- Class imbalance handled using inverse-frequency class weighting  
- Data augmentation applied using Albumentations  
- Early stopping based on validation log loss  

## Model Architecture
- Backbone: ResNet50 (frozen)  
- Classification head: Dropout (p=0.5) + Fully Connected layer  
- Input resolution: 192 × 192  
- Trainable parameters: ~16k  

## Tools & Technologies
- Python  
- PyTorch & Torchvision  
- Albumentations  
- NumPy, Pandas  
- Matplotlib  
- TensorBoard  

## Results
- Best validation log loss: 1.6188  
- Best validation accuracy: ~40%  
- Strong performance on visually prominent species  
- Reduced overfitting through frozen backbone and early stopping  

## Key Insights
- Site-based splitting is critical for realistic evaluation of camera trap data  
- Frozen backbones reduce overfitting to background and location-specific features  
- Rare species and visually subtle classes remain challenging under baseline models  

## Future Improvements
- Partial backbone fine-tuning  
- Stronger data augmentation strategies  
- Test-time augmentation (TTA)  
- Incorporating temporal context from camera trap sequences  
- Semi-supervised or self-supervised pretraining  

## Academic Report
A detailed technical report describing the full methodology, experiments, and analysis is included in this repository.

## Contribution
This project was completed as a group project. My individual contributions included:
- Designing the data pipeline and site-aware train/validation split
- Implementing data augmentation and class weighting
- Model training, evaluation, and analysis
- Writing and organizing significant portions of the technical report

