# Tear Film Break-Up Analysis using Computer Vision

## Overview
This project focuses on the development of an automatic computer vision system for tear film break-up (BUT) analysis, a clinical test used in the diagnosis of Dry Eye Syndrome (DES).

The goal is to process sequences of eye images captured during a BUT test, automatically extract the region of interest (ROI), and detect tear film break-up events by segmenting dark regions on the corneal surface.

This project was developed as part of the course *Fundamentals of Image Processing and Analysis*.

## Problem Context
Dry Eye Syndrome affects a significant portion of the population and impacts quality of life. In clinical practice, the Break-Up Time (BUT) test measures the time between a blink and the first appearance of dry spots on the cornea after fluorescein instillation.

Manual assessment is subjective and time-consuming. This project aims to automate the analysis using classical computer vision techniques.

## Dataset
- Sequences of eye images acquired during BUT tests
- Each sequence corresponds to a video captured between blinks
- Ground truth provided per frame indicating tear film break-up
- **Note:** The dataset is restricted and cannot be shared publicly

## Methodology

### 1. Region of Interest (ROI) Extraction
- Detection of iris position and size
- Alignment of frames within the same sequence
- ROI refinement to exclude eyelids and eyelashes
- Cropping focused on the corneal region

### 2. Tear Film Break-Up Analysis
- Conversion to suitable color space for contrast enhancement
- Segmentation of dark regions inside the ROI
- Thresholding based on segmented area size
- Identification of break-up frames within the sequence

### 3. Break-Up Time Estimation
- Analysis of frame sequence to determine the first break-up occurrence
- Comparison with ground truth annotations

## Tools & Technologies
- Python
- OpenCV
- scikit-image
- NumPy
- Matplotlib

## Results
- Successful extraction of a stable ROI across frames
- Effective segmentation of tear film break-up regions
- Automated identification of break-up frames aligned with ground truth
- Visual analysis used to validate segmentation quality and robustness

## Key Insights
- ROI alignment is critical for consistent analysis across video frames
- Illumination variation and reflections significantly affect segmentation
- Classical image processing techniques remain effective for structured medical imaging problems

## Ethical & Data Usage Note
Due to dataset usage restrictions, raw images and videos are not included in this repository. Only code, algorithms, and derived visualizations are provided.

## Academic Context
This project was completed following official course specifications and evaluation criteria, including ROI extraction accuracy, break-up detection performance, and technical reporting.


