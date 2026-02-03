# Tear Film Break-Up Detection Using Classical Image Processing  
(Fundamentals of Image Processing and Analysis – FIPA Project)

This repository contains the complete implementation of a **classical image processing pipeline** for detecting **tear film break-up (BUT)** in fluorescein eye image sequences.

The project was developed as part of the **Fundamentals of Image Processing and Analysis (FIPA)** course and strictly follows the **methodology, logic, assumptions, and evaluation protocol described in the accompanying academic report**.  
No machine learning or deep learning techniques are used; the emphasis is on **interpretability, reproducibility, and domain-driven image analysis**.

---

## Project Motivation

Tear film break-up is a clinically relevant indicator in dry eye disease assessment. Automated detection of break-up regions in fluorescein images can support objective analysis and reduce subjectivity in manual inspection.

The objective of this project is to design and evaluate a **classical, explainable image processing pipeline** capable of detecting tear film break-up at the **frame level** in eye image sequences.

---

## Scope and Design Philosophy

- Classical image processing only (no ML / DL)
- Deterministic and explainable decisions
- Minimal external dependencies
- Reproducible academic pipeline
- Focus on **understanding the image formation and artefacts**
- Designed for **educational and research purposes**

---

## Dataset Description

The dataset consists of **fluorescein eye image sequences**, organized by subject/sequence.

### Folder Structure

dataset/
├── but1/
│ ├── but1_frameXXXX_breakup0.png
│ ├── but1_frameXXXX_breakup1.png
├── but2/
├── but3/
├── but4/
├── but5/
├── but6/
├── but7/
├── but8/
└── data.txt


Each `butX` folder contains a temporal sequence of frames captured under fluorescein illumination.

---

### Ground Truth Annotation

Ground truth labels are provided in `data.txt`, using a **semicolon-separated format**:

image;breakup
but1/but1_frame1440_breakup0.png;0
but1/but1_frame1470_breakup1.png;1


- `0` → No tear film break-up  
- `1` → Tear film break-up present  

Ground truth is used **only for evaluation**, not during detection or threshold estimation.

---

## Methodology Overview

The pipeline operates **sequence-by-sequence** and processes all frames in temporal order.

---

## Step-by-Step Processing Pipeline

### 1. Preprocessing

- Green channel extraction (fluorescein response)
- Contrast enhancement using CLAHE
- Gaussian smoothing for noise reduction

---

### 2. Iris Detection

The iris is detected using a **multi-stage classical approach**:

- Adaptive thresholding
- Otsu thresholding
- Edge detection (Canny)
- Circular Hough Transform
- Candidate scoring based on:
  - Edge support
  - Circularity
  - Geometric consistency

The highest-scoring circle is selected as the iris region.

---

### 3. Frame Alignment (Enabled)

To reduce temporal instability caused by eye motion:

- The first valid frame is selected as reference
- Subsequent frames are aligned using iris center translation
- Alignment corrects only for translation (no rotation or scaling)

---

### 4. Eyelid and Eyelash Removal (Semi-Automatic)

Upper and lower eyelid regions are removed using **sequence-specific cut parameters**:

python
SEQUENCE_PARAMS = {
  "but1": {"top_cut": 0.60, "bottom_cut": 0.00},
  "but2": {"top_cut": 0.00, "bottom_cut": 0.05},
  ...
}

Upper and lower eyelid regions are removed using manually defined, sequence-specific cropping parameters. These parameters were selected based on visual inspection to minimize eyelash and eyelid occlusion while preserving the tear film region.

**Note on Automation**  
This step makes the pipeline **semi-automatic**. The eyelid removal parameters are not automatically estimated and must be defined once per sequence. This limitation is explicitly acknowledged in the academic report and preserved here to maintain methodological consistency.

---

### 5. Black Threshold Estimation

Tear film break-up appears as **dark regions** within the fluorescein-enhanced tear film. To detect these regions robustly, a **black intensity threshold** is estimated.

- The threshold is computed **only from the first valid frame** of each sequence
- The calculation is restricted to the iris region after eyelid removal
- Low-intensity pixel statistics are analyzed to determine an adaptive threshold
- The estimated threshold remains **fixed for all frames** in the sequence

This design ensures robustness to illumination changes while maintaining temporal consistency.

---

### 6. Break-Up Segmentation

For each frame in the sequence:

- The green channel is extracted
- Pixels darker than the black threshold inside the iris ROI are classified as break-up candidates
- Binary break-up maps are generated
- Morphological opening removes isolated noise
- Small connected components are discarded based on area thresholds

The result is a cleaned, binary break-up mask for each frame.

---

### 7. Break-Up Evolution Curve

To analyze tear film stability over time, a **break-up evolution curve** is computed:

- For each frame, the percentage of break-up pixels is calculated
- This produces a temporal signal across the sequence
- The signal is smoothed using polynomial fitting to reduce frame-to-frame noise

The evolution curve provides a stable representation of tear film degradation over time.

---

### 8. Frame-Level Break-Up Decision

A frame is classified as containing tear film break-up if:

- The smoothed evolution curve exceeds the adaptive break-up threshold

Otherwise, the frame is classified as **no break-up**.

Important characteristics of this decision logic:

- No temporal voting
- No post-hoc smoothing
- Each frame is evaluated independently
- The same logic is applied consistently to all sequences

---

## Evaluation Methodology

Evaluation is performed at the **frame level** by comparing predicted labels against ground truth annotations from `data.txt`.

Metrics are computed both:

- **Per sequence**, and
- **Overall**, across all frames

---

## Metrics Reported

- Confusion Matrix
- Accuracy
- Precision
- Recall
- F1 Score
- Per-sequence performance summary

All metrics follow standard definitions and are computed deterministically.

---

## Final Results

### Overall Performance

- **Total frames evaluated:** 90  
- **Correct predictions:** 74  
- **Overall Accuracy:** 82.22%

### Confusion Matrix (Overall)

- **True Positives (TP):** 27  
- **True Negatives (TN):** 47  
- **False Positives (FP):** 8  
- **False Negatives (FN):** 8  

### Derived Metrics

- **Precision:** 0.771  
- **Recall:** 0.771  
- **F1 Score:** 0.771  

The results reproduced by this repository match those reported in the final academic submission.

---

## Per-Sequence Performance Summary

Performance varies across sequences due to differences in:

- Illumination
- Eyelid occlusion
- Tear film stability patterns

Some sequences achieve perfect detection, while others are more challenging due to severe occlusions or subtle break-up patterns. This variability highlights the importance of robust preprocessing and domain-specific heuristics.

---

## Key Insights

- Classical image processing can achieve competitive performance when domain knowledge is applied
- Temporal analysis significantly improves robustness compared to single-frame decisions
- Eyelid and eyelash occlusion remains the dominant challenge in fluorescein imaging
- The explainability of classical methods is a major advantage over black-box approaches

---

## Limitations

- Eyelid and eyelash removal is **semi-automatic**
- Sequence-specific parameters must be manually defined
- The method assumes relatively stable illumination
- The pipeline is not designed for real-time clinical deployment

These limitations are intentionally preserved to remain faithful to the academic scope of the project.

---

## Ethical and Academic Use Statement

This project was developed strictly for **academic and educational purposes**.

- No clinical decisions should be made using this code
- The system is not intended for diagnostic use
- All results are experimental and research-oriented

---

## Data Privacy Note

- The dataset is treated as anonymized
- Image filenames contain no personally identifiable information
- No patient metadata is stored, inferred, or transmitted
- Users are responsible for ensuring compliance with local data protection regulations when reusing the dataset

---

## Reproducibility

To reproduce the reported results:

1. Place the dataset in the `dataset/` directory
2. Run the notebook cells **in order**
3. Do not modify sequence parameters
4. Do not alter thresholding or evaluation logic

Any deviation from the original pipeline may lead to different results.

---

## Acknowledgment

This project was completed as part of the **Fundamentals of Image Processing and Analysis (FIPA)** course and reflects the academic objectives of the curriculum.

---

## Author Note

This repository intentionally prioritizes **clarity, correctness, and interpretability** over automation and optimization, in alignment with the educational goals of the project.

