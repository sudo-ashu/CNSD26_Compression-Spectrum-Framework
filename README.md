# Compression Spectrum Analysis of Monkey ECoG Signals Across Awake, Sleep, and Anesthetized States

## Overview

This repository contains the complete analysis pipeline for investigating neural complexity across different states of consciousness using monkey electrocorticography (ECoG) recordings from the NeuroTycho dataset.
The pipeline processes raw ECoG recordings, extracts compression-spectrum-derived features, performs statistical analysis, and generates visualizations for comparing Awake, Sleep, and Anesthetized conditions.

---

## Project Objectives

The primary goals of this project are:
- Characterize neural dynamics across different consciousness states.
- Quantify neural complexity using compression spectrum analysis.
- Extract interpretable features from ECoG recordings.
- Compare Awake, Sleep, and Anesthetized states statistically.
- Investigate the suitability of compression-spectrum features as biomarkers of consciousness.

---

## Dataset

Dataset: NeuroTycho Monkey ECoG Dataset
Monkey:
- Chibi (KTMD and Sleep experiments)

Experimental Conditions:
- Awake
- Sleep
- Anesthetized

Sampling Frequency:
- Original: 1000 Hz
- Downsampled: 200 Hz

---

## Analysis Pipeline

### Step 1: Data Loading

Load raw ECoG recordings and associated condition information.

Input:
- ECoG recordings
- Condition labels
- Time annotations

Output:
- Session-wise ECoG data

---

### Step 2: Preprocessing

The preprocessing pipeline consists of:

1. Downsampling
   - 1000 Hz → 200 Hz
2. Re-referencing
3. Filtering
   - Band-pass filtering
4. Artifact inspection
Output:
- Clean ECoG recordings

---

### Step 3: Segmentation

Each recording is divided into non-overlapping bins.

Configuration:
- Bin Duration = 2 seconds
- Sampling Frequency = 200 Hz
- Samples per Bin = 400

Output:
- 2-second ECoG bins

---

### Step 4: Recording Construction

For statistical analysis:
- 10 bins = 1 recording
- 15 recordings per state

States:
- Awake
- Sleep
- Anesthetized

Output:
- Recording-level datasets

---

### Step 5: Compression Spectrum Analysis

Compression spectrum analysis is applied to each ECoG signal.

Generated outputs:
- Compression Ratio
- Scale Formation
- Entropy of Scale Formation
- Iterative Compression Statistics

---

### Step 6: Feature Extraction

The following features are extracted:
1. Effort-To-Compress (ETC)
2. Spectrum Bandwidth
3. Maximum Scale
4. Mean Fluctuation
5. Mean Entropy

Features are computed per recording and aggregated across states.

---

### Step 7: Visualization

The pipeline generates:
- Compression Spectrum
  - Log(Compression Ratio) vs Log(Scale)
- Scale vs Iteration
- Entropy vs Scale
- Histograms
- Boxplots
- Spectral Analysis (FFT)

---

### Step 8: Statistical Analysis

Normality Assessment:
- Shapiro-Wilk Test

Parametric Analysis:
- One-Way ANOVA

Non-Parametric Analysis:
- Kruskal-Wallis Test

Post-Hoc Analysis:
- Tukey HSD
- Mann-Whitney U Test
- Holm Correction

---

## Main Findings

Significant differences were observed across Awake, Sleep, and Anesthetized states.

Features showing statistically significant differences:
- ETC
- Bandwidth
- Maximum Scale
- Fluctuation

The results suggest that compression-spectrum-derived measures capture meaningful changes in neural dynamics associated with different levels of consciousness.

---

To be presented at: [CNSD, 2026 IISER TVM]

## Contact

Ashutosh Rathore
M.S. Computational Engineering,
School of Interdisciplinary Studies
Indian Institute of Technology Madras
Chennai, 600036

Research Topic:
Compression Spectrum Analysis of Monkey ECoG Signals Across
Awake, Sleep, and Anesthetized States

Guide:
Dr. Aditi Kathpalia
Department of Applied Mechanics and Biomedical Engineering
Indian Institute of Technology Madras
Chennai, 600036

- Session-wise clean bins data <br>
https://www.kaggle.com/datasets/simp0la/session-wise-clean-monkey-data

- State-wise clean bins data <br>
https://www.kaggle.com/datasets/simp0la/state-wise-clean-bin-data

- State-wise clean bins data (30-bins each in the state) <br>
https://www.kaggle.com/datasets/simp0la/state-wise-monkey-brain-recording-30-bins-each?select=Sleep_state
