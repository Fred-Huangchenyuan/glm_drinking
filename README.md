# GLM Model for Drinking Behavior Analysis

This repository contains analysis scripts for the research paper ***A bottom-up septal inhibitory circuit anticipates thirst satiation*** (Xu et al, 2025). The scripts implement a Generalized Linear Model (GLM) to analyze neuronal activity during drinking.

## Scripts

### 1. Model Training (`model_training.R`)
- Implements GLM model optimization using experimental data
- Performs model fitting and parameter estimation

### 2. Statistical Analysis (`R2_beta_calculation.R`)
- Calculates partial R² and standardized effect sizes (β) for each covariate
- Evaluates the relative importance of different predictors

### 3. Neural Activity Prediction (`df_f_prediction.R`)
- Applies the optimized model to predict Δf/f values
- Validates model predictions against experimental data
- 
### 4. Frequency Analysis (`frequency_analysis.py`)
- Analyzes event frequency from CSV data files
- Converts timestamps from nanoseconds to seconds
- Calculates and plots event frequency per second

### 5. K-means Clustering Analysis (`kmeans_clustering3.0.py`)
- Performs cluster analysis on neural activity data
- Implements both automatic and manual K selection
- Generates visualizations including:
  - PCA-based cluster plots
  - Correlation heatmaps
  - Activity pattern heatmaps
  - Cluster activity curves with SEM
- Exports all visualizations as SVG files

### 6. Lick Event Visualization (`lick_visualize.py`)
- Interactive tool for visualizing licking events
- Outputs:
  - Summary statistics for each analyzed segment
  - Visual representation of licking patterns
  - Temporal distribution of licking events

## Usage
Please refer to individual subfolders and scripts for detailed documentation and implementation details.

## Citation
If you use this code in your research, please cite our paper: [Paper citation details]
