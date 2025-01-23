# Ensemble Learning Combination Methods

This project implements a comprehensive suite of ensemble learning methods for combining predictions from multiple classifiers. It supports various combination techniques including majority voting, weighted majority voting, ranking-based methods, and several metaheuristic optimization approaches.

## Overview

The system provides a flexible framework for:
- Combining predictions from multiple classifiers using various ensemble methods
- Incorporating confidence scores and validation accuracies
- Evaluating ensemble performance using multiple metrics
- Supporting both classification and prediction aggregation workflows

## Input Files

The system requires the following input files in the `data` directory:

### Predictions Data (`predictions_data.csv`)
Contains binary predictions (0 or 1) from multiple classifiers with the following structure:
- First column: Instance ID
- Middle columns: Predictions from each classifier (T1, T2, T3, etc.)
- Last column: True class label
```
ID,T1,T2,T3,...,class
1,0,1,1,...,1
2,1,0,1,...,0
```

### Confidence Degrees (`confidence_degrees.csv`)
Contains confidence scores (0-1) for each classifier's predictions:
```
Cd1_T1,Cd1_T2,Cd1_T3,...
0.8,0.6,0.9,...
0.7,0.85,0.75,...
```

### Validation Accuracies (`validation_accuracies.csv`)
Contains historical validation accuracies for each classifier:
```
T1,T2,T3,T4,...
0.612,0.785,0.72,0.68,...
```

## Combining Methods

The system implements seven different ensemble methods:

### 1. Majority Voting (MV)
- Simple majority voting with equal classifier weights
- Highly optimized implementation using NumPy vectorized operations
- Usage: Set `combining_technique: MV` in config

### 2. Weighted Majority Voting (WMV)
- Enhanced voting incorporating:
  - Confidence scores for each prediction
  - Historical validation accuracies
- Properly handles both class predictions with weighted votes
- Usage: Set `combining_technique: WMV` in config

### 3. Basic Ranking (RK)
- Creates diversity matrix based on prediction disagreements
- Incorporates confidence scores with adaptive weighting
- Selects optimal subset of classifiers
- Usage: Set `combining_technique: RK` in config

### 4. Entropy-Based Ranking (ERK)
- Advanced ranking using multiple diversity metrics:
  - Disagreement measure
  - Correlation measure
  - Kappa statistic
  - Double-fault measure
- Adaptive weighting based on prediction entropy
- Dynamic ensemble size selection
- Usage: Set `combining_technique: ERK` in config

### 5. Hill Climbing (HC)
- Metaheuristic optimization approach
- Iteratively explores neighbor solutions
- Incorporates early stopping with patience
- Usage: Set `combining_technique: HC` in config

### 6. Simulated Annealing (SA)
- Temperature-based optimization
- Accepts suboptimal solutions with decreasing probability
- Balanced exploration and exploitation
- Usage: Set `combining_technique: SA` in config

### 7. Tabu Search (TS)
- Memory-based metaheuristic
- Maintains tabu list to avoid revisiting solutions
- Extended neighborhood exploration
- Usage: Set `combining_technique: TS` in config

## System Architecture

### Pipeline Components

#### EnsemblePipeline
- Manages the overall ensemble learning workflow
- Handles both classification and prediction combination
- Coordinates data loading, processing, and evaluation

#### DataLoader
- Handles loading and validation of input files
- Supports different file encodings and separators
- Manages configuration-based file handling

#### ClassificationManager
- Manages the classification process when enabled
- Supports multiple runs with different random seeds
- Handles stratified data splitting and evaluation

#### Evaluation Metrics
The system calculates comprehensive performance metrics:
- Accuracy and Balanced Accuracy
- Sensitivity and Specificity
- Precision and F1 Score
- Matthews Correlation Coefficient
- Confusion Matrix

### Configuration

Configuration is managed through a YAML file:
```yaml
paths:
  data: ${hydra:runtime.cwd}/data
  output: ${hydra:runtime.cwd}/output

data:
  predictions:
    file_name: predictions_data.csv
    encoding: utf-8
    separator: ","
  confidence:
    file_name: confidence_degrees.csv
    encoding: utf-8
    separator: ","
  validation_acc:
    filename: validation_accuracies.csv
    encoding: utf-8
    separator: ","

settings:
  verbose: true
  combining_technique: HC  # Options: MV, WMV, RK, ERK, HC, SA, TS
```

## Output and Results

The system generates:
- Combined predictions with confidence scores
- Detailed performance metrics and evaluations
- Ensemble member contributions and weights
- Comprehensive logging and progress information

All outputs are saved in the configured output directory with method-specific naming conventions.

## Error Handling and Logging

The system provides:
- Robust error handling for file operations
- Informative error messages and warnings
- Progress tracking and status updates
- Detailed logging of the ensemble process

## Advanced Features

- Support for large-scale classification tasks
- Adaptive parameter selection
- Multiple diversity metrics
- Flexible evaluation framework
- Rich console output with progress indicators

## Requirements

- Python 3.7+
- NumPy
- Pandas
- Scikit-learn
- Rich (for console output)
- Hydra (for configuration management)
- Logging

For detailed implementation information, refer to the individual method documentation in the source code.