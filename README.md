# Ensemble Learning Combination Methods

This project implements various methods for combining predictions from multiple classifiers in an ensemble learning context. It supports different combination techniques including majority voting, weighted majority voting (with confidence scores and validation accuracies), and ranking-based methods.

## Input Files

The project requires several input files placed in the `data` directory:

### 1. Predictions Data (`predictions_data.csv`)
- Contains binary predictions (0 or 1) from multiple classifiers
- Format:
  - First column: Instance ID
  - Middle columns: Predictions from each classifier (T1, T2, T3, etc.)
  - Last column: Actual class (if available)
- Example:
```
ID,T1,T2,T3,...,actual_class
1,0,1,1,...,1
2,1,0,1,...,0
```

### 2. Confidence Degrees (`confidence_degrees.csv`)
- Contains confidence scores for predictions from each classifier
- Values range between 0 and 1
- Format: One column per classifier (Cd1_T1, Cd1_T2, etc.)
- Example:
```
Cd1_T1,Cd1_T2,Cd1_T3,...
0.8,0.6,0.9,...
0.7,0.85,0.75,...
```

### 3. Validation Accuracies (`validation_accuracies.csv`)
- Contains historical validation accuracies for each classifier
- Values range between 0 and 1
- Format: Single row CSV with classifier names as headers and their corresponding accuracies
- Example:
```
T1,T2,T3,T4,...
0.612,0.785,0.72,0.68,...
```

## Combining Methods

### 1. Majority Voting (MV)
- Simple majority voting where each classifier gets equal weight
- Each classifier votes for class 0 or 1
- Final prediction is the class with the most votes
- Usage: Set `combining_technique: MV` in config

### 2. Weighted Majority Voting (WMV)
- Advanced voting that considers:
  - Confidence degrees for each prediction
  - Historical validation accuracy of each classifier
- Weights are computed as: √(confidence × validation_accuracy)
- Handles both classes (0 and 1) properly by calculating weighted votes for each class
- Usage: Set `combining_technique: WMV` in config

### 3. Ranking-based Methods

#### 3.1 Basic Ranking (RK)
The basic ranking method creates a matrix of common prediction errors between classifier pairs. It calculates diversity by examining how often classifiers make different mistakes, then combines this with validation accuracy using a fixed 50-50 weighting. The final ensemble is selected by choosing the top-performing classifiers based on their combined diversity-accuracy scores. The method uses simple majority voting for final predictions, making it computationally efficient but less adaptable to different dataset characteristics.
- Usage: Set `combining_technique: RK` in config

#### 3.2 Entropy-Based Ranking (ERK)
This advanced method uses multiple diversity metrics (disagreement, correlation, Q-statistic, and double-fault) with weights that adapt based on prediction entropy. Higher entropy leads to more weight on complex measures, while lower entropy favors simpler ones. It employs a greedy selection algorithm that starts with the most confident classifier and iteratively adds members based on their marginal contribution to both diversity and confidence. The final prediction uses weighted voting based on classifier confidence scores.
- Usage: Set `combining_technique: ERK` in config

## Configuration

Configuration is managed through a YAML file with the following structure:
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
  feature_vector:
    folder: Feature_vector_file
  validation_acc:
    filename: validation_accuracies.csv
    encoding: utf-8
    separator: ","

settings:
  verbose: true
  combining_technique: WMV  # Options: MV, WMV, RK, ERK
```

## Key Components

### DataLoader
- Handles loading and preprocessing of all input files
- Manages file encodings and separators
- Loads validation accuracy data from single-row CSV format

### Combining Methods
- `majority_voting`: Implements simple majority voting
- `weighted_majority_voting`: Implements weighted voting with confidence and validation accuracy
- `ranking`: Implements ranking-based combination

### Utilities
- `ConsolePrinter`: Handles formatted console output
- Rich progress bars and status indicators
- Error handling and reporting

## Output

The system provides:
- Final combined predictions
- Detailed voting information (in verbose mode)
- Progress and status updates
- Error messages and warnings when applicable

## Error Handling

The system handles various error cases:
- Missing input files
- Invalid data formats
- Configuration errors
- Runtime errors with informative messages


| **Aspect**                | **Enhanced Entropy Ranking**                          | **Normal Ranking**                        |
|---------------------------|------------------------------------------------------|-------------------------------------------|
| **Task Selection Method** | Greedy algorithm based on diversity + confidence     | Fixed rule (top 50%, min 3, max 10)       |
| **Diversity Metrics**     | Multiple metrics (disagreement, correlation, kappa, double-fault) | Simple disagreement count                |
| **Weights**               | Dynamic weights based on entropy and data characteristics | Fixed weights (diversity_weight = 0.5)    |
| **Task Selection**        | Adaptive ensemble size based on score distribution   | Limited to top-performing tasks           |
| **Flexibility**           | Highly flexible, adapts to data and score differences | Less flexible, follows fixed rules        |
| **Confidence Handling**   | Considers both mean confidence and stability         | Uses mean confidence only                 |
| **Ensemble Size**         | Adaptive size based on score differences (e.g., 3-7) | Fixed size (top 50%, min 3, max 10)       |
| **Prediction Method**     | Weighted voting based on confidence and stability    | Simple majority voting                    |