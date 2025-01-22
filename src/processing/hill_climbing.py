from rich import print
import pandas as pd
import numpy as np
from typing import List, Tuple
import logging

def hill_climbing_combine(predictions_df: pd.DataFrame, confidence_df: pd.DataFrame = None, verbose: bool = False) -> pd.DataFrame:
    """
    Implement hill climbing-based ensemble method.
    
    Args:
        predictions_df: DataFrame with predictions
        confidence_df: Optional DataFrame with confidence scores
        verbose: Whether to print detailed information
    
    Returns:
        DataFrame with combined predictions
    """
    print("[bold green]Executing Hill Climbing-based Method[/bold green]")
    
    # Get task columns (features)
    task_cols = [col for col in predictions_df.columns if col.startswith('T')]
    
    if verbose:
        print(f"\nAnalyzing {len(task_cols)} tasks...")
    
    # Initialize parameters
    max_iter = 1000
    num_neighbors = 100
    patience = 200
    random_normal_loc = 0
    random_normal_scale = 0.1
    
    # Prepare data
    X = predictions_df[task_cols].values
    num_features = len(task_cols)
    
    # Initialize weights
    weights = np.random.rand(num_features)
    best_weights = weights.copy()
    best_score = 0
    iterations_without_improvement = 0
    
    # Hill climbing optimization
    for iteration in range(max_iter):
        # Generate neighbors
        neighbors = [np.copy(weights) for _ in range(num_neighbors)]
        for i in range(num_neighbors):
            neighbors[i] += np.random.normal(random_normal_loc, random_normal_scale, num_features)
            neighbors[i] = np.clip(neighbors[i], 0, 1)  # Ensure weights are between 0 and 1
        
        # Evaluate neighbors
        improved = False
        for neighbor in neighbors:
            weighted_sum = np.dot(X, neighbor)
            predictions = (weighted_sum > 0.5).astype(int)
            
            # Calculate score (using agreement between predictions as metric)
            score = np.mean([np.mean(predictions == X[:, i]) for i in range(num_features)])
            
            if score > best_score:
                best_score = score
                best_weights = neighbor.copy()
                improved = True
                iterations_without_improvement = 0
        
        if not improved:
            iterations_without_improvement += 1
            
        if iterations_without_improvement >= patience:
            if verbose:
                print("\nEarly stopping triggered")
            break
            
        if verbose and iteration % 100 == 0:
            print(f"\nIteration {iteration}: Best score so far: {best_score:.3f}")
    
    # Generate final predictions using best weights
    final_weighted_sum = np.dot(X, best_weights)
    final_predictions = (final_weighted_sum > 0.5).astype(int)
    
    # Create result DataFrame
    result_df = predictions_df.copy()
    result_df['predicted_class'] = final_predictions
    
    if verbose:
        print("\n[bold]Final Predictions:[/bold]")
        print(result_df)
    
    return result_df