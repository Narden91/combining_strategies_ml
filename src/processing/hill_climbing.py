from rich import print
import pandas as pd
import numpy as np
from typing import Optional


def hill_climbing_combine(predictions_df: pd.DataFrame, validation_acc: Optional[pd.DataFrame] = None,
                          verbose: bool = False) -> pd.DataFrame:
    """
    Implement hill climbing-based ensemble method using validation accuracies if available.

    Args:
        predictions_df: DataFrame with predictions.
        validation_acc: Optional DataFrame with validation accuracies of models.
        verbose: Whether to print detailed information.

    Returns:
        DataFrame with combined predictions.
    """
    print("[bold green]Executing Hill Climbing-based Method with Validation Accuracies[/bold green]")
    
    if verbose:
        print("\n[bold]Predictions:[/bold]")
        print(predictions_df)
        if validation_acc is not None:
            print("\n[bold]Validation Accuracies:[/bold]")
            print(validation_acc)
    
    # Extract task columns
    task_cols = [col for col in predictions_df.columns if col.startswith('T')]

    if verbose:
        print(f"\nProcessing {len(task_cols)} tasks with hill climbing...")

    # Initialize parameters
    max_iter = 1000
    num_neighbors = 100
    patience = 400
    random_normal_loc = 0.2
    random_normal_scale = 0.5

    # Prepare data
    X = predictions_df[task_cols].values
    num_features = len(task_cols)

    # Use validation accuracies for better initialization
    if validation_acc is not None:
        acc_values = validation_acc.values  # Extract accuracies (skip the ID column)
        acc_values = np.clip(acc_values, 0.5, 1)  # Ensure minimum weight of 0.5
        weights = acc_values / np.sum(acc_values)  # Normalize weights
    else:
        weights = np.random.rand(num_features)  # Default random weights

    best_weights = weights.copy()
    best_score = 0
    iterations_without_improvement = 0

    # **Hill Climbing Optimization**
    for iteration in range(max_iter):
        # Generate neighbors by slightly modifying the weights
        neighbors = [np.copy(weights) for _ in range(num_neighbors)]
        for i in range(num_neighbors):
            neighbors[i] += np.random.normal(random_normal_loc, random_normal_scale, num_features)
            neighbors[i] = np.clip(neighbors[i], 0, 1)  # Ensure weights remain between 0 and 1

        # Evaluate each neighbor
        improved = False
        for neighbor in neighbors:
            weighted_sum = np.dot(X, neighbor)
            predictions = (weighted_sum > 0.5).astype(int)
            
            # **Weighted Score Calculation**
            if validation_acc is not None:
                task_weights = validation_acc.iloc[0, 1:].values
                score = np.average([np.mean(predictions == X[:, i]) for i in range(num_features)], weights=task_weights)
            else:
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
                print("\n[bold yellow]Early stopping triggered![/bold yellow]")
            break

        if verbose and iteration % 100 == 0:
            print(f"\nIteration {iteration}: Best score so far: {best_score:.3f}")

    # **Generate final predictions using best weights**
    final_weighted_sum = np.dot(X, best_weights)
    final_predictions = (final_weighted_sum > 0.5).astype(int)

    # Create final result DataFrame
    result_df = predictions_df.copy()
    result_df['predicted_class'] = final_predictions

    if verbose:
        print("\n[bold]Final Predictions:[/bold]")
        print(result_df)

    return result_df
