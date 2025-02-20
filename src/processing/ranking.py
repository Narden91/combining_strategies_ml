from rich import print
import pandas as pd
import numpy as np
from typing import List, Tuple
import logging

def create_common_task_matrix(predictions_df: pd.DataFrame, task_columns: List[str]) -> pd.DataFrame:
    """Create matrix showing common incorrect predictions between tasks."""
    tasks = len(task_columns)
    wrongly_predicted_sets = {}
    
    # Get sets of wrongly predicted instances for each task pair
    for i, col1 in enumerate(task_columns):
        for j, col2 in enumerate(task_columns):
            if i != j:
                # Compare predictions between pairs of tasks
                disagreements = set(
                    predictions_df[
                        predictions_df[col1] != predictions_df[col2]
                    ].index
                )
                wrongly_predicted_sets[(col1, col2)] = disagreements
    
    # Create matrix of common elements
    common_elements_matrix = np.zeros((tasks, tasks))
    for i, col1 in enumerate(task_columns):
        for j, col2 in enumerate(task_columns):
            if i == j:
                common_elements_matrix[i, j] = np.nan
            else:
                common_elements_matrix[i, j] = len(wrongly_predicted_sets[(col1, col2)])
    
    return pd.DataFrame(common_elements_matrix, columns=task_columns, index=task_columns)

def calculate_diversity_score(matrix: pd.DataFrame) -> np.ndarray:
    """Calculate diversity scores for each task."""
    # Remove diagonal and sum rows
    np_matrix = matrix.to_numpy()
    diversity_scores = np.nansum(np_matrix, axis=1)
    
    # Normalize scores
    if np.max(diversity_scores) > 0:
        diversity_scores = diversity_scores / np.max(diversity_scores)
    
    return diversity_scores


def create_task_rankings(
    matrix: pd.DataFrame, 
    accuracy_df: pd.DataFrame = None,
    diversity_weight: float = 0.5
) -> pd.DataFrame:
    """Create task rankings based on diversity and accuracy scores."""
    
    print(f"Accuracy df:\n {accuracy_df}")
    
    # Calculate diversity scores
    diversity_scores = calculate_diversity_score(matrix)
    tasks = matrix.columns.tolist()

    # Handle accuracy scores
    accuracy_scores = np.zeros(len(diversity_scores))
    if accuracy_df is not None:
        accuracy_scores = []
        for task in tasks:
            if task in accuracy_df.columns:
                accuracy_scores.append(accuracy_df[task].values[0])  # Extract accuracy value
            else:
                accuracy_scores.append(0)  # Default if missing
        accuracy_scores = np.array(accuracy_scores)

    # Normalize scores
    def normalize(arr):
        return (arr - np.min(arr)) / np.ptp(arr) if np.ptp(arr) > 0 else arr
    
    diversity_scores = normalize(diversity_scores)
    accuracy_scores = normalize(accuracy_scores)

    # Compute final ranking score
    combined_scores = (
        diversity_weight * diversity_scores + 
        (1 - diversity_weight) * accuracy_scores
    )

    # Create and sort rankings
    rankings_df = pd.DataFrame({
        'Task': tasks,
        'Diversity_Score': diversity_scores,
        'Accuracy_Score': accuracy_scores,
        'Overall_Score': combined_scores
    })

    return rankings_df.sort_values('Overall_Score', ascending=False, ignore_index=True)


def get_ensemble_prediction(predictions_df: pd.DataFrame, task_columns: List[str]) -> pd.Series:
    """Get ensemble prediction using selected tasks."""
    # Use mode for final prediction
    predictions = predictions_df[task_columns].mode(axis=1)
    
    # Handle cases where there might be multiple modes
    if len(predictions.columns) > 1:
        # If there are multiple modes, use the first one
        return predictions[0]
    return predictions

def ranking(predictions_df: pd.DataFrame, accuracy_df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Implement ranking-based ensemble method.
    
    Args:
        predictions_df: DataFrame with predictions
        accuracy_df: DataFrame with confidence scores
        verbose: Whether to print detailed information
    
    Returns:
        DataFrame with combined predictions
    """
    print("[bold green]Executing Ranking-based Method[/bold green]")
    
    # Get task columns
    task_cols = [col for col in predictions_df.columns if col.startswith('T')]
    
    if verbose:
        print(f"\nAnalyzing {len(task_cols)} tasks...")
        print(f"Tasks: {task_cols}")
    
    # Create common elements matrix
    common_matrix = create_common_task_matrix(predictions_df, task_cols)
    
    if verbose:
        print(f"Common matrix:\n {common_matrix}")
        print(f"Confidence df:\n {accuracy_df}")
    
    if accuracy_df.empty:
        raise ValueError("Error: Confidence DataFrame is empty. Check input data.")

    # Create rankings
    rankings = create_task_rankings(common_matrix, accuracy_df)
    
    if verbose:
        print("\n[bold]Task Rankings:[/bold]")
        print(rankings)
    
    # Use top performing tasks for final prediction
    # Use top 50% of tasks, minimum 3, maximum 10
    n_tasks = max(3, min(10, len(task_cols) // 2))
    top_tasks = rankings['Task'].tolist()[:n_tasks]
    
    if verbose:
        print(f"\nUsing top {n_tasks} tasks: {top_tasks}")
    
    # Create final predictions using top tasks
    ensemble_predictions = get_ensemble_prediction(predictions_df, top_tasks)
    
    # Update DataFrame with predictions
    result_df = predictions_df.copy()
    result_df['predicted_class'] = ensemble_predictions

    # Handle NaN values: Fill with the most common class or fallback to 0
    if result_df['predicted_class'].isna().any():
        most_common_class = result_df['predicted_class'].mode().iloc[0] if not result_df['predicted_class'].mode().empty else 0
        result_df['predicted_class'].fillna(most_common_class, inplace=True)

    # Convert to integer
    result_df['predicted_class'] = result_df['predicted_class'].astype(int)

    if verbose:
        print("\n[bold]Final Predictions:[/bold]")
        print(result_df)

    return result_df