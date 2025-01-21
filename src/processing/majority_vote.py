from rich import print
import pandas as pd
import numpy as np


def majority_voting(predictions_df, confidence_df):
    """
    Performs efficient majority voting on prediction columns using vectorized operations.
    
    Parameters:
    predictions_df (pandas.DataFrame): DataFrame containing prediction columns
    confidence_df (pandas.DataFrame): DataFrame containing confidence scores
    
    Returns:
    pandas.DataFrame: Original DataFrame with an additional 'predicted_class' column
    """
    print("[bold green]Executing Majority Voting[/bold green]")
    
    # Get prediction columns (all except first and last)
    pred_cols = predictions_df.columns[1:-1]
    
    # Convert predictions to numpy array for faster computation
    predictions_array = predictions_df[pred_cols].values
    
    # Calculate row-wise counts of 0s and 1s using vectorized operations
    counts_1 = np.sum(predictions_array == 1, axis=1)
    counts_0 = predictions_array.shape[1] - counts_1  
    
    # Determine majority vote using vectorized comparison
    # If counts are equal (tie), it will be True (1), matching original behavior
    majority_votes = (counts_1 >= counts_0).astype(int)
    
    # Create result DataFrame
    result_df = predictions_df.copy()
    result_df['predicted_class'] = majority_votes
    
    print("\n[bold]Predictions Data:[/bold]")
    print(result_df)
    
    return result_df


