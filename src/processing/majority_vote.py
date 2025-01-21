from rich import print
import pandas as pd
import numpy as np


def majority_voting(predictions_df, confidence_df, verbose=False):
    """
    Highly optimized majority voting implementation using NumPy vectorized operations.
    
    Parameters:
    predictions_df (pandas.DataFrame): DataFrame containing prediction columns
    confidence_df (pandas.DataFrame): DataFrame containing confidence scores (not used in current implementation)
    
    Returns:
    pandas.DataFrame: Original DataFrame with additional 'predicted_class' column
    """
    print("[bold green]Executing Majority Voting[/bold green]")
    
    # Extract only prediction columns (no copy, just view)
    X = predictions_df.iloc[:, 1:-1].to_numpy()
    
    # Compute majority vote in a single vectorized operation
    # mean >= 0.5 is equivalent to count of 1s > count of 0s
    # This handles ties (equal counts) by choosing 1
    majority_votes = (X.mean(axis=1) >= 0.5).astype(np.int8)
    
    # Efficient DataFrame update without full copy
    result_df = predictions_df.copy()
    result_df.loc[:, 'predicted_class'] = majority_votes
    
    print("\n[bold]Predictions Data:[/bold]")
    print(result_df)
    
    return result_df


