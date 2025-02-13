from rich import print
import pandas as pd
import numpy as np


def majority_voting(predictions_df, verbose=False):
    """
    Highly optimized majority voting implementation using NumPy vectorized operations,
    ignoring NaN values. For each row, the vote is computed as the fraction of 1s among 
    non-NaN predictions. If no predictions are available (all values are NaN), the default
    is to choose 1.
    
    Parameters:
    predictions_df (pandas.DataFrame): DataFrame containing the prediction columns.
      (Assumes that the columns to vote on start with 'TASK_'.)
    verbose (bool): If True, prints additional debugging information.
    
    Returns:
    pandas.DataFrame: A copy of the original DataFrame with an additional column 
                      'predicted_class' containing the result of majority voting.
    """
    print("[bold green]Executing Majority Voting[/bold green]")
    
    # Select only the columns corresponding to tasks.
    task_columns = [col for col in predictions_df.columns if col.startswith("TASK_")]
    # Convert the selected columns to a NumPy array.
    X = predictions_df[task_columns].to_numpy()
    
    # Compute the sum of ones ignoring NaN values.
    ones = np.nansum(X, axis=1)
    # Count how many non-NaN predictions are present in each row.
    valid_counts = np.sum(~np.isnan(X), axis=1)
    
    # Compute the voting fraction for each row.
    # For rows with no valid predictions, we set the fraction to 1 (tie-breaker).
    vote_fraction = np.where(valid_counts > 0, ones / valid_counts, 1.0)
    
    # Majority vote: if the fraction is >= 0.5, choose 1; otherwise, 0.
    majority_votes = (vote_fraction >= 0.5).astype(np.int8)
    
    # Create a copy of the DataFrame to add the predicted class.
    result_df = predictions_df.copy()
    result_df['predicted_class'] = majority_votes
    
    if verbose:
        print("\n[bold]Predictions Data:[/bold]")
        print(result_df)
    
    return result_df

# def majority_voting(predictions_df, verbose=False):
#     """
#     Highly optimized majority voting implementation using NumPy vectorized operations.
    
#     Parameters:
#     predictions_df (pandas.DataFrame): DataFrame containing prediction columns
#     confidence_df (pandas.DataFrame): DataFrame containing confidence scores (not used in current implementation)
    
#     Returns:
#     pandas.DataFrame: Original DataFrame with additional 'predicted_class' column
#     """
#     print("[bold green]Executing Majority Voting[/bold green]")
    
#     # Extract only prediction columns (no copy, just view)
#     X = predictions_df.iloc[:, 1:-1].to_numpy()
    
#     # Compute majority vote in a single vectorized operation
#     # mean >= 0.5 is equivalent to count of 1s > count of 0s
#     # This handles ties (equal counts) by choosing 1
#     majority_votes = (X.mean(axis=1) >= 0.5).astype(np.int8)
    
#     # Efficient DataFrame update without full copy
#     result_df = predictions_df.copy()
#     result_df.loc[:, 'predicted_class'] = majority_votes
    
#     print("\n[bold]Predictions Data:[/bold]")
#     print(result_df)
    
#     return result_df


