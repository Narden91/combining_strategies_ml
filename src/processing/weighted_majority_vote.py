from rich import print
import numpy as np
import pandas as pd


def weighted_majority_voting(predictions_df, confidence_df):
    """
    Weighted majority voting implementation that incorporates confidence scores.
    Uses the confidence values as weights when predictions are tied.
    
    Parameters:
    predictions_df (pandas.DataFrame): DataFrame containing prediction columns
    confidence_df (pandas.DataFrame): DataFrame containing confidence scores
    
    Returns:
    pandas.DataFrame: Original DataFrame with additional 'predicted_class' column
    """
    print("[bold green]Executing Weighted Majority Voting[/bold green]")
    
    # Extract predictions
    X = predictions_df.iloc[:, 1:-1].to_numpy()
    
    # Get confidence scores for class 1 predictions (Cd1_*)
    conf_cols = [col for col in confidence_df.columns if col.startswith('Cd1_')]
    confidences = confidence_df[conf_cols].to_numpy()
    
    # Calculate weighted votes
    weighted_votes = X * confidences
    
    # Sum of weights for each class
    vote_sums = np.sum(weighted_votes, axis=1)
    possible_sums = np.sum(confidences, axis=1)
    
    # Make decision based on weighted majority
    majority_votes = (vote_sums >= (possible_sums / 2)).astype(np.int8)
    
    # Update DataFrame efficiently
    result_df = predictions_df.copy()
    result_df.loc[:, 'predicted_class'] = majority_votes
    
    print("\n[bold]Predictions Data:[/bold]")
    print(result_df)