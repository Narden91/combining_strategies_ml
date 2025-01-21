from rich import print
import numpy as np
import pandas as pd


def weighted_majority_voting(predictions_df, confidence_df, validation_accuracies_df=None, verbose = False):
    """
    Enhanced weighted majority voting implementation that properly handles both classes (0 and 1)
    while incorporating confidence scores and validation accuracies.
    
    Parameters:
    predictions_df (pandas.DataFrame): DataFrame containing prediction columns
    confidence_df (pandas.DataFrame): DataFrame containing confidence scores
    validation_accuracies_df (pandas.DataFrame): DataFrame containing validation accuracies for each task
    
    Returns:
    pandas.DataFrame: Original DataFrame with additional 'predicted_class' column
    """
    print("[bold green]Executing Weighted Majority Voting[/bold green]")
    
    # Extract predictions
    X = predictions_df.iloc[:, 1:-1].to_numpy()
    
    # Get confidence scores
    conf_cols = [col for col in confidence_df.columns if col.startswith('Cd1_')]
    confidences = confidence_df[conf_cols].to_numpy()
    
    # Prepare weights (combining confidences with validation accuracies if available)
    if validation_accuracies_df is not None:
        # Extract validation accuracies and match them to the prediction columns
        accuracies = []
        for col in predictions_df.columns[1:-1]:
            task_name = col.replace('T', '')
            if task_name in validation_accuracies_df.columns:
                accuracies.append(validation_accuracies_df[task_name].iloc[0])
            else:
                accuracies.append(1.0)
        
        accuracies = np.array(accuracies).reshape(1, -1)
        weights = np.sqrt(confidences * accuracies)
    else:
        weights = confidences
    
    # Calculate weighted votes for both classes
    class1_votes = X * weights  # Votes for class 1
    class0_votes = (1 - X) * weights  # Votes for class 0
    
    # Sum votes for each class
    sum_class1 = np.sum(class1_votes, axis=1)
    sum_class0 = np.sum(class0_votes, axis=1)
    
    # Make decision based on which class has more weighted votes
    majority_votes = (sum_class1 > sum_class0).astype(np.int8)
    
    # Update DataFrame efficiently
    result_df = predictions_df.copy()
    result_df.loc[:, 'predicted_class'] = majority_votes
    
    if 'verbose' in globals() and verbose:
        print("\n[bold]Vote Distribution:[/bold]")
        for i in range(len(majority_votes)):
            print(f"Instance {i + 1}:")
            print(f"  Class 1 weighted votes: {sum_class1[i]:.3f}")
            print(f"  Class 0 weighted votes: {sum_class0[i]:.3f}")
            print(f"  Final prediction: {majority_votes[i]}")
    
    print("\n[bold]Predictions Data:[/bold]")
    print(result_df)
    
    return result_df