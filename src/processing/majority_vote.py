from rich import print
import numpy as np
import pandas as pd


def majority_voting(predictions_df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Perform majority voting for ensemble classification.
    
    Args:
        predictions_df (pd.DataFrame): DataFrame with 'Id' column and task predictions in columns starting with 'T'
        verbose (bool): Whether to print debugging info
    
    Returns:
        pd.DataFrame: DataFrame with 'Id' and predicted 'Class'
    """
    print("[bold green]Executing Weighted Majority Voting[/bold green]")
    # Identify the task prediction columns (T columns)
    task_columns = [col for col in predictions_df.columns if col.startswith('T')]

    # Drop rows where all T* columns are NaN (make a copy to avoid warnings)
    predictions_df = predictions_df.dropna(subset=task_columns, how='all').copy()

    # Determine the majority class per row (ignoring NaN values)
    predictions_df.loc[:, 'Predicted_Class'] = predictions_df[task_columns].mode(axis=1)[0]

    # Ensure the predicted class column is of integer type (if applicable)
    predictions_df.loc[:, 'Predicted_Class'] = predictions_df['Predicted_Class'].astype('Int64')  # Allows NaN handling
    
    if verbose:
        print("\n[bold]Predictions Data:[/bold]")
        print(predictions_df)

    # Return only necessary columns
    return predictions_df.copy()



