from rich import print
import numpy as np
import pandas as pd


def weighted_majority_voting(predictions_df, confidence_df, validation_accuracies_df=None, verbose=False):
    """
    Enhanced weighted majority voting implementation that properly handles both classes (0 and 1)
    while incorporating confidence scores and validation accuracies.
    Missing prediction values (NaN) are ignored in the voting process.
    
    Parameters:
    predictions_df (pandas.DataFrame): DataFrame containing prediction columns.
        (Assumes that the prediction columns are located from the second column to the second-to-last column,
         with the first column typically being an identifier such as 'subject' and the last column reserved 
         for an output if present.)
    confidence_df (pandas.DataFrame): DataFrame containing confidence scores.
        (Assumes that the corresponding confidence columns start with 'Cd1_'.)
    validation_accuracies_df (pandas.DataFrame): DataFrame containing validation accuracies for each task.
        (If provided, the validation accuracy for each task is used to adjust the weight.)
    verbose (bool): If True, prints additional debugging information.
    
    Returns:
    pandas.DataFrame: A copy of the original predictions DataFrame with an additional 'predicted_class' column.
    """
    print("[bold green]Executing Weighted Majority Voting[/bold green]")
    
    # -------------------------------------------------------------------
    # 1. Extract predictions and create a mask for valid (non-NaN) entries.
    #    We assume that predictions are in columns 1 to -1.
    # -------------------------------------------------------------------
    X = predictions_df.iloc[:, 1:-1].to_numpy()  # shape: (n_samples, n_tasks)
    valid_mask = ~np.isnan(X)  # True where predictions are valid
    
    # -------------------------------------------------------------------
    # 2. Get confidence scores.
    #    We assume the confidence columns start with 'Cd1_'.
    # -------------------------------------------------------------------
    conf_cols = [col for col in confidence_df.columns if col.startswith('Cd1_')]
    confidences = confidence_df[conf_cols].to_numpy()  # Expected to match the shape of X
    
    # -------------------------------------------------------------------
    # 3. Prepare weights by optionally incorporating validation accuracies.
    # -------------------------------------------------------------------
    if validation_accuracies_df is not None:
        # For each prediction column (assumed to be in predictions_df.iloc[:, 1:-1]),
        # get its corresponding validation accuracy. The code assumes that by replacing
        # 'T' in the column name you can match a column name in validation_accuracies_df.
        accuracies = []
        for col in predictions_df.columns[1:-1]:
            task_name = col.replace('T', '')  # adjust as needed to match your validation accuracies columns
            if task_name in validation_accuracies_df.columns:
                accuracies.append(validation_accuracies_df[task_name].iloc[0])
            else:
                accuracies.append(1.0)
        # Reshape to (1, n_tasks) so that broadcasting works with confidences.
        accuracies = np.array(accuracies).reshape(1, -1)
        # Combine confidences and accuracies. Taking the square root moderates the effect.
        weights = np.sqrt(confidences * accuracies)
    else:
        weights = confidences
    
    # -------------------------------------------------------------------
    # 4. Calculate weighted votes for each class, ignoring NaN predictions.
    #    For valid predictions, class 1 gets a vote proportional to X and
    #    class 0 gets a vote proportional to (1 - X). For NaN values, both contribute 0.
    # -------------------------------------------------------------------
    class1_votes = np.where(valid_mask, X, 0) * weights  # votes for class 1
    class0_votes = np.where(valid_mask, (1 - X), 0) * weights  # votes for class 0
    
    # Sum the weighted votes for each class for each instance (row)
    sum_class1 = np.sum(class1_votes, axis=1)
    sum_class0 = np.sum(class0_votes, axis=1)
    
    # -------------------------------------------------------------------
    # 5. Decide the predicted class.
    #    If the weighted vote for class 1 is greater than or equal to class 0,
    #    choose class 1 (this serves as the tie-breaker).
    # -------------------------------------------------------------------
    majority_votes = (sum_class1 >= sum_class0).astype(np.int8)
    
    # -------------------------------------------------------------------
    # 6. Update the predictions DataFrame with the predicted class.
    # -------------------------------------------------------------------
    result_df = predictions_df.copy()
    result_df.loc[:, 'predicted_class'] = majority_votes
    
    if verbose:
        print("\n[bold]Vote Distribution:[/bold]")
        for i in range(len(majority_votes)):
            print(f"Instance {i + 1}:")
            print(f"  Class 1 weighted votes: {sum_class1[i]:.3f}")
            print(f"  Class 0 weighted votes: {sum_class0[i]:.3f}")
            print(f"  Final prediction: {majority_votes[i]}")
    
    print("\n[bold]Predictions Data:[/bold]")
    print(result_df)
    
    return result_df
