import os
import pandas as pd
import numpy as np
from pathlib import Path

from metrics import evaluate_predictions


def load_prediction_files(base_folder):
    """
    Load 'Predictions.csv' files from all run_X subfolders
    
    Args:
        base_folder (str): Path to the folder containing run_1...run_20 subfolders
        
    Returns:
        list: List of tuples (run_number, dataframe) for each valid prediction file
    """
    base_path = Path(base_folder)
    predictions = []
    
    # Iterate through run_1 to run_20
    for i in range(1, 21):
        run_folder = base_path / f"run_{i}"
        predictions_file = run_folder / "Predictions.csv"
        
        # Check if folder and file exist
        if run_folder.exists() and predictions_file.exists():
            try:
                # Read the CSV file
                df = pd.read_csv(predictions_file)
                print(f"Loaded {predictions_file} with shape {df.shape}")
                predictions.append((i, df))
                
            except Exception as e:
                print(f"Error loading {predictions_file}: {str(e)}")
        else:
            print(f"Skipping {run_folder} - folder or predictions file not found")
    
    return predictions

def perform_majority_vote_row_masked(run_num, df, n):
    """
    Perform majority voting after masking rows to simulate each subject only performing n tasks
    
    Args:
        run_num (int): Run number
        df (pandas.DataFrame): Dataframe with predictions
        n (int): Number of task results to keep unmasked per subject (row)
        
    Returns:
        pandas.DataFrame: Dataframe with majority vote results
    """
    # Identify ground truth column as 'class'
    ground_truth_col = 'Class'
    
    if ground_truth_col not in df.columns:
        print(f"Warning: No 'class' column found in run {run_num}, cannot evaluate")
        return None
    
    # Get all columns that contain T01, T02, ..., T28, T29
    task_columns = [col for col in df.columns if col.startswith('T')]
    
    if not task_columns:
        print(f"Warning: No task columns (T1, T28, etc.) found in run {run_num}")
        return None
    
    print(f"Run {run_num}: Found {len(task_columns)} task columns")
    
    # If n is greater than available tasks, use all tasks
    n_use = min(n, len(task_columns))
    
    # Set a consistent random seed for reproducibility, but different for each run
    np.random.seed(42 + run_num)
    
    # For each subject (row), perform row-wise masking and majority voting
    majority_votes = []
    
    for i, row in df.iterrows():
        # Randomly select n task columns for this subject
        selected_tasks = np.random.choice(task_columns, n_use, replace=False)
        
        # Extract the predictions from these selected tasks
        task_predictions = [row[task] for task in selected_tasks]
        
        # Perform majority voting on the selected tasks
        if task_predictions:
            # Count occurrences of each prediction
            prediction_counts = {}
            for pred in task_predictions:
                if not pd.isna(pred):  # Skip NaN values
                    prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
            
            if prediction_counts:
                # Find the most common prediction(s)
                max_count = max(prediction_counts.values())
                most_common = [pred for pred, count in prediction_counts.items() if count == max_count]
                
                # If there's a tie, pick randomly
                if len(most_common) > 1:
                    majority_vote = np.random.choice(most_common)
                else:
                    majority_vote = most_common[0]
            else:
                # If all predictions were NaN, use a default (e.g., 0)
                majority_vote = 0
        else:
            # If no task predictions found (should not happen if code is correct)
            majority_vote = 0
        
        majority_votes.append(majority_vote)
    
    # Create result dataframe with ground truth and majority vote
    result_df = pd.DataFrame({
        'class': df[ground_truth_col],
        'predicted_class': majority_votes
    })
    
    return result_df

def compute_metrics_per_run(base_folder, n):
    """
    Compute metrics for each run after row-wise masking and majority voting
    
    Args:
        base_folder (str): Path to the folder containing run_X subfolders
        n (int): Number of task results to keep unmasked per subject (row)
        
    Returns:
        pandas.DataFrame: Dataframe with metrics for each run
    """
    # Load prediction files
    predictions = load_prediction_files(base_folder)
    
    if not predictions:
        print("No prediction files found. Exiting.")
        return None
    
    metrics_rows = []
    
    for run_num, df in predictions:
        # Perform majority voting with row-wise masking
        result_df = perform_majority_vote_row_masked(run_num, df, n)
        
        if result_df is not None:
            try:
                # Compute metrics using the metrics.py module
                _, metrics = evaluate_predictions(result_df)
                
                # Add run number to metrics
                metrics['Run'] = run_num
                metrics_rows.append(metrics)
                
                print(f"Run {run_num} metrics computed successfully")
                
            except Exception as e:
                print(f"Error computing metrics for run {run_num}: {str(e)}")
    
    if not metrics_rows:
        print("No metrics could be computed. Check that the data format is correct.")
        return None
    
    # Create a dataframe with all metrics
    metrics_df = pd.DataFrame(metrics_rows)
    
    # Reorder columns to have 'Run' first
    cols = ['Run'] + [col for col in metrics_df.columns if col != 'Run']
    metrics_df = metrics_df[cols]
    
    return metrics_df

def main(folder_path, n):
    """
    Main function to perform the entire process with row-wise masking
    
    Args:
        folder_path (str): Path to the folder containing run_X subfolders
        n (int): Number of task results to keep unmasked per subject (row)
        
    Returns:
        pandas.DataFrame: Dataframe with metrics for each run
    """
    print(f"Starting analysis with {n} tasks per subject...")
    metrics_df = compute_metrics_per_run(folder_path, n)
    
    if metrics_df is not None:
        print("\nMetrics for each run (keeping {} tasks per subject):".format(n))
        print(metrics_df)
        
        # Calculate average metrics across all runs
        avg_metrics = metrics_df.drop(columns=['Run']).mean().to_dict()
        print("\nAverage metrics across all runs:")
        for metric, value in avg_metrics.items():
            print(f"{metric}: {value:.3f}")
        
        # Save results to CSV
        output_file = os.path.join(folder_path, f"row_masked_vote_metrics_n{n}.csv")
        metrics_df.to_csv(output_file, index=False)
        print(f"Metrics saved to {output_file}")
        
        # Save average metrics
        avg_df = pd.DataFrame([avg_metrics])
        avg_df['Tasks_per_subject'] = n
        avg_output_file = os.path.join(folder_path, f"row_masked_avg_metrics_n{n}.csv")
        avg_df.to_csv(avg_output_file, index=False)
        print(f"Average metrics saved to {avg_output_file}")
    
    return metrics_df

if __name__ == "__main__":
    # Specify parameters
    FOLDER_PATH = r"C:\Users\Emanuele\Desktop\rf"  
    
    for n in range(3, 15, 2):
        metrics_df = main(FOLDER_PATH, n)