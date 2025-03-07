import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, matthews_corrcoef

def calculate_metrics(true_labels, predictions):
    """
    Calculate classification metrics for true labels and binary predictions.
    """
    true_labels = np.array(true_labels).astype(int)
    predictions = np.array(predictions).astype(int)
    
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    sensitivity = recall_score(true_labels, predictions, zero_division=0)  # Recall = Sensitivity
    f1 = f1_score(true_labels, predictions, zero_division=0)
    mcc = matthews_corrcoef(true_labels, predictions)
    
    # Specificity from confusion matrix
    tn, fp, fn, tp = confusion_matrix(true_labels, predictions, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "F1 Score": f1,
        "MCC": mcc
    }


def main():
    # Set the CSV path directly
    csv_path = r"C:\Users\Emanuele\Desktop\offline_ResNet50.csv" 
    
    # Read the CSV
    try:
        df = pd.read_csv(csv_path, delimiter=",")
    except FileNotFoundError:
        print(f"Error: File '{csv_path}' not found.")
        return
    
    # Check for 'class' column and valid labels
    if "Class" not in df.columns:
        print("Error: 'Class' column not found.")
        return
    true_labels = df["Class"].values
    if not np.all(np.isin(true_labels, [0, 1])):
        print("Error: 'class' column has values other than 0 and 1.")
        return
    
    # Define task columns (TASK_01 to TASK_25)
    task_columns = [f"TASK_{i:02d}" for i in range(1, 26)]
    missing_tasks = [task for task in task_columns if task not in df.columns]
    if missing_tasks:
        print(f"Error: Missing task columns: {missing_tasks}")
        return
    
    # Calculate metrics for each task
    results = []
    for task in task_columns:
        # Convert float predictions to binary integers
        predictions_float = df[task].values
        binary_predictions = (predictions_float >= 0.5).astype(int)
        
        # Calculate metrics
        metrics = calculate_metrics(true_labels, binary_predictions)
        results.append({"Task": task, **metrics})
    
    # Create and format the results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df[["Task", "Accuracy", "Precision", "Sensitivity", "Specificity", "F1 Score", "MCC"]]
    
    # Display results
    print("\nClassification Metrics for Each Task:")
    print(results_df)
    
    # Save to CSV
    results_df.to_csv("classification_metrics.csv", index=False)
    print("\nResults saved to 'classification_metrics.csv'")

if __name__ == "__main__":
    main()