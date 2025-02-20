import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from typing import Dict, Tuple
from rich import print


class EvaluationMetrics:
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Initialize with true and predicted labels."""
        self.y_true = y_true
        self.y_pred = y_pred
        self.confusion_matrix = self._compute_confusion_matrix()
        self.tn, self.fp, self.fn, self.tp = self.confusion_matrix.ravel()
        
    def _compute_confusion_matrix(self) -> np.ndarray:
        """Compute the confusion matrix."""
        return confusion_matrix(self.y_true, self.y_pred)
    
    def get_confusion_matrix(self) -> Dict[str, int]:
        """Return confusion matrix as a dictionary."""
        return {
            'True Negatives': int(self.tn),
            'False Positives': int(self.fp),
            'False Negatives': int(self.fn),
            'True Positives': int(self.tp)
        }
    
    def accuracy(self) -> float:
        """Calculate accuracy."""
        return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)
    
    def sensitivity(self) -> float:
        """Calculate sensitivity (recall)."""
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0
    
    def specificity(self) -> float:
        """Calculate specificity."""
        return self.tn / (self.tn + self.fp) if (self.tn + self.fp) > 0 else 0.0
    
    def precision(self) -> float:
        """Calculate precision."""
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
    
    def f1_score(self) -> float:
        """Calculate F1 score."""
        precision = self.precision()
        recall = self.sensitivity()
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    def mcc(self) -> float:
        """Calculate Matthews Correlation Coefficient."""
        numerator = (self.tp * self.tn) - (self.fp * self.fn)
        denominator = np.sqrt(
            (self.tp + self.fp) * 
            (self.tp + self.fn) * 
            (self.tn + self.fp) * 
            (self.tn + self.fn)
        )
        return numerator / denominator if denominator != 0 else 0.0
    
    def balanced_accuracy(self) -> float:
        """Calculate balanced accuracy."""
        return (self.sensitivity() + self.specificity()) / 2
    
    def get_all_metrics(self) -> Dict[str, float]:
        """Get all evaluation metrics as a dictionary."""
        return {
            'Accuracy': self.accuracy(),
            'Balanced Accuracy': self.balanced_accuracy(),
            'Sensitivity': self.sensitivity(),
            'Specificity': self.specificity(),
            'Precision': self.precision(),
            'F1 Score': self.f1_score(),
            'MCC': self.mcc()
        }

def evaluate_predictions(predictions_df: pd.DataFrame, verbose: bool = False) -> Tuple[Dict[str, int], Dict[str, float]]:
    """
    Evaluate predictions and return confusion matrix and metrics.
    
    Args:
        predictions_df: DataFrame with 'class' and 'predicted_class' columns
        verbose: Whether to print results
    
    Returns:
        Tuple containing confusion matrix and metrics dictionaries
    """
    # Possible column names for the ground truth
    possible_true_cols = ["class", "Class", "true_class", "True_Class", "actual_label", "Actual_Label"]
    # Possible column names for the prediction
    possible_pred_cols = ["predicted_class", "Predicted_Class", "prediction", "Prediction", "predicted", "Pred"]

    # Find matching columns in the DataFrame
    found_true_col = next((col for col in possible_true_cols if col in predictions_df.columns), None)
    found_pred_col = next((col for col in possible_pred_cols if col in predictions_df.columns), None)
    
    print(f"Found true column: {found_true_col}")
    print(f"Found pred column: {found_pred_col}")

    print(f"Predictions:\n{predictions_df}")
    
    # Check if there are NaN values in the columns
    if predictions_df[found_true_col].isnull().values.any() or predictions_df[found_pred_col].isnull().values.any():
        # print(f"Predictions found: {predictions_df[found_true_col].to_numpy()}")
        print(f"Predictions found: {predictions_df[found_pred_col].to_numpy()}")
        raise ValueError("NaN values found in the columns. Please handle them before evaluation.")
    
    if not found_true_col or not found_pred_col:
        raise ValueError("Could not find the required columns in the DataFrame.")

    evaluator = EvaluationMetrics(
        predictions_df[found_true_col].values,
        predictions_df[found_pred_col].values
    )
    
    confusion = evaluator.get_confusion_matrix()
    metrics = evaluator.get_all_metrics()
    
    if verbose:
        print("\n[bold]Confusion Matrix:[/bold]")
        print(f"TN: {confusion['True Negatives']}, FP: {confusion['False Positives']}")
        print(f"FN: {confusion['False Negatives']}, TP: {confusion['True Positives']}")
        
        print("\n[bold]Metrics:[/bold]")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.3f}")
    
    return confusion, metrics