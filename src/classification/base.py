from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from rich import print
import json
import os

class BaseClassifier(ABC):
    """Abstract base class for all classifiers."""
    
    def __init__(self, task_id: str, output_dir: Path, random_state: int = 42):
        """
        Initialize the classifier.
        
        Args:
            task_id: Identifier for the specific task
            output_dir: Base directory for saving outputs
            random_state: Random seed for reproducibility
        """
        self.task_id = task_id
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self._is_fitted = False
        
        # Setup output directory
        self.output_dir = output_dir / self.get_classifier_name() / f"task_{task_id}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    @abstractmethod
    def get_classifier_name(self) -> str:
        """Get the name of the classifier for file organization."""
        pass
        
    def preprocess_data(self, X: pd.DataFrame) -> np.ndarray:
        """
        Preprocess the input data.
        
        Args:
            X: Input features
            
        Returns:
            Preprocessed features as numpy array
        """
        return self.scaler.fit_transform(X)
        
    def save_run_results(self, run_id: int, metrics: Dict, 
                        predictions: np.ndarray, probabilities: np.ndarray,
                        true_labels: np.ndarray) -> None:
        """
        Save results from a single run.
        
        Args:
            run_id: Identifier for the specific run
            metrics: Dictionary of performance metrics
            predictions: Array of predictions
            probabilities: Array of prediction probabilities
            true_labels: Array of true labels
        """
        run_dir = self.output_dir / f"run_{run_id}"
        run_dir.mkdir(exist_ok=True)
        
        # Save metrics
        with open(run_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Save predictions and probabilities
        np.save(run_dir / "predictions.npy", predictions)
        np.save(run_dir / "probabilities.npy", probabilities)
        np.save(run_dir / "true_labels.npy", true_labels)
        
        # Save as CSV for easier viewing
        pd.DataFrame({
            'true_label': true_labels,
            'predicted_label': predictions,
            'probability_class_1': probabilities[:, 1]
        }).to_csv(run_dir / "results.csv", index=False)
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseClassifier':
        """
        Fit the classifier to the data.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Self for method chaining
        """
        X_processed = self.preprocess_data(X)
        self._fit_implementation(X_processed, y)
        self._is_fitted = True
        return self
        
    @abstractmethod
    def _fit_implementation(self, X: np.ndarray, y: pd.Series) -> None:
        """Implementation of the specific fitting logic."""
        pass
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of predictions
        """
        if not self._is_fitted:
            raise RuntimeError("Classifier must be fitted before making predictions")
            
        X_processed = self.scaler.transform(X)
        return self._predict_implementation(X_processed)
        
    @abstractmethod
    def _predict_implementation(self, X: np.ndarray) -> np.ndarray:
        """Implementation of the specific prediction logic."""
        pass
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of prediction probabilities
        """
        if not self._is_fitted:
            raise RuntimeError("Classifier must be fitted before getting probabilities")
            
        X_processed = self.scaler.transform(X)
        return self._predict_proba_implementation(X_processed)
        
    @abstractmethod
    def _predict_proba_implementation(self, X: np.ndarray) -> np.ndarray:
        """Implementation of the specific probability prediction logic."""
        pass

class ClassificationManager:
    """Manager class to handle the classification process."""
    
    def __init__(self, data_path: Path, output_base: Path, 
                 n_runs: int = 30, base_seed: int = 42):
        """
        Initialize the classification manager.
        
        Args:
            data_path: Path to the feature vector files
            output_base: Base path for saving outputs
            n_runs: Number of runs with different seeds
            base_seed: Base random seed
        """
        self.data_path = data_path
        self.output_base = output_base
        self.n_runs = n_runs
        self.base_seed = base_seed
        
    def load_task_data(self, task_id: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load data for a specific task.
        
        Args:
            task_id: Identifier for the specific task
            
        Returns:
            Tuple of (features, labels)
        """
        file_path = self.data_path / f"feature_vector_{task_id}.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"No data file found for task {task_id}")
            
        df = pd.read_csv(file_path)
        X = df.drop('class', axis=1)
        y = df['class']
        return X, y
        
    def stratified_split(self, X: pd.DataFrame, y: pd.Series, 
                        test_size: float, random_state: int) -> Tuple:
        """
        Perform stratified split with handling for small/imbalanced datasets.
        
        Args:
            X: Features
            y: Labels
            test_size: Proportion for test set
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Check class distribution
        class_counts = y.value_counts()
        min_class_size = class_counts.min()
        
        # Calculate minimum samples needed for test set
        min_test_samples = int(min_class_size * test_size)
        
        if min_test_samples < 1:
            # If we can't maintain stratification, fall back to shuffle split
            # but warn about it
            print(f"[yellow]Warning: Insufficient samples in minority class "
                  f"(size={min_class_size}) for stratified split with "
                  f"test_size={test_size}. Using shuffled split instead.[/yellow]")
            return train_test_split(X, y, test_size=test_size, 
                                  random_state=random_state)
        
        # Use stratified split if we have enough samples
        return train_test_split(X, y, test_size=test_size, 
                              stratify=y, random_state=random_state)
        
    def train_classifier(self, classifier: BaseClassifier, X: pd.DataFrame, 
                        y: pd.Series, test_size: float = 0.2) -> Dict:
        """
        Train a classifier with multiple runs and return aggregated results.
        
        Args:
            classifier: Classifier instance to train
            X: Training features
            y: Training labels
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary of aggregated performance metrics
        """
        aggregated_metrics = []
        final_predictions = []
        final_probabilities = []
        final_true_labels = []
        
        for run in range(self.n_runs):
            # Create new random seed for this run
            run_seed = self.base_seed + run
            
            # Split data using appropriate strategy
            X_train, X_test, y_train, y_test = self.stratified_split(
                X, y, test_size, run_seed
            )
            
            # Train the classifier
            classifier.random_state = run_seed
            classifier.fit(X_train, y_train)
            
            # Get predictions and probabilities
            y_pred = classifier.predict(X_test)
            y_prob = classifier.predict_proba(X_test)
            
            # Calculate metrics for this run
            run_metrics = {
                'accuracy': np.mean(y_pred == y_test),
                'run_id': run,
                'seed': run_seed,
                'class_distribution': {
                    'train': dict(pd.Series(y_train).value_counts().items()),
                    'test': dict(pd.Series(y_test).value_counts().items())
                }
            }
            
            # Save results for this run
            classifier.save_run_results(
                run, run_metrics, y_pred, y_prob, y_test.values
            )
            
            aggregated_metrics.append(run_metrics)
            final_predictions.append(y_pred)
            final_probabilities.append(y_prob)
            final_true_labels.append(y_test.values)
        
        # Calculate aggregate statistics
        accuracies = [m['accuracy'] for m in aggregated_metrics]
        aggregate_results = {
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'individual_runs': aggregated_metrics,
            'dataset_statistics': {
                'total_samples': len(y),
                'class_distribution': dict(y.value_counts().items())
            }
        }
        
        # Save aggregate results
        with open(classifier.output_dir / "aggregate_metrics.json", 'w') as f:
            json.dump(aggregate_results, f, indent=4)
        
        return {
            'metrics': aggregate_results,
            'predictions': final_predictions,
            'probabilities': final_probabilities,
            'true_labels': final_true_labels
        }

def get_available_tasks(data_path: Path) -> List[str]:
    """
    Get list of available tasks based on feature vector files.
    
    Args:
        data_path: Path to the feature vector files
        
    Returns:
        List of task IDs
    """
    task_files = list(data_path.glob("feature_vector_*.csv"))
    return [f.stem.split('_')[-1] for f in task_files]