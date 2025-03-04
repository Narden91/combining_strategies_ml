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
    
    def __init__(self, task_id: str, random_state: int = 42):
        """
        Initialize the classifier.
        
        Args:
            task_id: Identifier for the specific task
            random_state: Random seed for reproducibility
        """
        self.task_id = task_id
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self._is_fitted = False
        
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
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseClassifier':
        """
        Fit the classifier to the data.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Self for method chaining
        """
        # Pop the Id column if present
        if "Id" in X.columns:
            id_col = X.pop("Id")
        else:
            id_col = None
        
        # Preprocess the remaining features
        X_processed_array = self.preprocess_data(X)
        
        # Convert processed array back to a DataFrame
        X_processed = pd.DataFrame(X_processed_array, columns=X.columns, index=X.index)
        
        # Reinsert the Id column at the beginning if it existed
        if id_col is not None:
            X_processed.insert(0, "Id", id_col)
        
        self._fit_implementation(X_processed, y)
        self._is_fitted = True
        return self
        
    @abstractmethod
    def _fit_implementation(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Ensure training uses a DataFrame with feature names."""
        X = pd.DataFrame(X, columns=X.columns)  # Force DataFrame
        self.model.fit(X, y)
        
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
    def _predict_implementation(self, X: pd.DataFrame) -> np.ndarray:
        """Ensure inference also uses a DataFrame with feature names."""
        X = pd.DataFrame(X, columns=self.model.feature_names_in_)  # Force column consistency
        return self.model.predict(X)
        
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
        """Ensure probability predictions use correct feature names."""
        X = pd.DataFrame(X, columns=self.model.feature_names_in_)  # Ensure matching columns
        return self.model.predict_proba(X)

class ClassificationManager:
    """Manager class to handle the classification process."""
    
    def __init__(self, data_path: Path, 
                 n_runs: int = 30, base_seed: int = 42):
        """
        Initialize the classification manager.
        
        Args:
            data_path: Path to the feature vector files
            n_runs: Number of runs with different seeds
            base_seed: Base random seed
        """
        self.data_path = data_path
        self.n_runs = n_runs
        self.base_seed = base_seed
                
    def load_task_data(self, task_id: str) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
        """
        Load data for a specific task, returning features, labels, and the ID column (if present).

        Args:
            task_id: Identifier for the specific task

        Returns:
            Tuple of (features, labels, id_series)
        """
        file_path = self.data_path / f"{task_id}.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"No data file found for task {task_id}")

        df = pd.read_csv(file_path)

        # Identify the correct class column (e.g., 'class', 'Class', 'label', 'Label')
        possible_class_columns = [col for col in df.columns if col.lower() in ["class", "label"]]
        if not possible_class_columns:
            raise ValueError(
                f"No class column found in {file_path}. Expected 'class', 'Class', 'Label', or 'label'."
            )

        class_column = possible_class_columns[0]
        y = df[class_column].copy()
        X = df.drop(columns=[class_column])  # Drop the identified class column from features

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
            print(f"[yellow]Warning: Insufficient samples in minority class "
                  f"(size={min_class_size}) for stratified split with "
                  f"test_size={test_size}. Using shuffled split instead.[/yellow]")
            return train_test_split(X, y, test_size=test_size, 
                                  random_state=random_state)
        
        # Use stratified split if we have enough samples
        return train_test_split(X, y, test_size=test_size, 
                              stratify=y, random_state=random_state)
    
    
    def train_classifier(self, classifier: BaseClassifier, X: pd.DataFrame, 
                     y: pd.Series, test_size: float, seed: int) -> Dict:
        """
        Train a classifier with a given seed and return results.
        
        Args:
            classifier: Classifier instance to train
            X: Training features
            y: Training labels
            id_column: ID column for the dataset
            test_size: Proportion of data to use for testing
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary of performance metrics and predictions
        """
        # Ensure column names are consistent
        X.columns = X.columns.str.strip()
        X_train, X_test, y_train, y_test = self.stratified_split(X, y, test_size, seed)
        
        # Identify the ID column (e.g., 'id', 'Id', etc.) and store it separately
        possible_id_columns = [col for col in X_train.columns if "id" in col.lower()]
        id_col = None
        if possible_id_columns:
            id_col_name = possible_id_columns[0]
            id_col = X_test[id_col_name].copy()  # Keep a copy
            X_train.drop(columns=possible_id_columns, inplace=True)
            X_test.drop(columns=possible_id_columns, inplace=True)

        # Re-align column order in test set with train set
        X_test = X_test[X_train.columns]

        classifier.random_state = seed
        classifier.fit(X_train, y_train)
        
        y_pred = classifier.predict(X_test)
        
        try:
            y_proba = classifier.predict_proba(X_test)
            if y_proba.ndim == 1:
                y_proba = np.column_stack((1 - y_proba, y_proba))  # Convert to (n_samples, 2)
        except AttributeError:
            # Some classifiers (like SVM with linear kernel) don't support predict_proba
            y_proba = np.zeros((len(y_pred), 2))  # Placeholder
            y_proba[:, 1] = y_pred  # Assign predicted values
            y_proba[:, 0] = 1 - y_pred  # Invert to simulate probabilities

        return classifier, y_pred, y_proba, id_col, y_test

def get_available_tasks(data_path: Path) -> List[str]:
    """
    Get list of available tasks based on feature vector files.
    
    Args:
        data_path: Path to the feature vector files
        
    Returns:
        List of task IDs
    """
    task_files = list(data_path.glob("*.csv"))
    return [f.stem.split('_')[0] for f in task_files]