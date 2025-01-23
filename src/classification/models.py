from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as SklearnRF
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from classification.base import BaseClassifier
from classification.tuning import TunedClassifierMixin, SearchSpace
import warnings


class SVMClassifier(TunedClassifierMixin, BaseClassifier):
    """
    Support Vector Machine classifier implementation with hyperparameter tuning.
    
    Uses a Support Vector Machine with RBF kernel by default, optimizing the following parameters:
    - C: Controls regularization strength (inverse)
    - gamma: Kernel coefficient for RBF
    - kernel: Can be 'rbf' or 'linear'
    
    The classifier includes automatic hyperparameter tuning using Halving Random Search,
    which efficiently explores the parameter space by evaluating more configurations
    on smaller subsets of data first.
    """
    
    def __init__(self, task_id: str, output_dir: Path, random_state: int = 42, verbose: bool = False):
        """
        Initialize SVM classifier.
        
        Args:
            task_id: Identifier for the specific task
            output_dir: Directory for saving outputs
            random_state: Random seed for reproducibility
            verbose: Whether to print detailed information
        """
        super().__init__(task_id, output_dir, random_state)
        self.model = SVC(probability=True, random_state=random_state)
        self.verbose = verbose
    
    def get_classifier_name(self) -> str:
        return "svm"
        
    def _fit_implementation(self, X: np.ndarray, y: pd.Series) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.tune_hyperparameters(X, y, SearchSpace.svm_space, verbose=self.verbose)
        self.model.fit(X, y)
        
    def _predict_implementation(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
        
    def _predict_proba_implementation(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

class RFClassifier(TunedClassifierMixin, BaseClassifier):
    """
    Random Forest classifier implementation with hyperparameter tuning.
    
    Uses a Random Forest ensemble, optimizing the following parameters:
    - n_estimators: Number of trees in the forest (50-300)
    - max_depth: Maximum depth of trees (5-30 or None)
    - min_samples_split: Minimum samples required to split a node (2-10)
    - min_samples_leaf: Minimum samples required at a leaf node (1-4)
    
    Includes automatic hyperparameter tuning using Halving Random Search for
    efficient parameter optimization.
    """
    
    def __init__(self, task_id: str, output_dir: Path, random_state: int = 42, verbose: bool = False):
        """
        Initialize Random Forest classifier.
        
        Args:
            task_id: Identifier for the specific task
            output_dir: Directory for saving outputs
            random_state: Random seed for reproducibility
            verbose: Whether to print detailed information
        """
        super().__init__(task_id, output_dir, random_state)
        self.model = SklearnRF(random_state=random_state)
        self.verbose = verbose
    
    def get_classifier_name(self) -> str:
        return "random_forest"
        
    def _fit_implementation(self, X: np.ndarray, y: pd.Series) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.tune_hyperparameters(X, y, SearchSpace.rf_space, verbose=self.verbose)
        self.model.fit(X, y)
        
    def _predict_implementation(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
        
    def _predict_proba_implementation(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

class NeuralNetworkClassifier(TunedClassifierMixin, BaseClassifier):
    """
    Neural Network classifier implementation with hyperparameter tuning.
    
    Uses a Multi-layer Perceptron classifier, optimizing the following parameters:
    - hidden_layer_sizes: Architecture of hidden layers
    - alpha: L2 regularization parameter
    - learning_rate_init: Initial learning rate
    
    The network architecture is automatically tuned using Halving Random Search,
    testing both single and double hidden layer configurations with various sizes.
    """
    
    def __init__(self, task_id: str, output_dir: Path, random_state: int = 42, verbose: bool = False):
        """
        Initialize Neural Network classifier.
        
        Args:
            task_id: Identifier for the specific task
            output_dir: Directory for saving outputs
            random_state: Random seed for reproducibility
            verbose: Whether to print detailed information
        """
        super().__init__(task_id, output_dir, random_state)
        self.model = MLPClassifier(max_iter=1000, random_state=random_state)
        self.verbose = verbose
    
    def get_classifier_name(self) -> str:
        return "neural_network"
        
    def _fit_implementation(self, X: np.ndarray, y: pd.Series) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.tune_hyperparameters(X, y, SearchSpace.nn_space, verbose=self.verbose)
        self.model.fit(X, y)
        
    def _predict_implementation(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
        
    def _predict_proba_implementation(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

class KNNClassifier(TunedClassifierMixin, BaseClassifier):
    """
    K-Nearest Neighbors classifier implementation with hyperparameter tuning.
    
    Optimizes the following parameters:
    - n_neighbors: Number of neighbors to use (3-10)
    - weights: Weight function ('uniform' or 'distance')
    - p: Power parameter for Minkowski metric (1=Manhattan, 2=Euclidean)
    
    Uses Halving Random Search to efficiently find the optimal combination
    of parameters for the specific dataset.
    """
    
    def __init__(self, task_id: str, output_dir: Path, random_state: int = 42, verbose: bool = False):
        """
        Initialize KNN classifier.
        
        Args:
            task_id: Identifier for the specific task
            output_dir: Directory for saving outputs
            random_state: Random seed for reproducibility
            verbose: Whether to print detailed information
        """
        super().__init__(task_id, output_dir, random_state)
        self.model = KNeighborsClassifier()
        self.verbose = verbose
    
    def get_classifier_name(self) -> str:
        return "knn"
        
    def _fit_implementation(self, X: np.ndarray, y: pd.Series) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.tune_hyperparameters(X, y, SearchSpace.knn_space, verbose=self.verbose)
        self.model.fit(X, y)
        
    def _predict_implementation(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
        
    def _predict_proba_implementation(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

def get_classifier(classifier_type: str, task_id: str, output_dir: Path, 
                  random_state: int = 42, verbose: bool = False) -> BaseClassifier:
    """
    Factory function to get classifier instance.
    
    Args:
        classifier_type: Type of classifier ('svm', 'rf', 'nn', or 'knn')
        task_id: Task identifier
        output_dir: Output directory for results
        random_state: Random seed for reproducibility
        verbose: Whether to print detailed information
        
    Returns:
        Initialized classifier instance
        
    Raises:
        ValueError: If classifier_type is not recognized
    """
    classifiers = {
        'svm': SVMClassifier,
        'rf': RFClassifier,
        'nn': NeuralNetworkClassifier,
        'knn': KNNClassifier
    }
    
    if classifier_type not in classifiers:
        raise ValueError(f"Unknown classifier type: {classifier_type}. "
                        f"Available types: {list(classifiers.keys())}")
                        
    return classifiers[classifier_type](task_id, output_dir, random_state, verbose)