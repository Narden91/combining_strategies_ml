from pathlib import Path
import catboost
import numpy as np
import pandas as pd
import sklearn
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as SklearnRF
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from classification.base import BaseClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from classification.tuning import ExtendedSearchSpace, TunedClassifierMixin, SearchSpace
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

class XGBoostClassifier(TunedClassifierMixin, BaseClassifier):
    """XGBoost classifier implementation with hyperparameter tuning."""
    
    def __init__(self, task_id: str, output_dir: Path, random_state: int = 42, verbose: bool = False):
        super().__init__(task_id, output_dir, random_state)
        self.model = XGBClassifier(
            random_state=random_state,
            eval_metric='logloss' 
        )
        self.verbose = verbose
    
    def get_classifier_name(self) -> str:
        return "xgboost"
    
    def _fit_implementation(self, X: np.ndarray, y: pd.Series) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.tune_hyperparameters(X, y, ExtendedSearchSpace.xgb_space, verbose=self.verbose)
        self.model.fit(X, y)
    
    def _predict_implementation(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def _predict_proba_implementation(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
    
    
class CatBoostClassifier(TunedClassifierMixin, BaseClassifier):
    """CatBoost classifier implementation with hyperparameter tuning."""
    
    def __init__(self, task_id: str, output_dir: Path, random_state: int = 42, verbose: bool = False):
        super().__init__(task_id, output_dir, random_state)
        self.model = catboost.CatBoostClassifier(  # Use fully qualified name
            random_state=random_state,
            verbose=False  # Disable logging output
        )
        self.verbose = verbose
    
    def get_classifier_name(self) -> str:
        return "catboost"
    
    def _fit_implementation(self, X: np.ndarray, y: pd.Series) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.tune_hyperparameters(X, y, ExtendedSearchSpace.catboost_space, verbose=self.verbose)
        self.model.fit(X, y)
    
    def _predict_implementation(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def _predict_proba_implementation(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
    

class DecisionTreeClassifier(TunedClassifierMixin, BaseClassifier):
    """Decision Tree classifier implementation with hyperparameter tuning."""
    
    def __init__(self, task_id: str, output_dir: Path, random_state: int = 42, verbose: bool = False):
        super().__init__(task_id, output_dir, random_state)
        self.model = sklearn.tree.DecisionTreeClassifier(random_state=random_state)
        self.verbose = verbose
    
    def get_classifier_name(self) -> str:
        return "decision_tree"
    
    def _fit_implementation(self, X: np.ndarray, y: pd.Series) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.tune_hyperparameters(X, y, ExtendedSearchSpace.dt_space, verbose=self.verbose)
        self.model.fit(X, y)
    
    def _predict_implementation(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def _predict_proba_implementation(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

class AdaBoostClassifier(TunedClassifierMixin, BaseClassifier):
    """AdaBoost classifier implementation with hyperparameter tuning."""
    
    def __init__(self, task_id: str, output_dir: Path, random_state: int = 42, verbose: bool = False):
        super().__init__(task_id, output_dir, random_state)
        # Use the fully qualified scikit-learn AdaBoost classifier
        self.model = sklearn.ensemble.AdaBoostClassifier(random_state=random_state)
        self.verbose = verbose
    
    def get_classifier_name(self) -> str:
        return "adaboost"
    
    def _fit_implementation(self, X: np.ndarray, y: pd.Series) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.tune_hyperparameters(X, y, ExtendedSearchSpace.adaboost_space, verbose=self.verbose)
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
        classifier_type: Type of classifier ('svm', 'rf', 'nn', 'knn', 'xgb', 'catboost', 'dt', 'ada')
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
        'knn': KNNClassifier,
        'xgb': XGBoostClassifier,
        'catboost': CatBoostClassifier,
        'dt': DecisionTreeClassifier,
        'ada': AdaBoostClassifier
    }
    
    if classifier_type not in classifiers:
        raise ValueError(f"Unknown classifier type: {classifier_type}. "
                        f"Available types: {list(classifiers.keys())}")
                        
    return classifiers[classifier_type](task_id, output_dir, random_state, verbose)