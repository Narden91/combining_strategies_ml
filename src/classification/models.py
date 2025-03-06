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
from xgboost import XGBClassifier, XGBRFClassifier
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
    
    def __init__(self, task_id: str, random_state: int = 42, verbose: bool = False):
        super().__init__(task_id, random_state)
        self.model = SVC(probability=True, random_state=random_state)
        self.verbose = verbose
        self.feature_names = None  # Store feature names

    def get_classifier_name(self) -> str:
        return "svm"

    def _fit_implementation(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Ensure feature names are stored during training."""
        X = pd.DataFrame(X, columns=X.columns)  # Ensure DataFrame
        self.feature_names = X.columns  # Store feature names

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.tune_hyperparameters(X, y, SearchSpace.svm_space, verbose=self.verbose)
        self.model.fit(X, y)

    def _predict_implementation(self, X: pd.DataFrame) -> np.ndarray:
        """Ensure inference uses stored feature names."""
        X = pd.DataFrame(X, columns=self.feature_names)  # Restore feature names
        return self.model.predict(X)

    def _predict_proba_implementation(self, X: pd.DataFrame) -> np.ndarray:
        """Ensure probability prediction uses stored feature names."""
        X = pd.DataFrame(X, columns=self.feature_names)  # Restore feature names
        return self.model.predict_proba(X)

class RFClassifier(TunedClassifierMixin, BaseClassifier):
    """
    Random Forest classifier implementation using XGBoost's XGBRFClassifier.
    """

    def __init__(self, task_id: str, random_state: int = 42, verbose: bool = False):
        super().__init__(task_id, random_state)
        self.model = XGBRFClassifier(random_state=random_state, eval_metric='logloss')
        self.verbose = verbose
        self.feature_names = None

    def get_classifier_name(self) -> str:
        return "random_forest"

    def _fit_implementation(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.feature_names = X.columns

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.tune_hyperparameters(X, y, ExtendedSearchSpace.xgb_rf_space, verbose=self.verbose)

        self.model.fit(X, y)

    def _predict_implementation(self, X: pd.DataFrame) -> np.ndarray:
        X = pd.DataFrame(X, columns=self.feature_names)
        return self.model.predict(X)

    def _predict_proba_implementation(self, X: pd.DataFrame) -> np.ndarray:
        X = pd.DataFrame(X, columns=self.feature_names)
        return self.model.predict_proba(X)
    
# class RFClassifier(TunedClassifierMixin, BaseClassifier):
#     """
#     Random Forest classifier implementation with hyperparameter tuning.
    
#     Uses a Random Forest ensemble, optimizing the following parameters:
#     - n_estimators: Number of trees in the forest (50-300)
#     - max_depth: Maximum depth of trees (5-30 or None)
#     - min_samples_split: Minimum samples required to split a node (2-10)
#     - min_samples_leaf: Minimum samples required at a leaf node (1-4)
    
#     Includes automatic hyperparameter tuning using Halving Random Search for
#     efficient parameter optimization.
#     """
    
#     def __init__(self, task_id: str, random_state: int = 42, verbose: bool = False):
#         super().__init__(task_id, random_state)
#         self.model = SklearnRF(random_state=random_state)
#         self.verbose = verbose
#         self.feature_names = None  # Store feature names

#     def get_classifier_name(self) -> str:
#         return "random_forest"

#     def _fit_implementation(self, X: pd.DataFrame, y: pd.Series) -> None:
#         """Ensure feature names are stored during training."""
#         X = pd.DataFrame(X, columns=X.columns)  # Ensure DataFrame
#         self.feature_names = X.columns  # Store feature names

#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
#             self.tune_hyperparameters(X, y, SearchSpace.rf_space, verbose=self.verbose)
#         self.model.fit(X, y)

#     def _predict_implementation(self, X: pd.DataFrame) -> np.ndarray:
#         """Ensure inference uses stored feature names."""
#         X = pd.DataFrame(X, columns=self.feature_names)  # Restore feature names
#         return self.model.predict(X)

#     def _predict_proba_implementation(self, X: pd.DataFrame) -> np.ndarray:
#         """Ensure probability prediction uses stored feature names."""
#         X = pd.DataFrame(X, columns=self.feature_names)  # Restore feature names
#         return self.model.predict_proba(X)

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
    
    def __init__(self, task_id: str, random_state: int = 42, verbose: bool = False):
        super().__init__(task_id, random_state)
        self.model = MLPClassifier(max_iter=1000, random_state=random_state)
        self.verbose = verbose
        self.feature_names = None  # Store feature names

    def get_classifier_name(self) -> str:
        return "neural_network"

    def _fit_implementation(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Ensure feature names are stored during training."""
        X = pd.DataFrame(X, columns=X.columns)  # Ensure DataFrame
        self.feature_names = X.columns  # Store feature names

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.tune_hyperparameters(X, y, SearchSpace.nn_space, verbose=self.verbose)
        self.model.fit(X, y)

    def _predict_implementation(self, X: pd.DataFrame) -> np.ndarray:
        """Ensure inference uses stored feature names."""
        X = pd.DataFrame(X, columns=self.feature_names)  # Restore feature names
        return self.model.predict(X)

    def _predict_proba_implementation(self, X: pd.DataFrame) -> np.ndarray:
        """Ensure probability prediction uses stored feature names."""
        X = pd.DataFrame(X, columns=self.feature_names)  # Restore feature names
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
    
    def __init__(self, task_id: str, random_state: int = 42, verbose: bool = False):
        super().__init__(task_id, random_state)
        self.model = KNeighborsClassifier()
        self.verbose = verbose
        self.feature_names = None  # Store feature names

    def get_classifier_name(self) -> str:
        return "knn"

    def _fit_implementation(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Ensure feature names are stored during training."""
        X = pd.DataFrame(X, columns=X.columns)  # Ensure DataFrame
        self.feature_names = X.columns  # Store feature names

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.tune_hyperparameters(X, y, SearchSpace.knn_space, verbose=self.verbose)
        self.model.fit(X, y)

    def _predict_implementation(self, X: pd.DataFrame) -> np.ndarray:
        """Ensure inference uses stored feature names."""
        X = pd.DataFrame(X, columns=self.feature_names)  # Restore feature names
        return self.model.predict(X)

    def _predict_proba_implementation(self, X: pd.DataFrame) -> np.ndarray:
        """Ensure probability prediction uses stored feature names."""
        X = pd.DataFrame(X, columns=self.feature_names)  # Restore feature names
        return self.model.predict_proba(X)

class XGBoostClassifier(TunedClassifierMixin, BaseClassifier):
    """XGBoost classifier implementation with hyperparameter tuning."""

    def __init__(self, task_id: str, random_state: int = 42, verbose: bool = False):
        super().__init__(task_id, random_state)
        self.model = XGBClassifier(
            random_state=random_state,
            eval_metric='logloss'
        )
        self.verbose = verbose
        self.feature_names = None  # Store feature names

    def get_classifier_name(self) -> str:
        return "xgboost"

    def _fit_implementation(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Ensure feature names are stored during training."""
        X = pd.DataFrame(X, columns=X.columns)  # Ensure DataFrame
        self.feature_names = X.columns  # Store feature names

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.tune_hyperparameters(X, y, ExtendedSearchSpace.xgb_space, verbose=self.verbose)
        self.model.fit(X, y)

    def _predict_implementation(self, X: pd.DataFrame) -> np.ndarray:
        """Ensure inference uses stored feature names."""
        X = pd.DataFrame(X, columns=self.feature_names)  # Restore feature names
        return self.model.predict(X)

    def _predict_proba_implementation(self, X: pd.DataFrame) -> np.ndarray:
        """Ensure probability prediction uses stored feature names."""
        X = pd.DataFrame(X, columns=self.feature_names)  # Restore feature names
        return self.model.predict_proba(X)
    
    
class CatBoostClassifier(TunedClassifierMixin, BaseClassifier):
    """CatBoost classifier implementation with hyperparameter tuning."""

    def __init__(self, task_id: str, random_state: int = 42, verbose: bool = False):
        super().__init__(task_id, random_state)
        # self.model = catboost.CatBoostClassifier(
        #     random_state=random_state,
        #     task_type='GPU', #if catboost.utils.get_gpu_device_count() > 0 else 'CPU',
        #     devices='0',  # Use first GPU
        #     thread_count=-1,  # Use all CPU cores
        #     early_stopping_rounds=20,
        #     od_type='Iter',
        #     bootstrap_type='Poisson',
        #     verbose=False        
        # )
        self.model = catboost.CatBoostClassifier(
            task_type='GPU',
            devices='0',
            iterations=500,                   # Lower if you don't need extreme precision
            depth=6,                          # Max depth between 4-6 usually gives faster training
            learning_rate=0.1,                # Moderate learning rate
            bootstrap_type='Poisson',         # Optimal for GPU training
            subsample=0.8,                    # Slightly smaller sample improves speed
            random_strength=1,                # Regularization to speed up
            early_stopping_rounds=50,         # Stop early to prevent overfitting
            verbose=False,
            random_state=random_state
        )

        self.verbose = verbose
        self.feature_names = None

    def get_classifier_name(self) -> str:
        return "catboost"

    def _fit_implementation(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Ensure feature names are stored during training."""
        X = pd.DataFrame(X, columns=X.columns)  # Ensure DataFrame
        self.feature_names = X.columns  # Store feature names

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # self.tune_hyperparameters(X, y, ExtendedSearchSpace.catboost_space, verbose=self.verbose)
        self.model.fit(X, y)

    def _predict_implementation(self, X: pd.DataFrame) -> np.ndarray:
        """Ensure inference uses stored feature names."""
        X = pd.DataFrame(X, columns=self.feature_names)  # Restore feature names
        return self.model.predict(X)

    def _predict_proba_implementation(self, X: pd.DataFrame) -> np.ndarray:
        """Ensure probability prediction uses stored feature names."""
        X = pd.DataFrame(X, columns=self.feature_names)  # Restore feature names
        return self.model.predict_proba(X)
    

class DecisionTreeClassifier(TunedClassifierMixin, BaseClassifier):
    """Decision Tree classifier implementation with hyperparameter tuning."""

    def __init__(self, task_id: str, random_state: int = 42, verbose: bool = False):
        super().__init__(task_id, random_state)
        self.model = sklearn.tree.DecisionTreeClassifier(random_state=random_state)
        self.verbose = verbose
        self.feature_names = None  # Store feature names

    def get_classifier_name(self) -> str:
        return "decision_tree"

    def _fit_implementation(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Ensure feature names are stored during training."""
        X = pd.DataFrame(X, columns=X.columns)  # Ensure DataFrame
        self.feature_names = X.columns  # Store feature names

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.tune_hyperparameters(X, y, ExtendedSearchSpace.dt_space, verbose=self.verbose)
        self.model.fit(X, y)

    def _predict_implementation(self, X: pd.DataFrame) -> np.ndarray:
        """Ensure inference uses stored feature names."""
        X = pd.DataFrame(X, columns=self.feature_names)  # Restore feature names
        return self.model.predict(X)

    def _predict_proba_implementation(self, X: pd.DataFrame) -> np.ndarray:
        """Ensure probability prediction uses stored feature names."""
        X = pd.DataFrame(X, columns=self.feature_names)  # Restore feature names
        return self.model.predict_proba(X)
    

class AdaBoostClassifier(TunedClassifierMixin, BaseClassifier):
    """AdaBoost classifier implementation with hyperparameter tuning."""
    
    def __init__(self, task_id: str, random_state: int = 42, verbose: bool = False):
        super().__init__(task_id, random_state)
        self.model = sklearn.ensemble.AdaBoostClassifier(random_state=random_state)
        self.verbose = verbose
        self.feature_names = None  # Store feature names for inference

    def get_classifier_name(self) -> str:
        return "adaboost"
    
    def _fit_implementation(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Ensure that feature names are stored during training."""
        X = pd.DataFrame(X, columns=X.columns)  # Ensure DataFrame
        self.feature_names = X.columns  # Store feature names

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.tune_hyperparameters(X, y, ExtendedSearchSpace.adaboost_space, verbose=self.verbose)
        self.model.fit(X, y)

    def _predict_implementation(self, X: pd.DataFrame) -> np.ndarray:
        """Ensure inference uses the correct feature names."""
        X = pd.DataFrame(X, columns=self.feature_names)  # Restore feature names
        return self.model.predict(X)

    def _predict_proba_implementation(self, X: pd.DataFrame) -> np.ndarray:
        """Ensure probability prediction uses correct feature names."""
        X = pd.DataFrame(X, columns=self.feature_names)  # Restore feature names
        return self.model.predict_proba(X)
    
    
def get_classifier(classifier_type: str, task_id: str,
                  random_state: int = 42, verbose: bool = False) -> BaseClassifier:
    """
    Factory function to get classifier instance.
    
    Args:
        classifier_type: Type of classifier ('svm', 'rf', 'nn', 'knn', 'xgb', 'catboost', 'dt', 'ada')
        task_id: Task identifier
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
                        
    return classifiers[classifier_type](task_id, random_state, verbose)