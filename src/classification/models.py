from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as SklearnRF
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from classification.base import BaseClassifier

class SVMClassifier(BaseClassifier):
    """Support Vector Machine classifier implementation."""
    
    def __init__(self, task_id: str, output_dir: Path, random_state: int = 42):
        super().__init__(task_id, output_dir, random_state)
        self.model = SVC(
            probability=True,
            random_state=random_state,
            kernel='rbf'
        )
    
    def get_classifier_name(self) -> str:
        return "svm"
        
    def _fit_implementation(self, X: np.ndarray, y: pd.Series) -> None:
        self.model.fit(X, y)
        
    def _predict_implementation(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
        
    def _predict_proba_implementation(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

class RFClassifier(BaseClassifier):
    """Random Forest classifier implementation."""
    
    def __init__(self, task_id: str, output_dir: Path, random_state: int = 42):
        super().__init__(task_id, output_dir, random_state)
        self.model = SklearnRF(
            n_estimators=100,
            random_state=random_state
        )
    
    def get_classifier_name(self) -> str:
        return "random_forest"
        
    def _fit_implementation(self, X: np.ndarray, y: pd.Series) -> None:
        self.model.fit(X, y)
        
    def _predict_implementation(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
        
    def _predict_proba_implementation(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

class NeuralNetworkClassifier(BaseClassifier):
    """Neural Network classifier implementation."""
    
    def __init__(self, task_id: str, output_dir: Path, random_state: int = 42):
        super().__init__(task_id, output_dir, random_state)
        self.model = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=1000,
            random_state=random_state
        )
    
    def get_classifier_name(self) -> str:
        return "neural_network"
        
    def _fit_implementation(self, X: np.ndarray, y: pd.Series) -> None:
        self.model.fit(X, y)
        
    def _predict_implementation(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
        
    def _predict_proba_implementation(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

class KNNClassifier(BaseClassifier):
    """K-Nearest Neighbors classifier implementation."""
    
    def __init__(self, task_id: str, output_dir: Path, random_state: int = 42):
        super().__init__(task_id, output_dir, random_state)
        self.model = KNeighborsClassifier(n_neighbors=5)
    
    def get_classifier_name(self) -> str:
        return "knn"
        
    def _fit_implementation(self, X: np.ndarray, y: pd.Series) -> None:
        self.model.fit(X, y)
        
    def _predict_implementation(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
        
    def _predict_proba_implementation(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

def get_classifier(classifier_type: str, task_id: str, output_dir: Path, 
                  random_state: int = 42) -> BaseClassifier:
    """
    Factory function to get classifier instance.
    
    Args:
        classifier_type: Type of classifier to create
        task_id: Task identifier
        output_dir: Output directory for results
        random_state: Random seed
        
    Returns:
        Classifier instance
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
                        
    return classifiers[classifier_type](task_id, output_dir, random_state)