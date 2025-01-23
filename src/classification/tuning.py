from typing import Dict, Any, List, Tuple, Union
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator
import random
from dataclasses import dataclass
from scipy import stats

@dataclass
class SearchSpace:
    """Define parameter search spaces for different classifiers."""
    
    @staticmethod
    def _create_uniform(low: float, high: float):
        """Create a uniform distribution with proper scaling."""
        return stats.uniform(loc=low, scale=high-low)
    
    @staticmethod
    def _create_randint(low: int, high: int):
        """Create a random integer distribution."""
        return stats.randint(low=low, high=high)
    
    svm_space = {
        'C': _create_uniform(0.1, 10.0),
        'gamma': _create_uniform(0.001, 0.1),
        'kernel': ['rbf', 'linear']
    }
    
    rf_space = {
        'n_estimators': _create_randint(50, 300),
        'max_depth': [None] + list(range(5, 31, 5)),
        'min_samples_split': _create_randint(2, 11),
        'min_samples_leaf': _create_randint(1, 5)
    }
    
    nn_space = {
        'hidden_layer_sizes': [(n,) for n in range(50, 201, 50)] + 
                            [(n1, n2) for n1, n2 in [(100,50), (150,75), (200,100)]],
        'alpha': _create_uniform(0.0001, 0.01),
        'learning_rate_init': _create_uniform(0.0001, 0.01)
    }
    
    knn_space = {
        'n_neighbors': _create_randint(3, 11),
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    }

class ExtendedSearchSpace(SearchSpace):
    """Extended search space including new classifiers."""
    
    xgb_space = {
        'n_estimators': SearchSpace._create_randint(50, 300),
        'max_depth': list(range(3, 11)),
        'learning_rate': SearchSpace._create_uniform(0.01, 0.3),
        'subsample': SearchSpace._create_uniform(0.6, 1.0),
        'colsample_bytree': SearchSpace._create_uniform(0.6, 1.0),
        'min_child_weight': list(range(1, 7))
    }
    
    catboost_space = {
        'iterations': SearchSpace._create_randint(100, 500),
        'depth': list(range(4, 11)),
        'learning_rate': SearchSpace._create_uniform(0.01, 0.3),
        'l2_leaf_reg': SearchSpace._create_uniform(1.0, 10.0),
        'subsample': SearchSpace._create_uniform(0.6, 1.0)
    }
    
    dt_space = {
        'max_depth': [None] + list(range(5, 31, 5)),
        'min_samples_split': SearchSpace._create_randint(2, 11),
        'min_samples_leaf': SearchSpace._create_randint(1, 5),
        'criterion': ['gini', 'entropy']
    }
    
    adaboost_space = {
        'n_estimators': SearchSpace._create_randint(50, 300),
        'learning_rate': SearchSpace._create_uniform(0.01, 2.0),
        'algorithm': ['SAMME', 'SAMME.R']
    }



class HalvingRandomSearch:
    """Implement Halving Random Search for efficient hyperparameter tuning."""
    
    def __init__(self, estimator: BaseEstimator, param_space: Dict, 
                 n_candidates: int = 20, factor: int = 3, min_resources: int = 10,
                 cv: int = 3, random_state: int = None):
        self.estimator = estimator
        self.param_space = param_space
        self.n_candidates = n_candidates
        self.factor = factor
        self.min_resources = min_resources
        self.cv = cv
        self.random_state = random_state
        
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
    
    def _sample_parameters(self) -> Dict[str, Any]:
        """Sample a random set of parameters from the search space."""
        params = {}
        for param_name, param_space in self.param_space.items():
            if isinstance(param_space, stats._distn_infrastructure.rv_frozen):
                value = param_space.rvs(random_state=self.random_state)
                params[param_name] = float(value) if isinstance(value, np.floating) else int(value)
            elif isinstance(param_space, list):
                params[param_name] = random.choice(param_space)
            else:
                raise ValueError(f"Unsupported parameter space type for {param_name}")
        return params
    
    def _evaluate_candidates(self, candidates: List[Dict], X: np.ndarray, 
                           y: np.ndarray, n_samples: int) -> List[Tuple[float, Dict]]:
        """Evaluate a list of candidate parameters."""
        results = []
        
        if n_samples < len(X):
            subset_idx = np.random.choice(len(X), n_samples, replace=False)
            X_subset = X[subset_idx]
            y_subset = y[subset_idx]
        else:
            X_subset = X
            y_subset = y
            
        for params in candidates:
            try:
                self.estimator.set_params(**params)
                cv_splits = min(self.cv, len(np.unique(y_subset)))
                if cv_splits < 2:
                    score = 0.0  # Not enough samples for meaningful CV
                else:
                    scores = cross_val_score(
                        self.estimator, X_subset, y_subset, 
                        cv=cv_splits, n_jobs=-1
                    )
                    score = np.mean(scores)
                
                if np.isfinite(score):
                    results.append((score, params))
            except Exception:
                continue
                
        return sorted(results, key=lambda x: x[0], reverse=True)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Tuple[Dict[str, Any], float]:
        """Run the hyperparameter search."""
        default_params = self._sample_parameters()
        best_params = default_params
        best_score = float('-inf')
        
        n_samples = len(X)
        current_candidates = [self._sample_parameters() 
                            for _ in range(self.n_candidates)]
        
        n_iterations = max(1, int(np.log(self.n_candidates) / np.log(self.factor)))
        current_resources = max(
            self.min_resources,
            int(n_samples / (self.factor ** n_iterations))
        )
        
        while current_resources <= n_samples and len(current_candidates) > 1:
            results = self._evaluate_candidates(
                current_candidates, X, y, current_resources
            )
            
            if results:
                if results[0][0] > best_score:
                    best_score = results[0][0]
                    best_params = results[0][1]
                
                n_candidates = max(1, len(current_candidates) // self.factor)
                current_candidates = [params for _, params in results[:n_candidates]]
                current_resources = min(n_samples, current_resources * self.factor)
            else:
                break
        
        return best_params, max(0.0, best_score)

class TunedClassifierMixin:
    """Mixin class to add tuning capabilities to classifiers."""
    
    def tune_hyperparameters(self, X: np.ndarray, y: np.ndarray, 
                           param_space: Dict, verbose: bool = False) -> None:
        """Tune hyperparameters using Halving Random Search."""
        if verbose:
            print(f"\nTuning hyperparameters for {self.__class__.__name__}")
        
        try:
            search = HalvingRandomSearch(
                self.model,
                param_space,
                random_state=self.random_state,
                cv=min(3, len(np.unique(y)))  # Adjust CV based on unique classes
            )
            
            best_params, best_score = search.fit(X, y)
            
            if verbose:
                print(f"Best parameters: {best_params}")
                print(f"Best CV score: {best_score:.3f}")
            
            self.model.set_params(**best_params)
            
        except Exception as e:
            if verbose:
                print(f"Warning: Hyperparameter tuning failed: {str(e)}")
                print("Using default parameters instead.")