from rich import print
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from scipy.stats import entropy
import logging

class DiversityMetrics:
    """Collection of diversity metrics for ensemble learning."""
    
    @staticmethod
    def disagreement_measure(predictions_df: pd.DataFrame, task1: str, task2: str) -> float:
        """Calculate disagreement between two classifiers."""
        return np.mean(predictions_df[task1] != predictions_df[task2])
    
    @staticmethod
    def correlation_measure(predictions_df: pd.DataFrame, task1: str, task2: str) -> float:
        """Calculate correlation-based diversity."""
        return 1 - np.corrcoef(predictions_df[task1], predictions_df[task2])[0, 1]
    
    @staticmethod
    def q_statistic(predictions_df: pd.DataFrame, task1: str, task2: str) -> float:
        """Calculate Q-statistic diversity measure."""
        N11 = np.sum((predictions_df[task1] == 1) & (predictions_df[task2] == 1))
        N00 = np.sum((predictions_df[task1] == 0) & (predictions_df[task2] == 0))
        N10 = np.sum((predictions_df[task1] == 1) & (predictions_df[task2] == 0))
        N01 = np.sum((predictions_df[task1] == 0) & (predictions_df[task2] == 1))
        
        try:
            Q = (N11 * N00 - N01 * N10) / (N11 * N00 + N01 * N10)
        except ZeroDivisionError:
            Q = 0
        return 1 - abs(Q)  # Convert to diversity measure

    @staticmethod
    def double_fault(predictions_df: pd.DataFrame, task1: str, task2: str) -> float:
        """Calculate double-fault diversity measure."""
        both_wrong = np.sum((predictions_df[task1] == 0) & (predictions_df[task2] == 0))
        return 1 - (both_wrong / len(predictions_df))

class EntropyRanking:
    """
    Enhanced ranking-based ensemble method with multiple diversity metrics
    and adaptive weighting.
    """
    
    def __init__(self, predictions_df: pd.DataFrame, confidence_df: pd.DataFrame = None):
        """
        Initialize the enhanced ranking system.
        
        Args:
            predictions_df: DataFrame containing predictions from multiple classifiers
            confidence_df: Optional DataFrame containing confidence scores
        """
        self.predictions_df = predictions_df
        self.confidence_df = confidence_df
        self.task_cols = [col for col in predictions_df.columns if col.startswith('T')]
        self.diversity_metrics = DiversityMetrics()
        self.diversity_weights = self._calculate_entropy_based_weights()

    def _calculate_entropy_based_weights(self) -> Dict[str, float]:
        """Calculate weights for diversity metrics based on prediction entropy."""
        predictions = self.predictions_df[self.task_cols].values
        
        # Calculate entropy for each classifier's predictions
        entropies = [entropy(np.bincount(col, minlength=2) / len(col)) 
                    for col in predictions.T]
        
        # Normalize entropies
        total_entropy = sum(entropies)
        if total_entropy == 0:
            return {
                'disagreement': 0.4,
                'correlation': 0.3,
                'q_statistic': 0.2,
                'double_fault': 0.1
            }
        
        # Higher entropy → more weight to complex diversity measures
        entropy_ratio = np.mean(entropies) / np.log(2)
        
        weights = {
            'disagreement': 0.4 * (1 - entropy_ratio) + 0.2 * entropy_ratio,
            'correlation': 0.3 * entropy_ratio + 0.2 * (1 - entropy_ratio),
            'q_statistic': 0.2 * entropy_ratio + 0.3 * (1 - entropy_ratio),
            'double_fault': 0.1 * entropy_ratio + 0.3 * (1 - entropy_ratio)
        }
        
        return weights

    def calculate_pairwise_diversity(self) -> np.ndarray:
        """Calculate weighted pairwise diversity matrix using multiple measures."""
        n_tasks = len(self.task_cols)
        diversity_matrix = np.zeros((n_tasks, n_tasks))
        
        for i, task1 in enumerate(self.task_cols):
            for j, task2 in enumerate(self.task_cols):
                if i != j:
                    disagreement = self.diversity_metrics.disagreement_measure(
                        self.predictions_df, task1, task2
                    )
                    correlation = self.diversity_metrics.correlation_measure(
                        self.predictions_df, task1, task2
                    )
                    q_stat = self.diversity_metrics.q_statistic(
                        self.predictions_df, task1, task2
                    )
                    d_fault = self.diversity_metrics.double_fault(
                        self.predictions_df, task1, task2
                    )
                    
                    diversity_matrix[i, j] = (
                        self.diversity_weights['disagreement'] * disagreement +
                        self.diversity_weights['correlation'] * correlation +
                        self.diversity_weights['q_statistic'] * q_stat +
                        self.diversity_weights['double_fault'] * d_fault
                    )
                else:
                    diversity_matrix[i, j] = np.nan
                    
        return diversity_matrix

    def calculate_confidence_scores(self) -> np.ndarray:
        """Calculate enhanced confidence scores incorporating temporal stability."""
        if self.confidence_df is None:
            return np.zeros(len(self.task_cols))
            
        conf_cols = [col for col in self.confidence_df.columns if col.startswith('Cd1_')]
        
        if not conf_cols:
            return np.zeros(len(self.task_cols))
            
        mean_conf = self.confidence_df[conf_cols].mean().to_numpy()
        conf_stability = 1 - self.confidence_df[conf_cols].std().to_numpy()
        
        return 0.7 * mean_conf + 0.3 * conf_stability

    def select_optimal_ensemble(self, diversity_matrix: np.ndarray, 
                              confidence_scores: np.ndarray,
                              max_size: int = 10) -> List[str]:
        """
        Select optimal ensemble members using a greedy approach.
        
        Args:
            diversity_matrix: Matrix of pairwise diversity scores
            confidence_scores: Array of confidence scores for each classifier
            max_size: Maximum size of the ensemble
            
        Returns:
            List of selected task names
        """
        n_tasks = len(self.task_cols)
        selected = []
        available = set(range(n_tasks))
        
        # Start with highest confidence classifier
        start_idx = np.argmax(confidence_scores)
        selected.append(start_idx)
        available.remove(start_idx)
        
        while len(selected) < max_size and available:
            best_score = -np.inf
            best_idx = None
            
            for idx in available:
                div_score = np.mean([diversity_matrix[idx, j] for j in selected])
                alpha = 0.7 - 0.4 * (len(selected) / max_size)
                score = alpha * div_score + (1 - alpha) * confidence_scores[idx]
                
                if score > best_score:
                    best_score = score
                    best_idx = idx
            
            if best_idx is None:
                break
                
            selected.append(best_idx)
            available.remove(best_idx)
        
        return [self.task_cols[i] for i in selected]

    def get_ensemble_prediction(self, selected_tasks: List[str]) -> pd.Series:
        """Get weighted ensemble prediction using selected tasks."""
        predictions = self.predictions_df[selected_tasks].values
        
        if self.confidence_df is not None:
            conf_cols = [f"Cd1_{task.replace('T', '')}" for task in selected_tasks]
            weights = self.confidence_df[conf_cols].mean().values
            weights = weights / weights.sum()
            weighted_votes = np.average(predictions, axis=1, weights=weights)
            return (weighted_votes >= 0.5).astype(int)
        else:
            return pd.Series(np.mean(predictions, axis=1) >= 0.5).astype(int)

def entropy_ranking(predictions_df: pd.DataFrame, confidence_df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Execute entropy ranking-based ensemble method.
    
    Args:
        predictions_df: DataFrame containing predictions from multiple classifiers
        confidence_df: DataFrame containing confidence scores
        verbose: Whether to print detailed information
    
    Returns:
        DataFrame with combined predictions
    """
    print("[bold green]Executing Entropy Ranking-based Method[/bold green]")
    
    # Initialize enhanced ranking
    ranker = EntropyRanking(predictions_df, confidence_df)
    
    if verbose:
        print(f"\nAnalyzing {len(ranker.task_cols)} tasks...")
        print("\nDiversity weights:")
        for metric, weight in ranker.diversity_weights.items():
            print(f"  {metric}: {weight:.3f}")
    
    # Calculate diversity matrix
    diversity_matrix = ranker.calculate_pairwise_diversity()
    
    if verbose:
        print("\n[bold]Diversity Matrix:[/bold]")
        print(pd.DataFrame(diversity_matrix, columns=ranker.task_cols, index=ranker.task_cols))
    
    # Calculate confidence scores
    confidence_scores = ranker.calculate_confidence_scores()
    
    # Select optimal ensemble
    selected_tasks = ranker.select_optimal_ensemble(diversity_matrix, confidence_scores)
    
    if verbose:
        print(f"\nSelected tasks: {selected_tasks}")
    
    # Get ensemble predictions
    ensemble_predictions = ranker.get_ensemble_prediction(selected_tasks)
    
    # Create result DataFrame
    result_df = predictions_df.copy()
    result_df['predicted_class'] = ensemble_predictions
    
    if verbose:
        print("\n[bold]Final Predictions:[/bold]")
        print(result_df)
    
    return result_df