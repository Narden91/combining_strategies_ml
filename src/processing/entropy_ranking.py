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
        # Only consider rows where both classifiers made predictions
        mask = predictions_df[[task1, task2]].notna().all(axis=1)
        if not mask.any():
            return 0.0
        
        predictions1 = predictions_df.loc[mask, task1]
        predictions2 = predictions_df.loc[mask, task2]
        return float(np.mean(predictions1 != predictions2))

    @staticmethod
    def correlation_measure(predictions_df: pd.DataFrame, task1: str, task2: str) -> float:
        """Calculate correlation-based diversity."""
        # Only consider rows where both classifiers made predictions
        mask = predictions_df[[task1, task2]].notna().all(axis=1)
        if not mask.any():
            return 0.0
        
        predictions1 = predictions_df.loc[mask, task1]
        predictions2 = predictions_df.loc[mask, task2]
        
        if len(predictions1.unique()) < 2 or len(predictions2.unique()) < 2:
            return 0.0
            
        correlation = np.corrcoef(predictions1, predictions2)[0, 1]
        if np.isnan(correlation):
            return 0.0
        return float(1 - abs(correlation))

    @staticmethod
    def q_statistic(predictions_df: pd.DataFrame, task1: str, task2: str) -> float:
        """Calculate Q-statistic diversity measure."""
        mask = predictions_df[[task1, task2]].notna().all(axis=1)
        if not mask.any():
            return 0.0
            
        df = predictions_df[mask]
        N11 = np.sum((df[task1] == 1) & (df[task2] == 1))
        N00 = np.sum((df[task1] == 0) & (df[task2] == 0))
        N10 = np.sum((df[task1] == 1) & (df[task2] == 0))
        N01 = np.sum((df[task1] == 0) & (df[task2] == 1))
        
        denominator = (N11 * N00 + N01 * N10)
        if denominator == 0:
            return 0.0
            
        try:
            Q = (N11 * N00 - N01 * N10) / denominator
            return float(1 - abs(Q))
        except:
            return 0.0

    @staticmethod
    def double_fault(predictions_df: pd.DataFrame, task1: str, task2: str) -> float:
        """Calculate double-fault diversity measure."""
        mask = predictions_df[[task1, task2]].notna().all(axis=1)
        if not mask.any():
            return 0.0
            
        df = predictions_df[mask]
        total = len(df)
        if total == 0:
            return 0.0
            
        both_wrong = np.sum((df[task1] == 0) & (df[task2] == 0))
        return float(1 - (both_wrong / total))

class EntropyRanking:
    def __init__(self, predictions_df: pd.DataFrame, confidence_df: pd.DataFrame = None):
        """Initialize the enhanced ranking system."""
        self.predictions_df = predictions_df
        self.confidence_df = confidence_df
        self.task_cols = [col for col in predictions_df.columns if col.startswith('T')]
        self.diversity_metrics = DiversityMetrics()
        self.diversity_weights = self._calculate_entropy_based_weights()

    def _calculate_entropy_based_weights(self) -> Dict[str, float]:
        """Calculate weights for diversity metrics based on prediction entropy."""
        if not self.task_cols:
            return {
                'disagreement': 0.4,
                'correlation': 0.3,
                'q_statistic': 0.2,
                'double_fault': 0.1
            }
            
        # Calculate entropy for each classifier's predictions
        entropies = []
        for col in self.task_cols:
            valid_preds = self.predictions_df[col].dropna()
            if len(valid_preds) > 0:
                counts = np.bincount(valid_preds.astype(int))
                probabilities = counts / len(valid_preds)
                ent = -np.sum(probabilities * np.log2(probabilities + 1e-10))
                entropies.append(ent)
        
        if not entropies:
            return {
                'disagreement': 0.4,
                'correlation': 0.3,
                'q_statistic': 0.2,
                'double_fault': 0.1
            }
        
        # Calculate entropy ratio
        max_entropy = -np.sum([0.5 * np.log2(0.5)] * 2)  # Maximum entropy for binary classification
        entropy_ratio = np.mean(entropies) / max_entropy
        
        # Adjust weights based on entropy
        weights = {
            'disagreement': 0.4 * (1 - entropy_ratio) + 0.2 * entropy_ratio,
            'correlation': 0.3 * entropy_ratio + 0.2 * (1 - entropy_ratio),
            'q_statistic': 0.2 * entropy_ratio + 0.3 * (1 - entropy_ratio),
            'double_fault': 0.1 * entropy_ratio + 0.3 * (1 - entropy_ratio)
        }
        
        # Normalize weights to sum to 1
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}

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
                    
                    # Combine metrics using weights
                    diversity_matrix[i, j] = (
                        self.diversity_weights['disagreement'] * disagreement +
                        self.diversity_weights['correlation'] * correlation +
                        self.diversity_weights['q_statistic'] * q_stat +
                        self.diversity_weights['double_fault'] * d_fault
                    )
                else:
                    diversity_matrix[i, j] = 0.0
                    
        return diversity_matrix

    def calculate_confidence_scores(self) -> np.ndarray:
        """Calculate enhanced confidence scores."""
        if self.confidence_df is None or len(self.task_cols) == 0:
            return np.zeros(len(self.task_cols))
            
        conf_cols = [f"Cd1_T{task[1:]}" for task in self.task_cols]
        valid_conf_cols = [col for col in conf_cols if col in self.confidence_df.columns]
        
        if not valid_conf_cols:
            return np.zeros(len(self.task_cols))
            
        conf_means = self.confidence_df[valid_conf_cols].mean()
        conf_stds = self.confidence_df[valid_conf_cols].std()
        
        # Normalize confidence scores
        conf_scores = np.zeros(len(self.task_cols))
        for i, task in enumerate(self.task_cols):
            conf_col = f"Cd1_T{task[1:]}"
            if conf_col in valid_conf_cols:
                mean_conf = conf_means[conf_col]
                std_conf = conf_stds[conf_col]
                conf_scores[i] = mean_conf * (1 - std_conf)
        
        # Normalize to [0, 1] range
        if conf_scores.max() > 0:
            conf_scores = conf_scores / conf_scores.max()
            
        return conf_scores

    def select_optimal_ensemble(self, diversity_matrix: np.ndarray, 
                              confidence_scores: np.ndarray,
                              max_size: int = 10) -> List[str]:
        """Select optimal ensemble members using a greedy approach."""
        if len(self.task_cols) == 0:
            return []
            
        n_tasks = len(self.task_cols)
        selected = []
        available = set(range(n_tasks))
        
        # Start with highest confidence classifier
        start_idx = np.argmax(confidence_scores)
        selected.append(start_idx)
        available.remove(start_idx)
        
        while len(selected) < min(max_size, n_tasks) and available:
            best_score = float('-inf')
            best_idx = None
            
            for idx in available:
                # Calculate average diversity with selected classifiers
                div_scores = [diversity_matrix[idx, j] for j in selected]
                div_score = np.mean(div_scores) if div_scores else 0
                
                # Weighted combination of diversity and confidence
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
        if not selected_tasks:
            # Return majority class if no tasks selected
            return pd.Series(np.zeros(len(self.predictions_df)), dtype=int)
            
        predictions = self.predictions_df[selected_tasks].values
        
        if self.confidence_df is not None:
            # Get corresponding confidence columns
            conf_cols = []
            for task in selected_tasks:
                conf_col = f"Cd1_T{task[1:]}"
                if conf_col in self.confidence_df.columns:
                    conf_cols.append(conf_col)
            
            if conf_cols:
                weights = self.confidence_df[conf_cols].mean().values
                weights = np.nan_to_num(weights, 0)  # Replace NaN with 0
                if weights.sum() > 0:
                    weights = weights / weights.sum()
                    weighted_votes = np.average(predictions, axis=1, weights=weights)
                    return pd.Series((weighted_votes >= 0.5).astype(int))
        
        # Fallback to simple majority voting
        return pd.Series((np.mean(predictions, axis=1) >= 0.5).astype(int))

def entropy_ranking(predictions_df: pd.DataFrame, confidence_df: pd.DataFrame, 
                   verbose: bool = False) -> pd.DataFrame:
    """Execute entropy ranking-based ensemble method."""
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
        div_df = pd.DataFrame(diversity_matrix, 
                            columns=ranker.task_cols,
                            index=ranker.task_cols)
        print(div_df)
    
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