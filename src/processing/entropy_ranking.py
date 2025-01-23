from rich import print
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from scipy.stats import entropy
import logging
from sklearn.preprocessing import MinMaxScaler


class DiversityMetrics:
    """Enhanced collection of diversity metrics for ensemble learning."""
    
    @staticmethod
    def disagreement_measure(predictions_df: pd.DataFrame, task1: str, task2: str) -> float:
        """
        Calculate normalized disagreement between two classifiers.
        Returns a value between 0 and 1, where higher values indicate greater diversity.
        """
        mask = predictions_df[[task1, task2]].notna().all(axis=1)
        if not mask.any():
            return 0.0
        
        predictions1 = predictions_df.loc[mask, task1]
        predictions2 = predictions_df.loc[mask, task2]
        
        # Calculate normalized disagreement
        total = len(predictions1)
        if total == 0:
            return 0.0
            
        disagreements = np.sum(predictions1 != predictions2)
        return disagreements / total

    @staticmethod
    def correlation_measure(predictions_df: pd.DataFrame, task1: str, task2: str) -> float:
        """
        Calculate correlation-based diversity using phi coefficient.
        Returns a value between 0 and 1, where higher values indicate greater diversity.
        """
        mask = predictions_df[[task1, task2]].notna().all(axis=1)
        if not mask.any():
            return 0.0
            
        predictions1 = predictions_df.loc[mask, task1]
        predictions2 = predictions_df.loc[mask, task2]
        
        if len(predictions1.unique()) < 2 or len(predictions2.unique()) < 2:
            return 0.0
        
        # Calculate phi coefficient
        n11 = np.sum((predictions1 == 1) & (predictions2 == 1))
        n00 = np.sum((predictions1 == 0) & (predictions2 == 0))
        n10 = np.sum((predictions1 == 1) & (predictions2 == 0))
        n01 = np.sum((predictions1 == 0) & (predictions2 == 1))
        
        n = n11 + n00 + n10 + n01
        if n == 0:
            return 0.0
            
        n1_ = n11 + n10
        n0_ = n01 + n00
        n_1 = n11 + n01
        n_0 = n10 + n00
        
        denominator = np.sqrt(n1_ * n0_ * n_1 * n_0)
        if denominator == 0:
            return 0.0
            
        phi = (n11 * n00 - n10 * n01) / denominator
        return (1 - abs(phi)) / 2  # Normalize to [0, 1]

    @staticmethod
    def kappa_measure(predictions_df: pd.DataFrame, task1: str, task2: str) -> float:
        """
        Calculate diversity using Cohen's Kappa statistic.
        Returns a value between 0 and 1, where higher values indicate greater diversity.
        """
        mask = predictions_df[[task1, task2]].notna().all(axis=1)
        if not mask.any():
            return 0.0
            
        predictions1 = predictions_df.loc[mask, task1]
        predictions2 = predictions_df.loc[mask, task2]
        
        n = len(predictions1)
        if n == 0:
            return 0.0
            
        n11 = np.sum((predictions1 == 1) & (predictions2 == 1))
        n00 = np.sum((predictions1 == 0) & (predictions2 == 0))
        n10 = np.sum((predictions1 == 1) & (predictions2 == 0))
        n01 = np.sum((predictions1 == 0) & (predictions2 == 1))
        
        observed_agreement = (n11 + n00) / n
        p1 = (n11 + n10) / n
        p2 = (n11 + n01) / n
        expected_agreement = p1 * p2 + (1 - p1) * (1 - p2)
        
        if expected_agreement == 1:
            return 0.0
            
        kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
        return 1 - abs(kappa)  # Convert to diversity measure

    @staticmethod
    def double_fault(predictions_df: pd.DataFrame, task1: str, task2: str) -> float:
        """
        Calculate enhanced double-fault diversity measure.
        Returns a value between 0 and 1, where higher values indicate greater diversity.
        """
        mask = predictions_df[[task1, task2]].notna().all(axis=1)
        if not mask.any():
            return 0.0
            
        df = predictions_df[mask]
        total = len(df)
        if total == 0:
            return 0.0
        
        # Calculate both wrong and both correct cases
        both_wrong = np.sum((df[task1] == 0) & (df[task2] == 0))
        both_correct = np.sum((df[task1] == 1) & (df[task2] == 1))
        
        # Consider both aspects in the diversity measure
        fault_ratio = both_wrong / total if total > 0 else 0
        correct_ratio = both_correct / total if total > 0 else 0
        
        # Combine both ratios to get final diversity score
        return 1 - (fault_ratio + correct_ratio) / 2

class AdaptiveWeightCalculator:
    """Calculates adaptive weights for diversity metrics based on data characteristics."""
    
    def __init__(self, predictions_df: pd.DataFrame):
        self.predictions_df = predictions_df
        self.task_cols = [col for col in predictions_df.columns if col.startswith('T')]
        
    def calculate_entropy_weights(self) -> Dict[str, float]:
        """Calculate entropy-based weights for diversity metrics."""
        # Calculate prediction entropy for each classifier
        entropies = []
        for col in self.task_cols:
            valid_preds = self.predictions_df[col].dropna()
            if len(valid_preds) > 0:
                counts = np.bincount(valid_preds.astype(int))
                probabilities = counts / len(valid_preds)
                ent = entropy(probabilities)
                entropies.append(ent)
        
        if not entropies:
            return self._get_default_weights()
        
        mean_entropy = np.mean(entropies)
        max_entropy = np.log(2)  # Maximum entropy for binary classification
        entropy_ratio = mean_entropy / max_entropy
        
        # Adjust weights based on entropy ratio
        weights = {
            'disagreement': 0.35 * (1 - entropy_ratio) + 0.25 * entropy_ratio,
            'correlation': 0.25 * entropy_ratio + 0.25 * (1 - entropy_ratio),
            'kappa': 0.25 * entropy_ratio + 0.25 * (1 - entropy_ratio),
            'double_fault': 0.15 * entropy_ratio + 0.25 * (1 - entropy_ratio)
        }
        
        # Normalize weights
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}
    
    def _get_default_weights(self) -> Dict[str, float]:
        """Return default weights when entropy cannot be calculated."""
        return {
            'disagreement': 0.30,
            'correlation': 0.25,
            'kappa': 0.25,
            'double_fault': 0.20
        }

class EnhancedEntropyRanking:
    """Enhanced implementation of entropy-based ranking for ensemble learning."""
    
    def __init__(self, predictions_df: pd.DataFrame, confidence_df: pd.DataFrame = None):
        self.predictions_df = predictions_df
        self.confidence_df = confidence_df
        self.task_cols = [col for col in predictions_df.columns if col.startswith('T')]
        self.diversity_metrics = DiversityMetrics()
        self.weight_calculator = AdaptiveWeightCalculator(predictions_df)
        self.diversity_weights = self.weight_calculator.calculate_entropy_weights()
        
    def calculate_diversity_matrix(self) -> pd.DataFrame:
        """Calculate enhanced diversity matrix using multiple weighted measures."""
        n_tasks = len(self.task_cols)
        diversity_matrices = {
            'disagreement': np.zeros((n_tasks, n_tasks)),
            'correlation': np.zeros((n_tasks, n_tasks)),
            'kappa': np.zeros((n_tasks, n_tasks)),
            'double_fault': np.zeros((n_tasks, n_tasks))
        }
        
        # Calculate individual diversity matrices
        for i, task1 in enumerate(self.task_cols):
            for j, task2 in enumerate(self.task_cols):
                if i != j:
                    diversity_matrices['disagreement'][i, j] = self.diversity_metrics.disagreement_measure(
                        self.predictions_df, task1, task2
                    )
                    diversity_matrices['correlation'][i, j] = self.diversity_metrics.correlation_measure(
                        self.predictions_df, task1, task2
                    )
                    diversity_matrices['kappa'][i, j] = self.diversity_metrics.kappa_measure(
                        self.predictions_df, task1, task2
                    )
                    diversity_matrices['double_fault'][i, j] = self.diversity_metrics.double_fault(
                        self.predictions_df, task1, task2
                    )
        
        # Combine matrices using weights
        combined_matrix = np.zeros((n_tasks, n_tasks))
        for metric, weight in self.diversity_weights.items():
            combined_matrix += weight * diversity_matrices[metric]
        
        return pd.DataFrame(combined_matrix, columns=self.task_cols, index=self.task_cols)
    
    def calculate_task_scores(self, diversity_matrix: pd.DataFrame) -> pd.DataFrame:
        """Calculate enhanced task scores combining diversity and confidence."""
        # Calculate diversity scores
        diversity_scores = pd.Series(np.nansum(diversity_matrix.values, axis=1), index=self.task_cols)
        
        # Normalize diversity scores
        scaler = MinMaxScaler()
        diversity_scores = pd.Series(
            scaler.fit_transform(diversity_scores.values.reshape(-1, 1)).flatten(),
            index=diversity_scores.index
        )
        
        # Calculate confidence scores
        confidence_scores = pd.Series(0.0, index=self.task_cols)
        if self.confidence_df is not None:
            for task in self.task_cols:
                conf_col = f"Cd1_{task}"
                if conf_col in self.confidence_df.columns:
                    # Calculate both mean and stability of confidence
                    conf_values = self.confidence_df[conf_col]
                    mean_conf = conf_values.mean()
                    stability = 1 - conf_values.std()  # Higher stability is better
                    confidence_scores[task] = mean_conf * stability
        
        # Normalize confidence scores
        if confidence_scores.max() > 0:
            confidence_scores = confidence_scores / confidence_scores.max()
        
        # Calculate adaptive weight for combining scores
        mean_diversity = diversity_scores.mean()
        adaptive_weight = 0.6 + 0.2 * (1 - mean_diversity)  # Adjust weight based on diversity
        
        # Combine scores with adaptive weight
        overall_scores = (
            adaptive_weight * diversity_scores +
            (1 - adaptive_weight) * confidence_scores
        )
        
        return pd.DataFrame({
            'Task': self.task_cols,
            'Diversity_Score': diversity_scores,
            'Confidence_Score': confidence_scores,
            'Overall_Score': overall_scores
        }).sort_values('Overall_Score', ascending=False)
    
    def select_ensemble(self, rankings: pd.DataFrame) -> List[str]:
        """Select optimal ensemble members using enhanced selection strategy."""
        # Calculate optimal ensemble size
        n_tasks = len(self.task_cols)
        base_size = max(3, min(n_tasks // 2, 10))
        
        # Adjust size based on score distribution
        score_diff = rankings['Overall_Score'].diff().abs().mean()
        if score_diff < 0.1:  # Small differences between scores
            ensemble_size = max(3, base_size - 1)  # Reduce size
        elif score_diff > 0.3:  # Large differences between scores
            ensemble_size = min(n_tasks, base_size + 1)  # Increase size
        else:
            ensemble_size = base_size
        
        return rankings['Task'].tolist()[:ensemble_size]
    
    def get_ensemble_prediction(self, selected_tasks: List[str]) -> pd.Series:
        """Get enhanced ensemble prediction using selected tasks."""
        if not selected_tasks:
            return pd.Series(0, index=self.predictions_df.index)
        
        predictions = self.predictions_df[selected_tasks]
        
        if self.confidence_df is not None:
            # Calculate dynamic weights based on confidence and consistency
            weights = []
            for task in selected_tasks:
                conf_col = f"Cd1_{task}"
                if conf_col in self.confidence_df.columns:
                    conf_values = self.confidence_df[conf_col]
                    mean_conf = conf_values.mean()
                    stability = 1 - conf_values.std()
                    weight = mean_conf * stability
                    weights.append(weight if not np.isnan(weight) else 0)
                else:
                    weights.append(1)
            
            weights = np.array(weights)
            if weights.sum() > 0:
                weights = weights / weights.sum()
                weighted_pred = np.average(predictions, axis=1, weights=weights)
                return (weighted_pred >= 0.5).astype(int)
        
        return (predictions.mean(axis=1) >= 0.5).astype(int)
    

def entropy_ranking(predictions_df: pd.DataFrame, confidence_df: pd.DataFrame = None, 
                   verbose: bool = False) -> pd.DataFrame:
    """
    Execute enhanced entropy ranking-based ensemble method with adaptive weighting.
    
    Args:
        predictions_df: DataFrame containing predictions from multiple classifiers
        confidence_df: Optional DataFrame containing confidence scores
        verbose: Whether to print detailed progress information
    
    Returns:
        DataFrame with original data and added predicted_class column
    """
    print("[bold green]Executing Enhanced Entropy Ranking-based Method[/bold green]")
    
    # Get task columns (classifiers)
    task_cols = [col for col in predictions_df.columns if col.startswith('T')]
    n_tasks = len(task_cols)
    
    if verbose:
        print(f"\nAnalyzing {n_tasks} tasks...")
    
    if n_tasks < 2:
        print("[yellow]Warning: Not enough tasks for ensemble. Using single classifier.[/yellow]")
        result_df = predictions_df.copy()
        result_df['predicted_class'] = predictions_df[task_cols[0]]
        return result_df
    
    # Calculate diversity matrix
    diversity_matrix = np.zeros((n_tasks, n_tasks))
    for i, task1 in enumerate(task_cols):
        for j, task2 in enumerate(task_cols):
            if i != j:
                # Calculate disagreement (proportion of different predictions)
                mask = predictions_df[[task1, task2]].notna().all(axis=1)
                if mask.any():
                    preds1 = predictions_df.loc[mask, task1]
                    preds2 = predictions_df.loc[mask, task2]
                    diversity_matrix[i, j] = np.mean(preds1 != preds2)
    
    # Calculate confidence scores
    confidence_scores = np.zeros(n_tasks)
    if confidence_df is not None:
        for i, task in enumerate(task_cols):
            conf_col = f"Cd1_{task}"
            if conf_col in confidence_df.columns:
                confidence_scores[i] = confidence_df[conf_col].mean()
    
    # Normalize confidence scores
    if confidence_scores.max() > 0:
        confidence_scores = confidence_scores / confidence_scores.max()
    
    # Calculate diversity scores
    diversity_scores = np.mean(diversity_matrix, axis=1)
    if diversity_scores.max() > 0:
        diversity_scores = diversity_scores / diversity_scores.max()
    
    # Calculate combined scores with adaptive weighting
    mean_diversity = np.mean(diversity_scores)
    diversity_weight = 0.7 - 0.2 * mean_diversity  # Adjust weight based on overall diversity
    
    overall_scores = (
        diversity_weight * diversity_scores + 
        (1 - diversity_weight) * confidence_scores
    )
    
    # Create rankings DataFrame
    rankings = pd.DataFrame({
        'Task': task_cols,
        'Diversity_Score': diversity_scores,
        'Confidence_Score': confidence_scores,
        'Overall_Score': overall_scores
    })
    rankings = rankings.sort_values('Overall_Score', ascending=False)
    
    if verbose:
        print("\n[bold]Task Rankings:[/bold]")
        print(rankings)
    
    # Select ensemble members
    score_diffs = rankings['Overall_Score'].diff().abs()
    mean_diff = score_diffs.mean()
    
    # Adaptive ensemble size based on score distribution
    if mean_diff < 0.1:  # Small differences between scores
        n_selected = max(3, min(n_tasks // 2, 5))
    elif mean_diff > 0.3:  # Large differences between scores
        n_selected = max(3, min(n_tasks // 2, 7))
    else:
        n_selected = max(3, min(n_tasks // 2, 6))
    
    selected_tasks = rankings['Task'].tolist()[:n_selected]
    
    if verbose:
        print(f"\nSelected {n_selected} tasks: {selected_tasks}")
    
    # Get predictions from selected tasks
    selected_predictions = predictions_df[selected_tasks].values
    
    # Calculate weights for selected tasks
    if confidence_df is not None:
        weights = []
        for task in selected_tasks:
            conf_col = f"Cd1_{task}"
            if conf_col in confidence_df.columns:
                weight = confidence_df[conf_col].mean()
                weights.append(weight if not np.isnan(weight) else 1.0)
            else:
                weights.append(1.0)
        
        weights = np.array(weights)
        if weights.sum() > 0:
            weights = weights / weights.sum()
            weighted_sum = np.average(selected_predictions, weights=weights, axis=1)
        else:
            weighted_sum = np.mean(selected_predictions, axis=1)
    else:
        weighted_sum = np.mean(selected_predictions, axis=1)
    
    # Generate final predictions
    final_predictions = (weighted_sum >= 0.5).astype(int)
    
    # Create result DataFrame
    result_df = predictions_df.copy()
    result_df['predicted_class'] = final_predictions
    
    if verbose:
        print("\n[bold]Final Predictions:[/bold]")
        print(result_df)
    
    return result_df