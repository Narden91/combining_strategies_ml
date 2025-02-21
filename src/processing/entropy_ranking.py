import pandas as pd
import numpy as np
from typing import List, Dict
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import entropy
from rich import print


class DiversityMetrics:
    """Collection of diversity metrics for ensemble learning."""
    
    @staticmethod
    def disagreement_measure(predictions_df: pd.DataFrame, task1: str, task2: str) -> float:
        mask = predictions_df[[task1, task2]].notna().all(axis=1)
        if not mask.any():
            return 0.0
        preds1 = predictions_df.loc[mask, task1]
        preds2 = predictions_df.loc[mask, task2]
        disagreements = np.sum(preds1 != preds2)
        return disagreements / len(preds1) if len(preds1) else 0.0

    @staticmethod
    def correlation_measure(predictions_df: pd.DataFrame, task1: str, task2: str) -> float:
        mask = predictions_df[[task1, task2]].notna().all(axis=1)
        if not mask.any():
            return 0.0
        p1 = predictions_df.loc[mask, task1]
        p2 = predictions_df.loc[mask, task2]
        if len(p1.unique()) < 2 or len(p2.unique()) < 2:
            return 0.0

        # Phi coefficient
        n11 = np.sum((p1 == 1) & (p2 == 1))
        n00 = np.sum((p1 == 0) & (p2 == 0))
        n10 = np.sum((p1 == 1) & (p2 == 0))
        n01 = np.sum((p1 == 0) & (p2 == 1))

        n = n11 + n00 + n10 + n01
        if n == 0:
            return 0.0

        n1_ = n11 + n10
        n0_ = n01 + n00
        n_1 = n11 + n01
        n_0 = n10 + n00

        denom = np.sqrt(n1_ * n0_ * n_1 * n_0)
        if denom == 0:
            return 0.0

        phi = (n11 * n00 - n10 * n01) / denom
        # Higher |phi| => more correlation => less diversity => use 1 - |phi|
        return (1 - abs(phi)) / 2

    @staticmethod
    def kappa_measure(predictions_df: pd.DataFrame, task1: str, task2: str) -> float:
        mask = predictions_df[[task1, task2]].notna().all(axis=1)
        if not mask.any():
            return 0.0
        p1 = predictions_df.loc[mask, task1]
        p2 = predictions_df.loc[mask, task2]

        n = len(p1)
        if n == 0:
            return 0.0

        n11 = np.sum((p1 == 1) & (p2 == 1))
        n00 = np.sum((p1 == 0) & (p2 == 0))
        n10 = np.sum((p1 == 1) & (p2 == 0))
        n01 = np.sum((p1 == 0) & (p2 == 1))

        observed = (n11 + n00) / n
        p1_ = (n11 + n10) / n
        p2_ = (n11 + n01) / n
        expected = p1_ * p2_ + (1 - p1_) * (1 - p2_)
        if expected == 1:
            return 0.0

        kappa = (observed - expected) / (1 - expected)
        # Higher |kappa| => more correlation => less diversity => return 1 - |kappa|
        return 1 - abs(kappa)

    @staticmethod
    def q_statistic(predictions_df: pd.DataFrame, task1: str, task2: str) -> float:
        mask = predictions_df[[task1, task2]].notna().all(axis=1)
        if not mask.any():
            return 0.0
        df = predictions_df.loc[mask, [task1, task2]]
        n11 = np.sum((df[task1] == 1) & (df[task2] == 1))
        n00 = np.sum((df[task1] == 0) & (df[task2] == 0))
        n10 = np.sum((df[task1] == 1) & (df[task2] == 0))
        n01 = np.sum((df[task1] == 0) & (df[task2] == 1))

        denom = n11 * n00 + n10 * n01
        if denom == 0:
            return 0.0
        numerator = n11 * n00 - n10 * n01
        q_value = numerator / denom
        # 1 - |Q|
        return 1 - abs(q_value)

    @staticmethod
    def double_fault(predictions_df: pd.DataFrame, task1: str, task2: str) -> float:
        mask = predictions_df[[task1, task2]].notna().all(axis=1)
        if not mask.any():
            return 0.0
        df = predictions_df[mask]
        total = len(df)
        if total == 0:
            return 0.0
        both_wrong = np.sum((df[task1] == 0) & (df[task2] == 0))
        both_correct = np.sum((df[task1] == 1) & (df[task2] == 1))

        return 1 - (both_wrong / total + both_correct / total) / 2


class AdaptiveWeightCalculator:
    """
    If you want to adapt the weighting of each metric (disagreement, correlation, etc.)
    based on average entropy of predictions, keep it. 
    Otherwise you can remove or simplify it.
    """
    def __init__(self, predictions_df: pd.DataFrame):
        self.predictions_df = predictions_df
        self.task_cols = [col for col in predictions_df.columns if col.startswith('T')]

    def calculate_entropy_weights(self) -> Dict[str, float]:
        """Calculate weights based on average entropy of predictions."""
        entropies = []
        for col in self.task_cols:
            valid_preds = self.predictions_df[col].dropna()
            if len(valid_preds) > 0:
                counts = np.bincount(valid_preds.astype(int))
                probabilities = counts / len(valid_preds)
                entropies.append(entropy(probabilities))
        if not entropies:
            return {'disagreement': 0.30, 'correlation': 0.25, 'kappa': 0.25, 'double_fault': 0.20, 'q_statistic': 0.0}
        
        mean_entropy = np.mean(entropies)
        max_entropy = np.log(2)
        ratio = mean_entropy / max_entropy

        # Adjust weights with z-score normalization or a dynamic function based on ratio
        base_weights = {
            'disagreement': 0.3 * (1 + ratio),
            'correlation': 0.2 * (1 - ratio),
            'kappa': 0.2 * (1 - ratio),
            'double_fault': 0.2 * (1 - ratio),
            'q_statistic': 0.1
        }
        adjusted = {k: v * (1 + ratio * 0.1) for k, v in base_weights.items()}
        total = sum(adjusted.values())
        return {k: v / total for k, v in adjusted.items()}


class EntropyRanking:
    """
    Class-based approach for ranking classifiers by diversity + accuracy,
    selecting the top subset, and producing a final predicted_class column.
    """

    def __init__(self, predictions_df: pd.DataFrame, accuracy_df: pd.DataFrame = None):
        """
        predictions_df: columns T1, T2, ... (each 0/1), plus 'class' optionally
        accuracy_df: a one-row DF with columns T1, T2, ... giving each classifier's accuracy
        """
        print("[bold green]Executing Entropy Ranking-based Method[/bold green]")
        self.predictions_df = predictions_df
        self.accuracy_df = accuracy_df
        self.task_cols = [c for c in predictions_df.columns if c.startswith('T')]
        self.metrics = DiversityMetrics()
        self.weight_calculator = AdaptiveWeightCalculator(predictions_df)
        self.diversity_weights = self.weight_calculator.calculate_entropy_weights()

    def calculate_diversity_matrix(self) -> pd.DataFrame:
        """Combine multiple diversity measures into a single matrix via self.diversity_weights."""
        n = len(self.task_cols)
        combined = np.zeros((n, n))
        # Precompute masks for each classifier column pair
        for i in range(n):
            for j in range(i+1, n):
                mask = self.predictions_df[[self.task_cols[i], self.task_cols[j]]].notna().all(axis=1)
                if not mask.any():
                    metric_values = {key: 0.0 for key in self.diversity_weights}
                else:
                    metric_values = {
                        'disagreement': self.metrics.disagreement_measure(self.predictions_df, self.task_cols[i], self.task_cols[j]),
                        'correlation':  self.metrics.correlation_measure(self.predictions_df, self.task_cols[i], self.task_cols[j]),
                        'kappa':        self.metrics.kappa_measure(self.predictions_df, self.task_cols[i], self.task_cols[j]),
                        'double_fault': self.metrics.double_fault(self.predictions_df, self.task_cols[i], self.task_cols[j]),
                        'q_statistic':  self.metrics.q_statistic(self.predictions_df, self.task_cols[i], self.task_cols[j])
                    }
                # Combine metrics using weights
                score = sum(self.diversity_weights[k] * metric_values[k] for k in metric_values)
                combined[i, j] = score
                combined[j, i] = score  # mirror symmetric value
                
        return pd.DataFrame(combined, index=self.task_cols, columns=self.task_cols)

    def run(
        self,
        diversity_weight: float = 0.5,
        use_accuracy_weighted_ensemble: bool = False,
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        1) Compute diversity matrix
        2) For each task, get a single "diversity score" (e.g. average row value)
        3) Blend with validation accuracy (if provided) using 'diversity_weight'
        4) Select a subset of tasks (like top half)
        5) Produce final predictions by either simple majority or accuracy weighting.
        Returns a DataFrame with 'predicted_class'.
        """
        if len(self.task_cols) < 2:
            # trivial
            out_df = self.predictions_df.copy()
            out_df['predicted_class'] = self.predictions_df[self.task_cols[0]]
            return out_df

        # 1) Combined diversity matrix
        dm = self.calculate_diversity_matrix()

        # 2) single diversity score per task (avg row, for instance)
        raw_diversity = dm.values.mean(axis=1)
        if np.ptp(raw_diversity) > 0:
            div_scores = (raw_diversity - raw_diversity.min()) / np.ptp(raw_diversity)
        else:
            div_scores = raw_diversity
        div_scores = pd.Series(div_scores, index=self.task_cols)

        # 3) get accuracy scores
        acc_scores = np.zeros(len(self.task_cols))
        if self.accuracy_df is not None:
            # assume self.accuracy_df has 1 row, columns T1, T2,...
            acc_list = []
            for task in self.task_cols:
                if task in self.accuracy_df.columns:
                    acc_list.append(self.accuracy_df[task].values[0])
                else:
                    acc_list.append(0.0)
            acc_scores = np.array(acc_list)
            if np.ptp(acc_scores) > 0:
                acc_scores = (acc_scores - acc_scores.min()) / np.ptp(acc_scores)

        # combine them
        overall = diversity_weight * div_scores.values + (1 - diversity_weight) * acc_scores
        # sort tasks descending
        sorted_idx = np.argsort(-overall)
        sorted_tasks = [self.task_cols[i] for i in sorted_idx]

        # pick top subset
        n_t = len(self.task_cols)
        n_sel = max(3, min(n_t // 2, 6))
        selected = sorted_tasks[:n_sel]

        if verbose:
            print("[EntropyRanking] Overall Scores:")
            for i, t in enumerate(sorted_tasks):
                print(f"  {t:5s} => {overall[sorted_idx[i]]:.3f}")
            print(f"[EntropyRanking] Selected => {selected}")

        # 5) final predictions
        if not use_accuracy_weighted_ensemble:
            # simple majority
            final_preds = (self.predictions_df[selected].mean(axis=1) >= 0.5).astype(int)
        else:
            # weight by each task's accuracy
            weights = []
            for t in selected:
                if t in self.accuracy_df.columns:
                    w = self.accuracy_df[t].values[0]
                else:
                    w = 0.5  # fallback
                weights.append(w)
            weights = np.array(weights, dtype=float)
            wsum = weights.sum()
            if wsum > 0:
                weights /= wsum
                # compute weighted average row by row
                arr = self.predictions_df[selected].values
                weighted_scores = np.average(arr, axis=1, weights=weights)
                final_preds = (weighted_scores >= 0.5).astype(int)
            else:
                # fallback to majority
                final_preds = (self.predictions_df[selected].mean(axis=1) >= 0.5).astype(int)

        out_df = self.predictions_df.copy()
        out_df['predicted_class'] = final_preds
        return out_df
