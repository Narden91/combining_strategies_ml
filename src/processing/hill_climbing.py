from rich import print
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from sklearn.preprocessing import MinMaxScaler
from scipy.special import softmax
from concurrent.futures import ThreadPoolExecutor
import numpy.typing as npt

@dataclass
class HillClimbingConfig:
    """Configuration class for hill climbing parameters."""
    max_iterations: int = 300
    num_neighbors: int = 50
    patience: int = 50
    noise_loc: float = 0.0
    noise_scale: float = 0.3
    min_weight: float = 0.3
    temperature: float = 1.0
    decay_rate: float = 0.995
    threshold_range: Tuple[float, float] = (0.4, 0.6)
    n_threshold_steps: int = 10  # Number of threshold steps to try
    n_threads: int = 8  # Number of threads for parallel processing

def initialize_weights(num_features: int, validation_acc: Optional[pd.DataFrame] = None) -> np.ndarray:
    """Vectorized weight initialization."""
    if validation_acc is not None:
        task_cols = [f'T{str(i+1).zfill(2)}' for i in range(num_features)]
        acc_values = np.full(num_features, 0.5)  # Pre-allocate array
        
        # Vectorized assignment
        mask = np.array([col in validation_acc.columns for col in task_cols])
        acc_values[mask] = validation_acc.iloc[0][np.array(task_cols)[mask]].values
        
        acc_values = np.clip(acc_values, 0.5, 1)
        weights = softmax(acc_values)
    else:
        limit = np.sqrt(6 / num_features)
        weights = softmax(np.random.uniform(-limit, limit, num_features))
    
    return weights

def generate_neighbors_vectorized(weights: np.ndarray, config: HillClimbingConfig, 
                                iteration: int) -> np.ndarray:
    """Vectorized neighbor generation."""
    num_features = len(weights)
    current_temp = config.temperature * (config.decay_rate ** iteration)
    noise_scale = config.noise_scale * current_temp
    
    # Generate all neighbors at once
    neighbors = np.tile(weights, (config.num_neighbors, 1))
    
    # Apply Gaussian noise to 70% of neighbors
    noise_mask = np.random.random(config.num_neighbors) < 0.7
    num_noise = np.sum(noise_mask)
    
    if num_noise > 0:
        noise = np.random.normal(config.noise_loc, noise_scale, 
                               (num_noise, num_features))
        neighbors[noise_mask] += noise
    
    # Apply swap mutations to remaining 30%
    swap_mask = ~noise_mask
    num_swaps = np.sum(swap_mask)
    
    if num_swaps > 0:
        idx1 = np.random.randint(0, num_features, num_swaps)
        idx2 = np.random.randint(0, num_features, num_swaps)
        
        # Vectorized swap operation
        swap_neighbors = neighbors[swap_mask]
        temp = swap_neighbors[np.arange(num_swaps), idx1].copy()
        swap_neighbors[np.arange(num_swaps), idx1] = swap_neighbors[np.arange(num_swaps), idx2]
        swap_neighbors[np.arange(num_swaps), idx2] = temp
        neighbors[swap_mask] = swap_neighbors
    
    # Normalize all neighbors at once
    neighbors = np.clip(neighbors, config.min_weight, 1.0)
    row_sums = neighbors.sum(axis=1, keepdims=True)
    neighbors = neighbors / row_sums
    
    return neighbors

def evaluate_predictions_batch(weighted_sums: np.ndarray, threshold: float, 
                             X: np.ndarray, validation_acc: Optional[pd.DataFrame] = None) -> np.ndarray:
    """Vectorized prediction evaluation for multiple neighbors."""
    # Fix shape issues
    predictions = (weighted_sums > threshold).astype(int)
    
    if validation_acc is not None:
        task_cols = [f'T{str(i+1).zfill(2)}' for i in range(X.shape[1])]
        task_weights = np.full(X.shape[1], 0.5)
        
        mask = np.array([col in validation_acc.columns for col in task_cols])
        task_weights[mask] = validation_acc.iloc[0][np.array(task_cols)[mask]].values
        task_weights = softmax(task_weights)
        
        # Fix broadcasting issues
        predictions = predictions.reshape(-1, X.shape[0])  # Reshape to match X
        matches = predictions[:, :, np.newaxis] == X  # Compare with X
        confidence = np.abs(weighted_sums.reshape(-1, X.shape[0])[:, :, np.newaxis] - 0.5) * 2
        
        # Average across samples first
        task_scores = np.mean(matches, axis=1)
        
        return np.average(task_scores, weights=task_weights, axis=1)
    else:
        predictions = predictions.reshape(-1, X.shape[0])
        return np.mean([np.mean(predictions == X[:, i], axis=1) 
                       for i in range(X.shape[1])], axis=0)

def evaluate_neighbor_batch(neighbors: np.ndarray, X: np.ndarray, 
                          config: HillClimbingConfig,
                          validation_acc: Optional[pd.DataFrame] = None) -> Tuple[float, np.ndarray, float]:
    """Evaluate a batch of neighbors with vectorized operations."""
    weighted_sums = np.dot(X, neighbors.T)  # Compute all weighted sums at once
    thresholds = np.linspace(config.threshold_range[0], config.threshold_range[1], 
                            config.n_threshold_steps)
    
    best_score = -np.inf
    best_threshold = config.threshold_range[0]
    best_neighbor = None
    
    # Evaluate all thresholds for all neighbors at once
    for threshold in thresholds:
        scores = evaluate_predictions_batch(weighted_sums.T, threshold, X, validation_acc)
        best_idx = np.argmax(scores)
        if scores[best_idx] > best_score:
            best_score = scores[best_idx]
            best_threshold = threshold
            best_neighbor = neighbors[best_idx]
    
    return best_score, best_neighbor, best_threshold

def hill_climbing_combine(predictions_df: pd.DataFrame, 
                         validation_acc: Optional[pd.DataFrame] = None,
                         config: Optional[HillClimbingConfig] = None,
                         verbose: bool = False) -> pd.DataFrame:
    """Optimized hill climbing-based ensemble method."""
    print("[bold green]Executing Optimized Hill Climbing Method[/bold green]")
    
    if config is None:
        config = HillClimbingConfig()
    
    task_cols = [col for col in predictions_df.columns if col.startswith('T')]
    X = predictions_df[task_cols].values
    X = np.nan_to_num(X, nan=0.5)
    
    weights = initialize_weights(len(task_cols), validation_acc)
    best_weights = weights.copy()
    
    weighted_sum = np.dot(X, weights)
    threshold = np.mean(config.threshold_range)
    best_score = evaluate_predictions_batch(weighted_sum.reshape(1, -1), threshold, X, validation_acc)[0]
    best_threshold = threshold
    
    # Progress tracking
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("Best Score: {task.fields[best_score]:.4f}"),
    ) as progress:
        task = progress.add_task(
            "[cyan]Optimizing weights...",
            total=config.max_iterations,
            best_score=best_score
        )
        
        iterations_without_improvement = 0
        improvement_threshold = 1e-5
        
        # Main optimization loop
        for iteration in range(config.max_iterations):
            # Generate neighbors
            neighbors = generate_neighbors_vectorized(weights, config, iteration)
            weighted_sums = np.dot(X, neighbors.T)
            
            # Evaluate neighbors in parallel batches
            with ThreadPoolExecutor(max_workers=config.n_threads) as executor:
                batch_size = config.num_neighbors // config.n_threads
                neighbor_batches = np.array_split(neighbors, config.n_threads)
                
                future_results = [
                    executor.submit(evaluate_neighbor_batch, 
                                  batch, X, config, validation_acc)
                    for batch in neighbor_batches
                ]
                
                # Collect results
                batch_results = [future.result() for future in future_results]
            
            # # Find best result across all batches            
            new_score, best_neighbor, new_threshold = max(batch_results, key=lambda x: x[0])
            
            if new_score > best_score + 1e-5:
                best_score = new_score
                best_weights = best_neighbor
                best_threshold = new_threshold
                weights = best_weights.copy()
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1

            # Update progress
            progress.update(
                task,
                advance=1,
                best_score=best_score,
            )
            
            # Early stopping check
            if iterations_without_improvement >= config.patience and verbose:
                print(Panel.fit(
                    "[yellow]Early stopping triggered - No improvement for "
                    f"{config.patience} iterations[/yellow]"
                ))
                break
    
    # Generate final predictions
    final_weighted_sum = np.dot(X, best_weights)
    final_predictions = (final_weighted_sum > best_threshold).astype(int)
    
    # Create result DataFrame
    result_df = predictions_df.copy()
    result_df['predicted_class'] = final_predictions
    result_df['confidence'] = np.abs(final_weighted_sum - best_threshold) * 2
    
    if verbose:
        print(Panel.fit(
            f"[bold green]Optimization Complete\n"
            f"Final Score: {best_score:.4f}\n"
            f"Final Threshold: {best_threshold:.4f}[/bold green]"
        ))
    
    return result_df