from rich import print
import pandas as pd
import numpy as np


def tabu_search_combine(predictions_df: pd.DataFrame, confidence_df: pd.DataFrame = None, verbose: bool = False) -> pd.DataFrame:
    """
    Implement tabu search-based ensemble method.
    """
    print("[bold green]Executing Tabu Search-based Method[/bold green]")
    
    task_cols = [col for col in predictions_df.columns if col.startswith('T')]
    if verbose:
        print(f"\nAnalyzing {len(task_cols)} tasks...")
    
    max_iter = 2000
    num_neighbors = 200
    patience = 200
    tabu_size = 50
    random_normal_loc = 0
    random_normal_scale = 0.2
    
    X = predictions_df[task_cols].values
    num_features = len(task_cols)
    
    best_weights = np.random.rand(num_features)
    best_score = -np.inf
    tabu_list = [best_weights.copy()]
    iterations_without_improvement = 0
    
    for iteration in range(max_iter):
        neighbors = np.random.normal(random_normal_loc, random_normal_scale, 
                                   (num_neighbors, num_features)) + best_weights
        neighbors = np.clip(neighbors, 0, 1)
        
        improved = False
        for neighbor in neighbors:
            if any(np.array_equal(neighbor, t) for t in tabu_list):
                continue
                
            weighted_sum = np.dot(X, neighbor)
            predictions = (weighted_sum > 0.5).astype(int)
            score = np.mean([np.mean(predictions == X[:, i]) for i in range(num_features)])
            
            if score > best_score:
                best_score = score
                best_weights = neighbor.copy()
                tabu_list.append(best_weights.copy())
                if len(tabu_list) > tabu_size:
                    tabu_list.pop(0)
                improved = True
                iterations_without_improvement = 0
                
        if not improved:
            iterations_without_improvement += 1
            
        if iterations_without_improvement >= patience:
            if verbose:
                print("\nEarly stopping triggered")
            break
            
        if verbose and iteration % 100 == 0:
            print(f"\nIteration {iteration}: Best score so far: {best_score:.3f}")
    
    final_weighted_sum = np.dot(X, best_weights)
    final_predictions = (final_weighted_sum > 0.5).astype(int)
    
    result_df = predictions_df.copy()
    result_df['predicted_class'] = final_predictions
    
    if verbose:
        print("\n[bold]Final Predictions:[/bold]")
        print(result_df)
    
    return result_df