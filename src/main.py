from enum import Enum
from typing import Callable, Dict, Tuple, List
import hydra
from omegaconf import DictConfig
import sys
import pandas as pd
import numpy as np
sys.dont_write_bytecode = True
from pathlib import Path
from utils.printer import ConsolePrinter
from data_loader.loader import DataLoader
from rich import print
from processing.majority_vote import majority_voting
from processing.weighted_majority_vote import weighted_majority_voting
from processing.ranking import ranking
from processing.entropy_ranking import entropy_ranking
from processing.hill_climbing import hill_climbing_combine
from processing.simulated_annealing import simulated_annealing_combine
from processing.tabu_search import tabu_search_combine
from evaluation.metrics import evaluate_predictions
from classification.base import ClassificationManager, get_available_tasks
from classification.models import get_classifier

printer = ConsolePrinter()

class CombiningMethod(Enum):
    MV = "mv"      # Majority Voting
    WMV = "wmv"    # Weighted Majority Voting
    RK = "rk"      # Basic Ranking
    ERK = "erk"    # Entropy Ranking
    HC = "hc"      # Hill Climbing
    SA = "sa"      # Simulated Annealing
    TS = "ts"      # Tabu Search

    @classmethod
    def get_method(cls, method_str: str) -> 'CombiningMethod':
        try:
            return cls(method_str.lower())
        except ValueError:
            raise ValueError(f"Invalid combining method: {method_str}. Valid options are: {[m.value for m in cls]}")

def aggregate_classification_results(base_output_dir: Path, tasks: List[str], 
                                  classifier_name: str, n_runs: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate results from multiple classification runs.
    
    Args:
        base_output_dir: Base directory containing results
        tasks: List of task IDs
        classifier_name: Name of the classifier used
        n_runs: Number of runs to aggregate
        
    Returns:
        Tuple of (predictions_df, confidence_df)
    """
    all_predictions = []
    all_probabilities = []
    
    # For each task
    for task_id in tasks:
        task_predictions = []
        task_probabilities = []
        
        # Load results from each run
        task_dir = base_output_dir / classifier_name / f"task_{task_id}"
        for run in range(n_runs):
            run_dir = task_dir / f"run_{run}"
            
            # Load predictions and probabilities
            preds = np.load(run_dir / "predictions.npy")
            probs = np.load(run_dir / "probabilities.npy")
            
            task_predictions.append(preds)
            task_probabilities.append(probs[:, 1])  # Probability for class 1
            
        # Average across runs
        avg_predictions = np.round(np.mean(task_predictions, axis=0)).astype(int)
        avg_probabilities = np.mean(task_probabilities, axis=0)
        
        all_predictions.append(avg_predictions)
        all_probabilities.append(avg_probabilities)
    
    # Get true labels from any run (they're the same across runs)
    true_labels = np.load(base_output_dir / classifier_name / f"task_{tasks[0]}/run_0/true_labels.npy")
    
    # Create predictions DataFrame
    pred_df = pd.DataFrame(
        np.column_stack(all_predictions),
        columns=[f'T{i+1}' for i in range(len(tasks))]
    )
    pred_df.insert(0, 'ID', range(1, len(pred_df) + 1))
    pred_df['class'] = true_labels
    
    # Create confidence DataFrame
    conf_df = pd.DataFrame(
        np.column_stack(all_probabilities),
        columns=[f'Cd1_T{i+1}' for i in range(len(tasks))]
    )
    
    return pred_df, conf_df


def generate_output_filename(cfg: DictConfig, combining_method: CombiningMethod, suffix: str) -> str:
    """
    Generate detailed output filename based on configuration and method.
    
    Args:
        cfg: Configuration object
        combining_method: Ensemble method used
        suffix: File suffix (e.g., 'predictions', 'metrics')
        
    Returns:
        Formatted filename
    """    
    # Build parts of the filename
    parts = []
    
    # Add ensemble method
    parts.append(f"ensemble_{combining_method.value}")
    
    # Add classifier info if classification was used
    if cfg.classification.enabled:
        parts.append(f"clf_{cfg.classification.classifier}")
        parts.append(f"runs_{cfg.classification.n_runs}")
    
    parts.append(suffix)
    
    # Combine all parts
    return "_".join(parts) + ".csv"


def run_classification(cfg: DictConfig, verbose: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the classification step for all available tasks.
    
    Args:
        cfg: Configuration object
        verbose: Whether to print detailed information
        
    Returns:
        Tuple of (predictions_df, confidence_df)
    """
    if verbose:
        printer.print_info("Starting classification phase...")
        
    feature_vector_path = Path(cfg.paths.data) / cfg.data.feature_vector.folder
    classification_output = Path(cfg.paths.output) / "classification_output"
    
    manager = ClassificationManager(
        feature_vector_path,
        classification_output,
        n_runs=cfg.classification.n_runs,
        base_seed=cfg.classification.base_seed
    )
    
    # Get available tasks
    tasks = get_available_tasks(feature_vector_path)
    if not tasks:
        raise ValueError("No feature vector files found")
    
    if verbose:
        printer.print_info(f"Found {len(tasks)} tasks to process")
        printer.print_info(f"Will perform {cfg.classification.n_runs} runs for each task")
    
    for task_id in tasks:
        if verbose:
            printer.print_info(f"Processing task {task_id}")
            
        # Get classifier instance
        classifier = get_classifier(
            cfg.classification.classifier,
            task_id,
            classification_output,
            cfg.classification.base_seed
        )
        
        # Load and process data
        X, y = manager.load_task_data(task_id)
        metrics = manager.train_classifier(
            classifier,
            X, y,
            test_size=cfg.classification.test_size
        )
        
        if verbose:
            print(f"\nTask {task_id} Results:")
            print(f"Mean Accuracy: {metrics['metrics']['mean_accuracy']:.3f}")
            print(f"Std Accuracy: {metrics['metrics']['std_accuracy']:.3f}")
    
    # Aggregate results from all runs
    return aggregate_classification_results(
        classification_output,
        tasks,
        classifier.get_classifier_name(),
        cfg.classification.n_runs
    )

def execute_majority_voting(predictions_df, _, __, verbose):
    return majority_voting(predictions_df, verbose=verbose)

def execute_weighted_majority_voting(predictions_df, confidence_df, data_loader, verbose):
    try:
        validation_accuracies_df = data_loader.load_validation_accuracies()
        printer.print_info("Successfully loaded validation accuracies")
        return weighted_majority_voting(predictions_df, confidence_df, validation_accuracies_df, verbose=verbose)
    except Exception as e:
        printer.print_warning(f"Could not load validation accuracies: {str(e)}")
        printer.print_warning("Proceeding with confidence scores only")
        return weighted_majority_voting(predictions_df, confidence_df, verbose=verbose)

def execute_ranking(predictions_df, confidence_df, _, verbose):
    return ranking(predictions_df, confidence_df, verbose)

def execute_entropy_ranking(predictions_df, confidence_df, _, verbose):
    return entropy_ranking(predictions_df, confidence_df, verbose)

def execute_hill_climbing(predictions_df, confidence_df, _, verbose):
    return hill_climbing_combine(predictions_df, confidence_df, verbose)

def execute_simulated_annealing(predictions_df, confidence_df, _, verbose):
    return simulated_annealing_combine(predictions_df, confidence_df, verbose)

def execute_tabu_search(predictions_df, confidence_df, _, verbose):
    return tabu_search_combine(predictions_df, confidence_df, verbose)

METHOD_MAPPING: dict[CombiningMethod, Callable] = {
    CombiningMethod.MV: execute_majority_voting,
    CombiningMethod.WMV: execute_weighted_majority_voting,
    CombiningMethod.RK: execute_ranking,
    CombiningMethod.ERK: execute_entropy_ranking,
    CombiningMethod.HC: execute_hill_climbing,
    CombiningMethod.SA: execute_simulated_annealing,
    CombiningMethod.TS: execute_tabu_search
}

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    verbose = cfg.settings.verbose
    
    try:
        printer.print_header("Application Started")
        printer.print_config_table(cfg) if verbose else None
        
        combining_method = CombiningMethod.get_method(cfg.settings.combining_technique)
        printer.print_spacer()
        
        # Create necessary directories
        for path_name, path_value in cfg.paths.items():
            Path(path_value).mkdir(parents=True, exist_ok=True)
            printer.print_directory_creation(path_name, path_value) if verbose else None
        
        printer.print_spacer()
        
        # Handle classification if enabled
        if cfg.classification.enabled:
            predictions_df, confidence_df = run_classification(cfg, verbose)
            
            # Save aggregated classification results
            output_path = Path(cfg.paths.output) / "classification_output" / "aggregated"
            output_path.mkdir(exist_ok=True)
            predictions_df.to_csv(output_path / "aggregated_predictions.csv", index=False)
            confidence_df.to_csv(output_path / "aggregated_confidences.csv", index=False)
        else:
            # Load existing predictions and confidence scores
            data_loader = DataLoader(cfg)
            predictions_df = data_loader.load_predictions()
            
            confidence_df = None
            if combining_method in [CombiningMethod.WMV, CombiningMethod.RK]:
                confidence_df = data_loader.load_confidence()
        
        if verbose:
            print("\n[bold]Working with Predictions:[/bold]")
            print(predictions_df)
            if confidence_df is not None:
                print("\n[bold]Working with Confidence Scores:[/bold]")
                print(confidence_df)
        
        # Execute selected combining method
        method_func = METHOD_MAPPING[combining_method]
        result_df = method_func(predictions_df, confidence_df, 
                              DataLoader(cfg) if not cfg.classification.enabled else None, 
                              verbose)
        
        # Ensure predicted_class column exists and is integer type
        if 'predicted_class' not in result_df.columns:
            raise ValueError("Missing 'predicted_class' column in result DataFrame")
        
        result_df['predicted_class'] = result_df['predicted_class'].astype(int)
        
        # Evaluate Performance
        confusion_matrix, metrics = evaluate_predictions(result_df, verbose)
        
        # Save final ensemble results with detailed filenames
        output_path = Path(cfg.paths.output) / "ensemble_output"
        output_path.mkdir(exist_ok=True)
        
        # Generate filenames for ensemble results
        predictions_file = generate_output_filename(cfg, combining_method, "predictions")
        metrics_file = generate_output_filename(cfg, combining_method, "metrics")
        
        # Save results with detailed filenames
        result_df.to_csv(output_path / predictions_file, index=False)
        pd.DataFrame([metrics]).to_csv(output_path / metrics_file, index=False)
        
        if verbose:
            printer.print_info(f"Results saved as:")
            printer.print_info(f"- {predictions_file}")
            printer.print_info(f"- {metrics_file}")
        
        printer.print_footer(success=True)
        return 0
        
    except Exception as e:
        printer.print_error(e)
        printer.print_footer(success=False)
        sys.exit(1)

if __name__ == "__main__":
    main()