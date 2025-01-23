from enum import Enum
from typing import Callable
import hydra
from omegaconf import DictConfig
import sys
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
        
        for path_name, path_value in cfg.paths.items():
            Path(path_value).mkdir(parents=True, exist_ok=True)
            printer.print_directory_creation(path_name, path_value) if verbose else None
        
        printer.print_spacer()
        
        data_loader = DataLoader(cfg)
        predictions_df = data_loader.load_predictions()
        
        confidence_df = None
        if combining_method in [CombiningMethod.WMV, CombiningMethod.RK]:
            confidence_df = data_loader.load_confidence()
        
        if verbose:
            print("\n[bold]Predictions Data:[/bold]")
            print(predictions_df)
            if confidence_df is not None:
                print("\n[bold]Confidence Data:[/bold]")
                print(confidence_df)
        
        method_func = METHOD_MAPPING[combining_method]
        result_df = method_func(predictions_df, confidence_df, data_loader, verbose)
        
        # Ensure that the 'predicted_class' column is present and in case convert to integer
        if 'predicted_class' not in result_df.columns:
            raise ValueError("Missing 'predicted_class' column in result DataFrame")
        
        result_df['predicted_class'] = result_df['predicted_class'].astype(int)
        
        # Evaluate Performance
        confusion_matrix, metrics = evaluate_predictions(result_df, verbose)
        
        printer.print_footer(success=True)
        
        return 0
        
    except Exception as e:
        printer.print_error(e)
        printer.print_footer(success=False)
        sys.exit(1)

if __name__ == "__main__":
    main()