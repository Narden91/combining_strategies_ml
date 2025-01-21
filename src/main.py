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


printer = ConsolePrinter()


class CombiningMethod(Enum):
    MV = "mv"
    WMV = "wmv"
    RK = "rk"

    @classmethod
    def get_method(cls, method_str: str) -> 'CombiningMethod':
        try:
            return cls(method_str.lower())
        except ValueError:
            raise ValueError(f"Invalid combining method: {method_str}. Valid options are: {[m.value for m in cls]}")
 
        
def execute_majority_voting(predictions_df, _, __, verbose):
    """
    Execute majority voting method.
    Only uses predictions, ignores other parameters.
    """
    return majority_voting(predictions_df, verbose=verbose)


def execute_weighted_majority_voting(predictions_df, confidence_df, data_loader, verbose):
    """
    Execute weighted majority voting with both confidence scores and validation accuracies.
    
    Args:
        predictions_df: DataFrame containing predictions
        confidence_df: DataFrame containing confidence scores
        data_loader: DataLoader instance for loading validation accuracies
    """
    try:
        validation_accuracies_df = data_loader.load_validation_accuracies()
        printer.print_info("Successfully loaded validation accuracies")
        return weighted_majority_voting(predictions_df, confidence_df, validation_accuracies_df, verbose=verbose)
    except Exception as e:
        printer.print_warning(f"Could not load validation accuracies: {str(e)}")
        printer.print_warning("Proceeding with confidence scores only")
        return weighted_majority_voting(predictions_df, confidence_df, verbose=verbose)


def execute_ranking(predictions_df, confidence_df, _, verbose):
    """Execute ranking method."""
    return ranking(predictions_df, confidence_df, verbose)


METHOD_MAPPING: dict[CombiningMethod, Callable] = {
    CombiningMethod.MV: execute_majority_voting,
    CombiningMethod.WMV: execute_weighted_majority_voting,
    CombiningMethod.RK: execute_ranking
}


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main function.
    
    Args:
        cfg: Hydra configuration object
    """
    verbose = cfg.settings.verbose
        
    try:
        # Print header and configuration
        printer.print_header("Application Started")
        printer.print_config_table(cfg) if verbose else None
        
        combining_method = CombiningMethod.get_method(cfg.settings.combining_technique)
        printer.print_spacer()
        
        # Create directories
        for path_name, path_value in cfg.paths.items():
            Path(path_value).mkdir(parents=True, exist_ok=True)
            printer.print_directory_creation(path_name, path_value) if verbose else None
        
        printer.print_spacer()
        
        # Initialize data loader and load data
        data_loader = DataLoader(cfg)
        predictions_df = data_loader.load_predictions()
        
        # Only load confidence data for methods that need it
        confidence_df = None
        if combining_method in [CombiningMethod.WMV, CombiningMethod.RK]:
            confidence_df = data_loader.load_confidence()
        
        if verbose:
            print("\n[bold]Predictions Data:[/bold]")
            print(predictions_df)
            if confidence_df is not None:
                print("\n[bold]Confidence Data:[/bold]")
                print(confidence_df)
        
        # Execute the selected combining method
        method_func = METHOD_MAPPING[combining_method]
        result = method_func(predictions_df, confidence_df, data_loader, verbose)
        
        # Print success footer
        printer.print_footer(success=True)
        
        return result
        
    except Exception as e:
        printer.print_error(e)
        printer.print_footer(success=False)
        sys.exit(1)

if __name__ == "__main__":
    main()