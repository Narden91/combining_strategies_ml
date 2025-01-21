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
 
        
METHOD_MAPPING: dict[CombiningMethod, Callable] = {
    CombiningMethod.MV: majority_voting,
    CombiningMethod.WMV: weighted_majority_voting,
    CombiningMethod.RK: ranking
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
        confidence_df = data_loader.load_confidence()
        
        if verbose:
            print("\n[bold]Predictions Data:[/bold]")
            print(predictions_df)
            print("\n[bold]Confidence Data:[/bold]")
            print(confidence_df)
        
        # Execute the selected combining method
        method_func = METHOD_MAPPING[combining_method]
        method_func(predictions_df, confidence_df)
        
        
        # Print success footer
        printer.print_footer(success=True)
        
    except Exception as e:
        printer.print_error(e)
        printer.print_footer(success=False)
        sys.exit(1)

if __name__ == "__main__":
    main()