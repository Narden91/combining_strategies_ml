import hydra
from omegaconf import DictConfig
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from utils.printer import ConsolePrinter
from utils.helpers import process_data
from data_loader.loader import DataLoader
from rich import print


printer = ConsolePrinter()


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
        
        print(f"\nPredictions data:\n{predictions_df}") if verbose else None
        print(f"\nConfidence data:\n{confidence_df}") if verbose else None
        
        # Process data (using your existing process_data function)
        # process_data(predictions_df, confidence_df)
        
        # Print success footer
        printer.print_footer(success=True)
        
    except Exception as e:
        printer.print_error(e)
        printer.print_footer(success=False)
        sys.exit(1)

if __name__ == "__main__":
    main()