import sys
from pathlib import Path
import hydra
from omegaconf import DictConfig

# Prevent Python from creating bytecode files
sys.dont_write_bytecode = True

from evaluation.metrics import evaluate_predictions
from utils.printer import ConsolePrinter
from pipeline.ensemble import EnsemblePipeline


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for the application."""
    pipeline = EnsemblePipeline(cfg)
    printer = ConsolePrinter()
    
    try:
        printer.print_header("Application Started")
        if cfg.settings.verbose:
            printer.print_config_table(cfg)
        
        pipeline.setup_directories()
        
        for run in range(cfg.classification.n_runs):
            printer.print_header(f"Performing run: {run + 1}")
            
            run_seed = cfg.classification.base_seed + run
                
            # Handle classification or load existing data
            if cfg.classification.enabled:
                predictions_df, confidence_df, output_clf = pipeline.run_classification(run, run_seed)
                
            else:
                predictions_df, confidence_df = pipeline.load_data()
        
            # Execute ensemble method and evaluate results
            result_df = pipeline.execute_ensemble_method(predictions_df, confidence_df)
            confusion_matrix, metrics = evaluate_predictions(result_df, cfg.settings.verbose)
                        
            # Save results
            if cfg.classification.enabled:
                pipeline.save_results_clf_ensemble(result_df, metrics, output_clf, run)
            else:
                pipeline.save_results(result_df, metrics, output_clf)
            
        printer.print_footer(success=True)
        return 0
        
    except Exception as e:
        printer.print_error(e)
        printer.print_footer(success=False)
        sys.exit(1)

if __name__ == "__main__":
    main()