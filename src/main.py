import sys
from pathlib import Path
import hydra
from omegaconf import DictConfig
import pandas as pd

# Prevent Python from creating bytecode files
sys.dont_write_bytecode = True

from utils.helpers import append_mean_std_for_method
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

        # DataFrame to store all run metrics
        all_metrics_df = pd.DataFrame()

        for run in range(cfg.classification.n_runs):
            printer.print_header(f"Performing run: {run + 1}")
            
            run_seed = cfg.classification.base_seed + run
                
            # Handle classification or load existing data
            if cfg.classification.enabled:
                predictions_df, confidence_df, output_clf, classifier_name, acc_df = pipeline.run_classification(run, run_seed, cfg.settings.save_individual_results)
            else:
                predictions_df, confidence_df, acc_df = pipeline.load_data()
                                
            for method in pipeline.ensemble_methods:
                if cfg.settings.verbose:
                    printer.print_info(f"Ensemble method {method.value} is being executed...")
                    printer.print_info(f"Predictions:\n{predictions_df}")
                    printer.print_info(f"Confidence:\n{confidence_df}")
                    printer.print_info(f"Accuracies:\n{acc_df}")

                # Get the ensemble result for this method
                result_df = pipeline.execute_ensemble_method(method, predictions_df, confidence_df, acc_df)
                
                # Evaluate predictions
                confusion_matrix, metrics = evaluate_predictions(result_df, cfg.settings.verbose)
                
                # Convert metrics to DataFrame
                run_metrics_df = pd.DataFrame([metrics])
                run_metrics_df.insert(0, "Run", run + 1)
                run_metrics_df.insert(1, "Method", method.value)
                
                # Append to accumulated results
                all_metrics_df = pd.concat([all_metrics_df, run_metrics_df], ignore_index=True)
        
        # Save aggregated metrics CSV
        if cfg.classification.enabled:
            metrics_output_path = Path(cfg.paths.output) / f"Metrics_{classifier_name}_ALL_METHODS.csv"
        else:
            metrics_output_path = Path(cfg.paths.output) / "Metrics_ALL_METHODS.csv"

        grouped = all_metrics_df.groupby("Method")
        
        final_metrics_df = grouped.apply(append_mean_std_for_method).reset_index(drop=True)

        final_metrics_df.to_csv(metrics_output_path, index=False)

        if cfg.settings.verbose:
            printer.print_info(f"Aggregated metrics saved to {metrics_output_path}")

        printer.print_footer(success=True)
        return 0
        
    except Exception as e:
        printer.print_error(e)
        printer.print_footer(success=False)
        sys.exit(1)

if __name__ == "__main__":
    main()