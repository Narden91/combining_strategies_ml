import sys
from pathlib import Path
import hydra
from omegaconf import DictConfig
import pandas as pd

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

        # DataFrame to store all run metrics
        all_metrics_df = pd.DataFrame()

        for run in range(cfg.classification.n_runs):
            printer.print_header(f"Performing run: {run + 1}")
            
            run_seed = cfg.classification.base_seed + run
                
            # Handle classification or load existing data
            if cfg.classification.enabled:
                predictions_df, confidence_df, output_clf, classifier_name = pipeline.run_classification(run, run_seed, cfg.settings.save_individual_results)
            else:
                predictions_df, confidence_df = pipeline.load_data()
                
            if cfg.settings.verbose:
                printer.print_info("Ensemble method is being executed...")
            
                printer.print_info(f"Predictions df:\n {predictions_df}")
                printer.print_info(f"Confidence df:\n {confidence_df}")
        
            # Execute ensemble method and evaluate results
            result_df = pipeline.execute_ensemble_method(predictions_df, confidence_df)
            confusion_matrix, metrics = evaluate_predictions(result_df, cfg.settings.verbose)
            
            # Convert metrics to DataFrame
            run_metrics_df = pd.DataFrame([metrics])
            run_metrics_df.insert(0, "Run", run + 1)  # Add run identifier
            
            # Append to accumulated results
            all_metrics_df = pd.concat([all_metrics_df, run_metrics_df], ignore_index=True)

            # # Save individual results if needed
            # if cfg.classification.enabled and cfg.settings.save_individual_results:
            #     pipeline.save_results_clf_ensemble(result_df, metrics, output_clf, run, cfg.settings.combining_technique)
            # elif cfg.settings.save_individual_results:
            #     pipeline.save_results(result_df, metrics, output_clf)

        # Save aggregated metrics CSV
        if cfg.classification.enabled:
            metrics_output_path = Path(cfg.paths.output) / f"Metrics_{classifier_name}_{cfg.settings.combining_technique}.csv"
        else:
            metrics_output_path = Path(cfg.paths.output) / f"Metrics_{cfg.settings.combining_technique}.csv"
        
        # Add 2 rows at the end of the DataFrame to store the mean and standard deviation of the metrics
        mean_metrics = all_metrics_df.mean()
        std_metrics = all_metrics_df.std()
        mean_metrics["Run"] = "Mean"
        std_metrics["Run"] = "Std"
        mean_metrics_df = pd.DataFrame([mean_metrics])
        std_metrics_df = pd.DataFrame([std_metrics])

        all_metrics_df = pd.concat([all_metrics_df, mean_metrics_df, std_metrics_df], ignore_index=True)

        all_metrics_df.to_csv(metrics_output_path, index=False)

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