from enum import Enum
import os
from typing import Callable, Dict, Tuple, List, Optional
from pathlib import Path
import pandas as pd
import numpy as np
from omegaconf import DictConfig
from rich.progress import Progress

from utils.printer import ConsolePrinter
from data_loader.loader import DataLoader
from processing import (
    majority_vote,
    weighted_majority_vote,
    ranking,
    entropy_ranking,
    hill_climbing,
    simulated_annealing,
    tabu_search
)
from evaluation.metrics import evaluate_predictions
from classification.base import ClassificationManager, get_available_tasks
from classification.models import get_classifier


class EnsembleMethod(Enum):
    """Available ensemble combining methods."""
    
    MV = "mv"   # Majority Voting
    WMV = "wmv" # Weighted Majority Voting
    RK = "rk"   # Basic Ranking
    ERK = "erk" # Entropy Ranking
    HC = "hc"   # Hill Climbing
    SA = "sa"   # Simulated Annealing
    TS = "ts"   # Tabu Search

    @classmethod
    def from_string(cls, method_str: str) -> 'EnsembleMethod':
        """Convert string to EnsembleMethod enum."""
        try:
            return cls(method_str.lower())
        except ValueError:
            raise ValueError(
                f"Invalid combining method: {method_str}. "
                f"Valid options are: {[m.value for m in cls]}"
            )


class EnsemblePipeline:
    """Main pipeline for ensemble learning combination methods."""
    
    def __init__(self, config: DictConfig):
        """Initialize the pipeline with configuration."""
        self.config = config
        self.printer = ConsolePrinter()
        self.verbose = config.settings.verbose
        self.ensemble_method = EnsembleMethod.from_string(
            config.settings.combining_technique
        )
        
    def setup_directories(self) -> None:
        """Create necessary output directories."""
        for path_name, path_value in self.config.paths.items():
            Path(path_value).mkdir(parents=True, exist_ok=True)
            if self.verbose:
                self.printer.print_directory_creation(path_name, path_value)
    
    def run_classification(self, run:int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run the classification phase.

        Args:
            run (int): run number
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: dataframes with predictions and confidences
        """
        if self.verbose:
            self.printer.print_info("Starting classification phase...")

        feature_vector_path = Path(self.config.paths.data) / self.config.data.feature_vector.folder
        classification_output = Path(self.config.paths.output) / "classification_output"

        # self.printer.print_info("Loading classification data...")
        # self.printer.print_info(f"Feature vector path: {feature_vector_path}")

        # Load all CSV files from the feature vector folder
        csv_files = list(feature_vector_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No feature vector files found in {feature_vector_path}")

        manager = ClassificationManager(
            feature_vector_path,
            classification_output,
            n_runs=self.config.classification.n_runs,
            base_seed=self.config.classification.base_seed
        )

        aggregated_predictions = {}  # Store predictions per task
        aggregated_confidences = {}  # Store confidence scores per task
        original_classes = None  # Store ground truth labels

        with Progress() as progress:
            task = progress.add_task("[cyan]Processing tasks...", total=len(csv_files))

            for csv_file in csv_files:
                task_id = csv_file.stem.replace("_test_features", "")  # Normalize task ID
                
                progress.update(task, description=f"[cyan]Processing {task_id}...")

                # Load dataset
                X, y, id_column = manager.load_task_data(csv_file.stem)

                if X.empty or y.empty:
                    self.printer.print_warning(f"Task {task_id} has no valid data. Skipping.")
                    raise ValueError(f"Task {task_id} has no valid data.")

                # Store original class labels (assumed consistent across tasks)
                if original_classes is None:
                    original_classes = y.rename("Class")  # Rename for clarity

                classifier = get_classifier(
                    self.config.classification.classifier,
                    task_id,
                    classification_output,
                    self.config.classification.base_seed
                )

                # Train classifier
                classifier.fit(X, y)
                y_pred = classifier.predict(X)

                try:
                    y_proba = classifier.predict_proba(X)
                    if y_proba.ndim == 1:
                        y_proba = np.column_stack((1 - y_proba, y_proba))  # Convert to (n_samples, 2)
                except AttributeError:
                    # Some classifiers (like SVM with linear kernel) don't support `predict_proba`
                    y_proba = np.zeros((len(y_pred), 2))  # Placeholder
                    y_proba[:, 1] = y_pred  # Assign predicted values
                    y_proba[:, 0] = 1 - y_pred  # Invert to simulate probabilities

                predictions_df = pd.DataFrame({"Id": id_column, f"{task_id}": y_pred})
                confidences_df = pd.DataFrame({
                    "Id": id_column,
                    f"Cd_0_{task_id}": y_proba[:, 0],
                    f"Cd_1_{task_id}": y_proba[:, 1]
                })

                aggregated_predictions[task_id] = predictions_df
                aggregated_confidences[task_id] = confidences_df

                progress.update(task, advance=1)
            
                if task_id == "T02":
                    break

        # Merge all task results into a single DataFrame
        final_predictions_df = pd.DataFrame()
        final_confidences_df = pd.DataFrame()

        for task_id, df in aggregated_predictions.items():
            if final_predictions_df.empty:
                final_predictions_df = df
            else:
                final_predictions_df = final_predictions_df.merge(df, on="Id", how="outer")

        for task_id, df in aggregated_confidences.items():
            if final_confidences_df.empty:
                final_confidences_df = df
            else:
                final_confidences_df = final_confidences_df.merge(df, on="Id", how="outer")

        # Add original class column
        final_predictions_df = final_predictions_df.merge(original_classes, left_index=True, right_index=True, how="outer")

        # Save final DataFrames
        # output_path_predictions = classification_output / f"{classifier.get_classifier_name()}_predictions.csv"
        # output_path_confidences = classification_output / f"{classifier.get_classifier_name()}_confidences.csv"
        
        output_clf = classification_output / f"{classifier.get_classifier_name()}" 
        os.makedirs(output_clf, exist_ok=True)
        
        output_run = output_clf / f"run_{run + 1}"
        os.makedirs(output_run, exist_ok=True)
        
        output_path_predictions = output_run / f"Predictions.csv"
        output_path_confidences = output_run / f"Confidences.csv"

        final_predictions_df.to_csv(output_path_predictions, index=False)
        final_confidences_df.to_csv(output_path_confidences, index=False)

        if self.verbose:
            self.printer.print_info(f"Aggregated predictions saved to {output_path_predictions}")
            self.printer.print_info(f"Aggregated confidences saved to {output_path_confidences}")

        return final_predictions_df, final_confidences_df, output_clf
    
    def _process_classification_tasks(
        self,
        manager: ClassificationManager,
        feature_vector_path: Path
    ) -> List[str]:
        """Process individual classification tasks."""
        tasks = get_available_tasks(feature_vector_path)
        if not tasks:
            raise ValueError("No feature vector files found")
            
        if self.verbose:
            self.printer.print_info(f"Found {len(tasks)} tasks to process")
            self.printer.print_info(
                f"Will perform {self.config.classification.n_runs} runs for each task"
            )
        
        for task_id in tasks:
            if self.verbose:
                self.printer.print_info(f"Processing task {task_id}")
                
            classifier = get_classifier(
                self.config.classification.classifier,
                task_id,
                Path(self.config.paths.output) / "classification_output",
                self.config.classification.base_seed
            )
            
            X, y = manager.load_task_data(task_id)
            metrics = manager.train_classifier(
                classifier,
                X, y,
                test_size=self.config.classification.test_size
            )
            
            if self.verbose:
                print(f"\nTask {task_id} Results:")
                print(f"Mean Accuracy: {metrics['metrics']['mean_accuracy']:.3f}")
                print(f"Std Accuracy: {metrics['metrics']['std_accuracy']:.3f}")
                
        return tasks
    
    def _aggregate_classification_results(
        self,
        output_dir: Path,
        tasks: List[str],
        manager: ClassificationManager
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Aggregate results from multiple classification runs."""
        all_predictions = []
        all_probabilities = []
        
        # Get classifier instance to access its name and configuration
        classifier = get_classifier(
            self.config.classification.classifier,
            tasks[0],  # Use first task to get classifier type
            output_dir,
            self.config.classification.base_seed
        )
        classifier_name = classifier.get_classifier_name()
        
        # Create output path with classifier-specific subfolder
        classifier_output_path = output_dir / "aggregated" / classifier_name
        classifier_output_path.mkdir(parents=True, exist_ok=True)
        
        for task_id in tasks:
            task_predictions = []
            task_probabilities = []
            
            task_dir = output_dir / classifier_name / f"task_{task_id}"
            
            for run in range(self.config.classification.n_runs):
                run_dir = task_dir / f"run_{run}"
                
                preds = np.load(run_dir / "predictions.npy")
                probs = np.load(run_dir / "probabilities.npy")
                
                task_predictions.append(preds)
                task_probabilities.append(probs[:, 1])
            
            avg_predictions = np.round(np.mean(task_predictions, axis=0)).astype(int)
            avg_probabilities = np.mean(task_probabilities, axis=0)
            
            all_predictions.append(avg_predictions)
            all_probabilities.append(avg_probabilities)
        
        # Get true labels using classifier name
        true_labels = np.load(
            output_dir / classifier_name / 
            f"task_{tasks[0]}/run_0/true_labels.npy"
        )
        
        # Create result dataframes
        pred_df, conf_df = self._create_result_dataframes(
            all_predictions,
            all_probabilities,
            true_labels,
            len(tasks)
        )
        
        # Generate filenames with classifier configuration
        config_str = self._get_classifier_config_string()
        
        pred_filename = f"aggregated_predictions_{classifier_name}_{config_str}.csv"
        conf_filename = f"aggregated_confidences_{classifier_name}_{config_str}.csv"
        
        # Save to classifier-specific directory
        pred_df.to_csv(classifier_output_path / pred_filename, index=False)
        conf_df.to_csv(classifier_output_path / conf_filename, index=False)
        
        if self.verbose:
            self.printer.print_info(f"Saved aggregated results for {classifier_name}:")
            self.printer.print_info(f"- {pred_filename}")
            self.printer.print_info(f"- {conf_filename}")
        
        return pred_df, conf_df

    def _get_classifier_config_string(self) -> str:
        """Generate a string representing classifier configuration."""
        config = self.config.classification
        parts = [
            f"runs_{config.n_runs}",
            f"seed_{config.base_seed}",
            f"test_{int(config.test_size * 100)}pct"
        ]
        return "_".join(parts)
    
    def _create_result_dataframes(
        self,
        predictions: List[np.ndarray],
        probabilities: List[np.ndarray],
        true_labels: np.ndarray,
        n_tasks: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create prediction and confidence DataFrames from results."""
        pred_df = pd.DataFrame(
            np.column_stack(predictions),
            columns=[f'T{i+1}' for i in range(n_tasks)]
        )
        pred_df.insert(0, 'ID', range(1, len(pred_df) + 1))
        pred_df['class'] = true_labels
        
        conf_df = pd.DataFrame(
            np.column_stack(probabilities),
            columns=[f'Cd1_T{i+1}' for i in range(n_tasks)]
        )
        
        return pred_df, conf_df
    
    def load_data(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Load prediction and confidence data."""
        data_loader = DataLoader(self.config)
        predictions_df = data_loader.load_predictions()
        
        confidence_df = None
        if self.ensemble_method in [EnsembleMethod.WMV, EnsembleMethod.RK]:
            confidence_df = data_loader.load_confidence()
            
        return predictions_df, confidence_df
    
    def execute_ensemble_method(
        self,
        predictions_df: pd.DataFrame,
        confidence_df: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """Execute the selected ensemble method."""
        method_mapping = {
            EnsembleMethod.MV: lambda: majority_vote.majority_voting(
                predictions_df, verbose=self.verbose
            ),
            EnsembleMethod.WMV: lambda: self._execute_weighted_majority_voting(
                predictions_df, confidence_df
            ),
            EnsembleMethod.RK: lambda: ranking.ranking(
                predictions_df, confidence_df, verbose=self.verbose
            ),
            EnsembleMethod.ERK: lambda: entropy_ranking.entropy_ranking(
                predictions_df, confidence_df, verbose=self.verbose
            ),
            EnsembleMethod.HC: lambda: hill_climbing.hill_climbing_combine(
                predictions_df, confidence_df, verbose=self.verbose
            ),
            EnsembleMethod.SA: lambda: simulated_annealing.simulated_annealing_combine(
                predictions_df, confidence_df, verbose=self.verbose
            ),
            EnsembleMethod.TS: lambda: tabu_search.tabu_search_combine(
                predictions_df, confidence_df, verbose=self.verbose
            )
        }
        
        return method_mapping[self.ensemble_method]()
    
    def _execute_weighted_majority_voting(
        self,
        predictions_df: pd.DataFrame,
        confidence_df: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """Special handling for weighted majority voting method."""
        try:
            data_loader = DataLoader(self.config)
            validation_accuracies_df = data_loader.load_validation_accuracies()
            self.printer.print_info("Successfully loaded validation accuracies")
            return weighted_majority_vote.weighted_majority_voting(
                predictions_df,
                confidence_df,
                validation_accuracies_df,
                verbose=self.verbose
            )
        except Exception as e:
            self.printer.print_warning(f"Could not load validation accuracies: {str(e)}")
            self.printer.print_warning("Proceeding with confidence scores only")
            return weighted_majority_vote.weighted_majority_voting(
                predictions_df,
                confidence_df,
                verbose=self.verbose
            )
    
    def save_results(
        self,
        result_df: pd.DataFrame,
        metrics: Dict
    ) -> None:
        """Save ensemble results and metrics."""
        output_path = Path(self.config.paths.output) / "ensemble_output"
        output_path.mkdir(exist_ok=True)
        
        predictions_file = self._generate_output_filename("predictions")
        metrics_file = self._generate_output_filename("metrics")
        
        result_df.to_csv(output_path / predictions_file, index=False)
        pd.DataFrame([metrics]).to_csv(output_path / metrics_file, index=False)
        
        if self.verbose:
            self.printer.print_info("Results saved as:")
            self.printer.print_info(f"- {predictions_file}")
            self.printer.print_info(f"- {metrics_file}")
    
    def _generate_output_filename(self, suffix: str) -> str:
        """Generate detailed output filename."""
        parts = [f"ensemble_{self.ensemble_method.value}"]
        
        if self.config.classification.enabled:
            parts.extend([
                f"clf_{self.config.classification.classifier}",
                f"runs_{self.config.classification.n_runs}"
            ])
        
        parts.append(suffix)
        return "_".join(parts) + ".csv"