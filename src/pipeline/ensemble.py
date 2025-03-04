from enum import Enum
import os
from typing import Callable, Dict, Set, Tuple, List, Optional
from pathlib import Path
import pandas as pd
import numpy as np
from omegaconf import DictConfig, ListConfig

from rich.progress import Progress
from sklearn.metrics import precision_score, recall_score, confusion_matrix, matthews_corrcoef

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
        self.clf_folder = config.classification.classifier
        
        if isinstance(config.settings.combining_technique, str):
            # Single method
            self.ensemble_methods = [
                EnsembleMethod.from_string(config.settings.combining_technique)
            ]
        elif isinstance(config.settings.combining_technique, (list, ListConfig)):
            # Multiple methods
            self.ensemble_methods = [
                EnsembleMethod.from_string(method_str)
                for method_str in config.settings.combining_technique
            ]
        else:
            raise ValueError("Invalid 'combining_technique'. Must be str or list[str].")
        
    def setup_directories(self) -> None:
        """Create necessary output directories."""
        for path_name, path_value in self.config.paths.items():
            Path(path_value).mkdir(parents=True, exist_ok=True)
            if self.verbose:
                self.printer.print_directory_creation(path_name, path_value)
    
    def run_classification(self, run: int, seed: int, metrics_output_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Path, str, pd.DataFrame]:
        """
        Run the classification phase for all tasks, ensuring that only IDs common
        to every task appear in the final test sets.

        This prevents rows with NaN values in the aggregated predictions and 
        confidence DataFrames. For each CSV in the feature vector folder:
        1. We load all CSVs first to compute the intersection of IDs.
        2. For each task, we filter the dataset to retain only the common IDs.
        3. We perform a train/test split and train the chosen classifier.
        4. We aggregate the predictions and confidence scores in one DataFrame 
            each (final_predictions_df, final_confidences_df).

        Args:
            run (int): The run number (0-based).
            seed (int): The base random seed for reproducibility.
            metrics_output_path (Path): The root directory where results will be saved,
                with a subfolder named 'run_<run_number>'.

        Returns:
        Tuple[pd.DataFrame, pd.DataFrame, Path, str, pd.DataFrame]:
            - final_predictions_df: DataFrame containing the final predictions
              from every task, merged on ID.
            - final_confidences_df: DataFrame containing the final confidence
              scores from every task, merged on ID.
            - output_clf: The path to the run-specific output directory (metrics_output_path/run_<number>).
            - classifier_name: Name of the classifier used.
            - accuracy_df: A one-row DataFrame containing accuracy values for each
              task, with columns labeled by task ID.
        """
        if self.verbose:
            self.printer.print_info("Starting classification phase...")

        # Paths setup
        feature_vector_path = Path(self.config.paths.data) / self.config.data.feature_vector.folder
        
        # Use metrics_output_path as the root, with run subfolder
        run_output_path = metrics_output_path / f"{self.clf_folder}" / f"run_{run + 1}"
        os.makedirs(run_output_path, exist_ok=True)
        
        # Collect all CSVs
        csv_files = list(feature_vector_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No feature vector files found in {feature_vector_path}")
        
        task_accuracies = {}
        task_metrics = []

        # STEP 1: Determine the intersection of IDs across all tasks
        common_ids: Set[str] = set()
        first_file = True
        for csv_file in csv_files:
            df_temp = pd.read_csv(csv_file)
            if first_file:
                common_ids = set(df_temp["Id"])
                first_file = False
            else:
                common_ids &= set(df_temp["Id"])

        if not common_ids:
            raise ValueError("No common IDs found across all tasks. Cannot proceed.")

        # Classification manager
        manager = ClassificationManager(
            feature_vector_path,
            n_runs=self.config.classification.n_runs,
            base_seed=seed
        )

        aggregated_predictions = {}
        aggregated_confidences = {}
        all_ids = set()
        original_classes = None

        # Rich progress bar
        with Progress() as progress:
            task = progress.add_task("[cyan]Processing tasks...", total=len(csv_files))

            for csv_file in csv_files:
                task_id = csv_file.stem.replace("_test_features", "")
                progress.update(task, description=f"[cyan]Processing {task_id}...")

                # ------------------------------------------------------------------
                # STEP 2: Load dataset, keep only rows with common IDs
                # ------------------------------------------------------------------
                full_df = pd.read_csv(csv_file)
                full_df = full_df[full_df["Id"].isin(common_ids)]

                if full_df.empty:
                    self.printer.print_warning(f"Task {task_id} has no common IDs. Skipping.")
                    raise ValueError(f"Task {task_id} has no valid data after ID intersection.")

                X = full_df.drop('Class', axis=1)
                y = full_df['Class']

                # ------------------------------------------------------------------
                # STEP 3: Train the classifier
                # ------------------------------------------------------------------
                classifier = get_classifier(
                    self.config.classification.classifier,
                    task_id,
                    seed
                )

                classifier, y_pred, y_proba, id_col_test, y_test = manager.train_classifier(
                    classifier,
                    X,
                    y,
                    test_size=self.config.classification.test_size,
                    seed=seed
                )

                # ------------------------------------------------------------------
                # STEP 3.5: Compute metrics for this task
                # ------------------------------------------------------------------
                accuracy = np.mean(y_pred == y_test)
                task_accuracies[task_id] = accuracy  # Keep for return

                # Calculate additional metrics
                precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
                sensitivity = recall_score(y_test, y_pred, average='binary', zero_division=0)
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                mcc = matthews_corrcoef(y_test, y_pred)

                # Store metrics for this task in task_metrics
                task_metrics.append({
                    'Task': task_id,
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Sensitivity': sensitivity,
                    'Specificity': specificity,
                    'MCC': mcc
                })

                # Merge all IDs from the test set(s)
                all_ids.update(id_col_test)

                if original_classes is None:
                    original_classes = pd.DataFrame({"Id": id_col_test, "Class": y_test})

                # ------------------------------------------------------------------
                # STEP 4: Store predictions & confidences
                # ------------------------------------------------------------------
                predictions_df = pd.DataFrame({
                    "Id": id_col_test,
                    f"{task_id}": y_pred
                })
                confidences_df = pd.DataFrame({
                    "Id": id_col_test,
                    f"Cd0_{task_id}": y_proba[:, 0],
                    f"Cd1_{task_id}": y_proba[:, 1]
                })

                aggregated_predictions[task_id] = predictions_df
                aggregated_confidences[task_id] = confidences_df

                progress.update(task, advance=1)

        # ----------------------------------------------------------------------
        # STEP 5: Merge into final DataFrames
        # ------------------------------------------------------------------
        all_ids_df = pd.DataFrame({"Id": list(all_ids)})
        final_predictions_df = all_ids_df.copy()
        for task_id, df_pred in aggregated_predictions.items():
            final_predictions_df = final_predictions_df.merge(df_pred, on="Id", how="outer")

        final_confidences_df = all_ids_df.copy()
        for task_id, df_conf in aggregated_confidences.items():
            final_confidences_df = final_confidences_df.merge(df_conf, on="Id", how="outer")

        final_predictions_df = final_predictions_df.merge(original_classes, on="Id", how="left")
        final_confidences_df = final_confidences_df.merge(original_classes, on="Id", how="left")

        # Path info
        classifier_name = classifier.get_classifier_name()

        # STEP 6: Save results 
        output_path_predictions = run_output_path / "Predictions.csv"
        output_path_confidences = run_output_path / "Confidences.csv"
        output_path_metrics = run_output_path / "Metrics.csv"

        final_predictions_df.to_csv(output_path_predictions, index=False)
        final_confidences_df.to_csv(output_path_confidences, index=False)

        metrics_df = pd.DataFrame(task_metrics).sort_values('Task').reset_index(drop=True)
        metrics_df.to_csv(output_path_metrics, index=False)

        if self.verbose:
            self.printer.print_info(f"Aggregated predictions saved to {output_path_predictions}")
            self.printer.print_info(f"Aggregated confidences saved to {output_path_confidences}")
            self.printer.print_info(f"Task metrics saved to {output_path_metrics}")

        # STEP 7: Drop rows with missing Class
        final_predictions_df = final_predictions_df.dropna(subset=["Class"])
        final_confidences_df = final_confidences_df.dropna(subset=["Class"])

        # STEP 8: Create accuracy_df for return
        accuracy_df = pd.DataFrame([task_accuracies]).reindex(sorted(task_accuracies.keys()), axis=1)

        return final_predictions_df, final_confidences_df, classifier_name, accuracy_df

    
    def _process_classification_tasks(
        self,
        manager: ClassificationManager,
        feature_vector_path: Path,
        seed: int
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
                test_size=self.config.classification.test_size,
                seed=seed
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
        """
        Load prediction and confidence data.
        If any of the chosen ensemble methods requires confidence,
        we also load the confidence data.
        """
        data_loader = DataLoader(self.config)
        predictions_df = data_loader.load_predictions()

        confidence_df = None
        val_acc = None
        # Check if at least one of the methods in self.ensemble_methods is WMV or RK
        if any(method in [EnsembleMethod.WMV, EnsembleMethod.RK, EnsembleMethod.ERK, EnsembleMethod.HC] for method in self.ensemble_methods):
            confidence_df = data_loader.load_confidence()
            val_acc = data_loader.load_validation_accuracies()

        return predictions_df, confidence_df, val_acc

    
    def execute_ensemble_method(
        self,
        method,
        predictions_df: pd.DataFrame,
        confidence_df: Optional[pd.DataFrame],
        accuracy_df: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """Execute the selected ensemble method."""
        method_mapping = {
        EnsembleMethod.MV: lambda: majority_vote.majority_voting(predictions_df, verbose=self.verbose),
        EnsembleMethod.WMV: lambda: self._execute_weighted_majority_voting(predictions_df, accuracy_df),
        EnsembleMethod.RK: lambda: self._execute_ranking(predictions_df, accuracy_df),
        EnsembleMethod.ERK: lambda: self._execute_entropy_ranking(predictions_df, accuracy_df),
        EnsembleMethod.HC: lambda: self._execute_hill_climbing(predictions_df, accuracy_df),
        EnsembleMethod.SA: lambda: self._execute_simulated_annealing(predictions_df, accuracy_df),
        EnsembleMethod.TS: lambda: self._execute_tabu_search(predictions_df, accuracy_df) 
        }
    
        return method_mapping[method]()
    
    def _execute_simulated_annealing(
        self,
        predictions_df: pd.DataFrame,
        accuracy_df: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """Execute the simulated annealing ensemble method."""
        return simulated_annealing.simulated_annealing_combine(predictions_df, accuracy_df, verbose=self.verbose) #TODO: Refactor function
    
    def _execute_tabu_search(
        self,
        predictions_df: pd.DataFrame,
        accuracy_df: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """Execute the simulated annealing ensemble method."""
        return tabu_search.tabu_search_combine(predictions_df, accuracy_df, verbose=self.verbose) #TODO: Refactor function

    def _execute_ranking(
        self,
        predictions_df: pd.DataFrame,
        accuracy_df: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """Execute the ranking ensemble method."""
        if accuracy_df is None:
            return ranking.ranking(predictions_df, verbose=self.verbose)
        else:
            return ranking.ranking(predictions_df, accuracy_df, verbose=self.verbose)    
        
    def _execute_entropy_ranking(
        self,
        predictions_df: pd.DataFrame,
        accuracy_df: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """Execute the entropy ranking ensemble method."""
        ranking = entropy_ranking.EntropyRanking(predictions_df, accuracy_df)
        return ranking.run(diversity_weight=0.5, use_accuracy_weighted_ensemble=True, verbose=self.verbose)
        
    
    def _execute_weighted_majority_voting(
        self,
        predictions_df: pd.DataFrame,
        accuracy_df: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """Special handling for weighted majority voting method."""
        
        if accuracy_df is None:
            return weighted_majority_vote.weighted_majority_voting(
                predictions_df,
                verbose=self.verbose
            )
        else:
            return weighted_majority_vote.weighted_majority_voting(
                predictions_df,
                accuracy_df,
                verbose=self.verbose
            )
    
    def _execute_hill_climbing(
        self,
        predictions_df: pd.DataFrame,
        accuracy_df: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """Execute the hill climbing ensemble method."""
                
        custom_config = hill_climbing.HillClimbingConfig(
            max_iterations=500,
            num_neighbors=100,
            patience=100,
            noise_scale=0.25,
            threshold_range=(0.35, 0.65)
        )
        
        return hill_climbing.hill_climbing_combine(predictions_df, accuracy_df, custom_config, verbose=self.verbose)
            
    
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
            
    def save_results_clf_ensemble(
        self,
        result_df: pd.DataFrame,
        metrics: Dict,
        clf_output: Path,
        run: int,
        combining_method: str
    ) -> None:
        """Save ensemble results and metrics."""
        output_path = Path(clf_output) / f"Output_{combining_method}"
        output_path.mkdir(exist_ok=True)
        
        predictions_file = self._generate_output_filename_classification("predictions", run=run)
        metrics_file = self._generate_output_filename_classification("metrics", run=run)
        
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
    
    def _generate_output_filename_classification(self, suffix: str, run: int) -> str:
        """Generate detailed output filename."""
        parts = [f"ensemble_{self.ensemble_method.value}"]
        
        if self.config.classification.enabled:
            parts.extend([
                f"run_{run + 1}",
            ])
        
        parts.append(suffix)
        return "_".join(parts) + ".csv"