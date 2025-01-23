from enum import Enum
from typing import Callable, Dict, Tuple, List, Optional
from pathlib import Path
import pandas as pd
import numpy as np
from omegaconf import DictConfig

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
    
    def run_classification(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Execute the classification phase if enabled."""
        if self.verbose:
            self.printer.print_info("Starting classification phase...")
            
        feature_vector_path = Path(self.config.paths.data) / self.config.data.feature_vector.folder
        classification_output = Path(self.config.paths.output) / "classification_output"
        
        manager = ClassificationManager(
            feature_vector_path,
            classification_output,
            n_runs=self.config.classification.n_runs,
            base_seed=self.config.classification.base_seed
        )
        
        tasks = self._process_classification_tasks(manager, feature_vector_path)
        
        return self._aggregate_classification_results(
            classification_output,
            tasks,
            manager
        )
    
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
        
        # Get classifier instance to access its name
        classifier = get_classifier(
            self.config.classification.classifier,
            tasks[0],  # Use first task to get classifier type
            output_dir,
            self.config.classification.base_seed
        )
        classifier_name = classifier.get_classifier_name()
        
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
        
        # Get true labels using classifier name instead of manager
        true_labels = np.load(
            output_dir / classifier_name / 
            f"task_{tasks[0]}/run_0/true_labels.npy"
        )
        
        return self._create_result_dataframes(
            all_predictions,
            all_probabilities,
            true_labels,
            len(tasks)
        )
    
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