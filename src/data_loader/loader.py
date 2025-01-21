import pandas as pd
from pathlib import Path
from omegaconf import DictConfig

class DataLoader:
    def __init__(self, cfg: DictConfig):
        """
        Initialize the DataLoader with Hydra configuration.
        
        Args:
            cfg (DictConfig): Hydra configuration object
        """
        self.cfg = cfg
        self.base_path = Path(cfg.paths.data)

    def load_predictions(self) -> pd.DataFrame:
        """
        Load predictions data based on configuration.
        
        Returns:
            pd.DataFrame: Loaded predictions data
        """
        file_path = self.base_path / self.cfg.data.predictions.file_name
        encoding = self.cfg.data.predictions.encoding
        separator = self.cfg.data.predictions.separator
        return self._load_data(file_path, encoding, separator)

    def load_confidence(self) -> pd.DataFrame:
        """
        Load confidence data based on configuration.
        
        Returns:
            pd.DataFrame: Loaded confidence data
        """
        file_path = self.base_path / self.cfg.data.confidence.file_name
        encoding = self.cfg.data.confidence.encoding
        separator = self.cfg.data.confidence.separator
        return self._load_data(file_path, encoding, separator)

    def load_validation_accuracies(self) -> pd.DataFrame:
        """
        Load validation accuracies data based on configuration.
        
        Returns:
            pd.DataFrame: Loaded validation accuracies data
        """
        file_path = self.base_path / self.cfg.data.validation_acc.filename
        encoding = self.cfg.data.validation_acc.encoding
        separator = self.cfg.data.validation_acc.separator
        
        try:
            # Read the CSV file
            df = pd.read_csv(file_path, encoding=encoding, sep=separator, nrows=1)
            return df
            
        except Exception as e:
            raise Exception(f"Error loading validation accuracies from {file_path}: {str(e)}")

    def _load_data(self, file_path: Path, encoding: str = 'utf-8', separator: str = ',') -> pd.DataFrame:
        """
        Internal method to load data with specified parameters.
        
        Args:
            file_path (Path): Path to the data file
            encoding (str): File encoding
            separator (str): CSV separator character
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            return pd.read_csv(file_path, encoding=encoding, sep=separator)
        except Exception as e:
            raise Exception(f"Error loading data from {file_path}: {str(e)}")