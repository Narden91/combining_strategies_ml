# src/data_loader/loader.py

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
        return self._load_data(file_path)

    def load_confidence(self) -> pd.DataFrame:
        """
        Load confidence data based on configuration.
        
        Returns:
            pd.DataFrame: Loaded confidence data
        """
        file_path = self.base_path / self.cfg.data.confidence.file_name
        return self._load_data(file_path)

    def _load_data(self, file_path: Path) -> pd.DataFrame:
        """
        Internal method to load data with default parameters.
        
        Args:
            file_path (Path): Path to the data file
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise Exception(f"Error loading data from {file_path}: {str(e)}")