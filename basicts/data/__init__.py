import os

from ..utils.misc import scan_modules
from .registry import SCALER_REGISTRY
from .dataset_zoo.simple_tsf_dataset import TimeSeriesForecastingDataset
from .dataset_zoo.m4_dataset import M4ForecastingDataset
from .dataset_zoo.pretrain_tsf_dataset import TimeSeriesReconstractDataset
__all__ = ["SCALER_REGISTRY", "TimeSeriesForecastingDataset", "M4ForecastingDataset",
           "TimeSeriesReconstractDataset"]

# fix bugs on Windows systems and on jupyter
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
scan_modules(project_dir, __file__, ["__init__.py", "registry.py"], ["dataset_zoo/", ".ipynb_checkpoints/"])
