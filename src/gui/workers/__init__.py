# Workers package for ezpz-BCI GUI
# Background worker threads for EEG processing

from .eeg_data_collector import EEGDataCollector
from .model_trainer import ModelTrainer
from .real_time_classifier import RealTimeClassifier

__all__ = ['EEGDataCollector', 'ModelTrainer', 'RealTimeClassifier']