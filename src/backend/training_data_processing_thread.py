from PySide6.QtCore import QThread, Signal
from src.backend.train_eegnet import train


class TrainingThread(QThread):
    finished = Signal()
    error = Signal(str)

    def run(self):
        try:
            hyperparameters = {
                "epochs": 200,
                "test-ratio": 0.3
            }

            name = "ezpz-model"
            load_path = "./data/ezpz-epochs-epo.fif"
            save_path_folder = "./data"
            train(name, load_path, save_path_folder, hyperparameters, save=True)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))