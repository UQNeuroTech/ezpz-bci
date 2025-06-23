from PySide6.QtCore import QThread, Signal
from src.backend.train_eegnet import train
from src.backend.process_openbci_data import load_openbci_data, convert_to_mne


class TrainingThread(QThread):
    finished = Signal()
    error = Signal(str)

    def run(self):
        try:
            samples, markers = load_openbci_data("./data", False, False)
            convert_to_mne("mne_data", "mne_data", "./data", samples, markers)
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