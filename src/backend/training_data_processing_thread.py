from PySide6.QtCore import QThread, Signal
from src.backend.process_openbci_data import load_openbci_data, convert_to_mne


class TrainingThread(QThread):
    finished = Signal()
    error = Signal(str)

    def run(self):
        try:
            samples, markers = load_openbci_data("./data")
            convert_to_mne("mne_data", "mne_data", "./data", samples, markers)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))