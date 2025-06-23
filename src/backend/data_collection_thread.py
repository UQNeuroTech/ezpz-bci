from PySide6.QtCore import QThread, Signal
from src.backend.collect_data_openbci import main as collect_data_main
from brainflow.board_shim import BoardIds

# Thread class to run data collection in background
class DataCollectionThread(QThread):
    finished = Signal()
    error = Signal(str)
    update_marker = Signal(str)

    def __init__(self):
        super().__init__()
        self.parent_widget = None
        self.is_running = True

    def set_parent_widget(self, widget):
        self.parent_widget = widget

    def set_board_id(self, board_id):
        # if 
        # self.board_id = board_id
        self.board_id = BoardIds.CYTON_BOARD  # Default to CYTON_BOARD, can be set later

    def update_ui(self, marker_text):
        # Emit signal to update UI
        self.update_marker.emit(marker_text)

    def stop(self):
        """Signal the thread to stop"""
        self.is_running = False

    def run(self):
        try:
            collect_data_main(self.update_ui, lambda: self.is_running, self.board_id)
            if self.is_running:
                self.finished.emit()
            else:
                print("Thread stopped by user request")
        except Exception as e:
            self.error.emit(str(e))
