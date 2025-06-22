import sys
import os
import json
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
    QPushButton, QLineEdit, QLabel, QMessageBox, QSpacerItem, QSizePolicy
)
from PySide6.QtCore import Qt

from src.backend.data_collection_thread import DataCollectionThread


class TrainingPage(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Training")
        self.setGeometry(100, 100, 400, 300)

        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Main layout with margins and spacing
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(20, 20, 20, 20)  # Set margins (left, top, right, bottom)
        self.layout.setSpacing(10)  # Set spacing between widgets
        self.central_widget.setLayout(self.layout)

        # Input fields for number of epochs and learning rate
        self.epochs_label = QLabel("Number of Epochs:", self)
        self.epochs_input = QLineEdit(self)
        self.epochs_input.setPlaceholderText("Enter number of epochs")

        self.learning_rate_label = QLabel("Learning Rate:", self)
        self.learning_rate_input = QLineEdit(self)
        self.learning_rate_input.setPlaceholderText("Enter learning rate")

        # Train button
        self.train_button = QPushButton("Train", self)
        self.train_button.setStyleSheet("background-color: red; color: white; font-size: 16px;")
        self.train_button.setCheckable(True)  # Make the button toggleable
        self.train_button.clicked.connect(self.toggle_train_button)

        # Status label for showing current action
        self.heading_label = QLabel("Ready", self)
        self.heading_label.setStyleSheet("font-size: 18px;")
        self.heading_label.setAlignment(Qt.AlignCenter)

        # Add widgets to the main layout with alignment to the top
        self.layout.addWidget(self.epochs_label, alignment=Qt.AlignTop)
        self.layout.addWidget(self.epochs_input, alignment=Qt.AlignTop)
        self.layout.addWidget(self.learning_rate_label, alignment=Qt.AlignTop)
        self.layout.addWidget(self.learning_rate_input, alignment=Qt.AlignTop)
        self.layout.addWidget(self.heading_label, alignment=Qt.AlignTop)

        # Add vertical spacer
        spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.layout.addItem(spacer)

        self.layout.addWidget(self.train_button, alignment=Qt.AlignTop)

        # Data collection thread
        self.data_collection_thread = None

    def toggle_train_button(self):
        """Toggle the train button between on (green) and off (red)."""
        if self.train_button.isChecked():
            self.train_button.setStyleSheet("background-color: green; color: white; font-size: 16px;")
            self.train_button.setText("Training...")
            self.heading_label.setText("Ready to collect data...")
            self.save_to_json()  # Save values to JSON when training starts
            self.start_data_collection() # Start data collection in a separate thread
        else:
            self.train_button.setStyleSheet("background-color: red; color: white; font-size: 16px;")
            self.train_button.setText("Train")
            self.heading_label.setText("Stopping data collection...")
            # Signal the thread to stop
            if self.data_collection_thread and self.data_collection_thread.isRunning():
                self.data_collection_thread.stop()
                self.heading_label.setText("Data collection stopping...")

    def save_to_json(self):
        """Save textbox values to db/categories.json."""
        epochs = self.epochs_input.text().strip()
        learning_rate = self.learning_rate_input.text().strip()

        # Validate inputs
        try:
            epochs = int(epochs)  # Convert to integer
            learning_rate = float(learning_rate)  # Convert to float
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter valid numbers for epochs and learning rate.")
            return

        # Ensure the directory exists
        os.makedirs("./db", exist_ok=True)
        json_file_path = "./src/gui/db/categories.json"

        # Load existing data if the file exists
        if os.path.exists(json_file_path):
            with open(json_file_path, "r") as json_file:
                try:
                    data = json.load(json_file)
                    if not isinstance(data, dict):  # Ensure it's a dictionary
                        data = {}
                except json.JSONDecodeError:
                    data = {}
        else:
            data = {}

        # Add or update key-value pairs
        data["epoch_count"] = epochs
        data["learning_rate"] = learning_rate

        # Save updated data back to the file
        with open(json_file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)
        print(f"Data saved to {json_file_path}: {data}")

    def start_data_collection(self):
        """Start data collection in a separate thread."""
        if self.data_collection_thread is not None and self.data_collection_thread.isRunning():
            return  # Already running

        self.data_collection_thread = DataCollectionThread()
        self.data_collection_thread.set_parent_widget(self)
        self.data_collection_thread.finished.connect(self.on_data_collection_finished)
        self.data_collection_thread.error.connect(self.on_data_collection_error)
        self.data_collection_thread.update_marker.connect(self.update_marker_display)
        self.data_collection_thread.start()

    def on_data_collection_finished(self):
        """Handle successful completion of data collection."""
        print("Data collection completed successfully")
        # Reset button state if it's still in checked state
        if self.train_button.isChecked():
            self.train_button.setChecked(False)
            self.train_button.setStyleSheet("background-color: red; color: white; font-size: 16px;")
            self.train_button.setText("Train")
        self.heading_label.setText("Ready")
        self.data_collection_thread = None

    def on_data_collection_error(self, error_msg):
        """Handle error in data collection."""
        QMessageBox.critical(self, "Data Collection Error", f"An error occurred during data collection: {error_msg}")

    def update_marker_display(self, marker_text):
        """Update the heading label with the current marker text."""
        self.heading_label.setText(f"Current action: {marker_text}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TrainingPage()
    window.show()
    sys.exit(app.exec_())
