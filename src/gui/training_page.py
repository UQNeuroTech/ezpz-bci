import sys
import os
import json
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
    QPushButton, QLineEdit, QLabel, QMessageBox
)
from PySide6.QtCore import Qt

class TrainingPage(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Training Page")
        self.setGeometry(100, 100, 400, 200)

        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
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

        # Layout for input fields
        input_layout = QVBoxLayout()
        input_layout.addWidget(self.epochs_label)
        input_layout.addWidget(self.epochs_input)
        input_layout.addWidget(self.learning_rate_label)
        input_layout.addWidget(self.learning_rate_input)

        # Add widgets to the main layout
        self.layout.addLayout(input_layout)
        self.layout.addWidget(self.train_button)

    def toggle_train_button(self):
        """Toggle the train button between on (green) and off (red)."""
        if self.train_button.isChecked():
            self.train_button.setStyleSheet("background-color: green; color: white; font-size: 16px;")
            self.train_button.setText("Training...")
            self.save_to_json()  # Save values to JSON when training starts
        else:
            self.train_button.setStyleSheet("background-color: red; color: white; font-size: 16px;")
            self.train_button.setText("Train")

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
        json_file_path = "./db/categories.json"

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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TrainingPage()
    window.show()
    sys.exit(app.exec_())
