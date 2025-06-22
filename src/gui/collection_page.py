import sys
import os
import json
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QLineEdit, QLabel
from PySide6.QtCore import QTimer, Qt

class CountdownApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Collection")
        self.setGeometry(100, 100, 1000, 300)

        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # Input box for category
        self.input_box = QLineEdit(self)
        self.input_box.setPlaceholderText("Enter category")

        # Input box for number of cycles
        self.cycles_box = QLineEdit(self)
        self.cycles_box.setPlaceholderText("Number of training cycles")

        # Input box for cycle duration
        self.duration_box = QLineEdit(self)
        self.duration_box.setPlaceholderText("Cycle duration")

        # Run button
        self.run_button = QPushButton("Run", self)

        # Heading field for instructions
        self.heading_label = QLabel("Press 'Run' to start", self)
        self.heading_label.setStyleSheet("font-size: 24px; text-align: center;")
        self.heading_label.setAlignment(Qt.AlignCenter)

        # Countdown display
        self.countdown_label = QLabel("", self)
        self.countdown_label.setStyleSheet("font-size: 48px; text-align: center;")
        self.countdown_label.setAlignment(Qt.AlignCenter)

        # Add widgets to layout
        input_layout = QHBoxLayout()
        input_layout.addWidget(self.input_box)
        input_layout.addWidget(self.cycles_box)
        input_layout.addWidget(self.duration_box)
        input_layout.addWidget(self.run_button)
        self.layout.addLayout(input_layout)
        self.layout.addWidget(self.heading_label)
        self.layout.addWidget(self.countdown_label)

        # Timer setup
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_countdown)
        self.countdown_value = 2
        self.phase = "Rest"  # Initial phase
        self.phase_count = 0  # Track the number of completed phases
        self.total_cycles = 0  # Number of cycles
        self.cycle_duration = 2  # Duration of each cycle

        # Connect button to action
        self.run_button.clicked.connect(self.save_to_json_and_start_countdown)

    def save_to_json_and_start_countdown(self):
        """Save text box contents to JSON and start the countdown."""
        textbox_contents = self.input_box.text().strip()
        if textbox_contents:
            # Ensure the directory exists
            json_file_path = "./src/gui/db/categories.json"
    
            # Load existing data if the file exists
            if os.path.exists(json_file_path):
                with open(json_file_path, "r") as json_file:
                    try:
                        data = json.load(json_file)
                        if not isinstance(data, dict):  # Ensure it's a dictionary
                            data = {"cycle_count": 0, "cycle_duration": 0, "categories": []}
                    except json.JSONDecodeError:
                        data = {"cycle_count": 0, "cycle_duration": 0, "categories": []}
            else:
                data = {"cycle_count": 0, "cycle_duration": 0, "categories": []}
    
            # Ensure the "categories" key exists
            if "categories" not in data:
                data["categories"] = []
    
            # Add the category only if it doesn't already exist
            if textbox_contents not in data["categories"]:
                data["categories"].append(textbox_contents)
    
            # Update cycle count and duration
            try:
                data["cycle_count"] = int(self.cycles_box.text().strip())
                data["cycle_duration"] = int(self.duration_box.text().strip())
            except ValueError:
                self.heading_label.setText("Invalid input for cycles or duration!")
                return
    
            # Save updated data back to the file
            try:
                with open(json_file_path, "w") as json_file:
                    json.dump(data, json_file, indent=4)
                print(f"Data successfully saved to {json_file_path}: {data}")  # Debugging output
            except Exception as e:
                print(f"Error saving data to {json_file_path}: {e}")  # Debugging output
    
        # Start the countdown
        self.start_countdown()

    def start_countdown(self):
        """Start the countdown."""
        self.countdown_value = self.cycle_duration
        self.phase = "Rest"  # Start with the "Rest" phase
        self.phase_count = 0  # Reset phase count
        self.heading_label.setText("Prepare to Rest...")
        self.countdown_label.setText(f"{self.countdown_value}")
        self.timer.start(1000)  # Update every second

    def update_countdown(self):
        """Update the countdown display."""
        self.countdown_value -= 1
        if self.countdown_value >= 0:
            self.countdown_label.setText(f"{self.countdown_value}")
        else:
            # Switch phases after countdown ends
            if self.phase == "Rest":
                self.phase = "Exert"
                self.heading_label.setText("Prepare to Exert...")
            elif self.phase == "Exert":
                self.phase_count += 1
                if self.phase_count >= self.total_cycles:  # End after completing all cycles
                    self.heading_label.setText("Done.")
                    self.countdown_label.setText("")
                    self.timer.stop()
                    return
                self.phase = "Rest"
                self.heading_label.setText("Prepare to Rest...")

            # Reset countdown for the next phase
            self.countdown_value = self.cycle_duration
            self.countdown_label.setText(f"{self.countdown_value}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CountdownApp()
    window.show()
    sys.exit(app.exec_())
