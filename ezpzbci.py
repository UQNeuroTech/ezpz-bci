import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QLineEdit, QLabel
from PyQt5.QtCore import QTimer, Qt

class CountdownApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Training")
        self.setGeometry(100, 100, 400, 300)

        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # Input box and run button
        self.input_box = QLineEdit(self)
        self.input_box.setPlaceholderText("Enter text here")
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

        # Connect button to action
        self.run_button.clicked.connect(self.start_countdown)

    def start_countdown(self):
        """Start the countdown."""
        self.countdown_value = 2
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
                self.phase = "Rest2"
                self.heading_label.setText("Rest!")
            elif self.phase == "Rest2":
                self.phase = "Exert"
                self.heading_label.setText("Prepare to Exert...")
            elif self.phase == "Exert":
                self.phase = "Exert2"
                self.heading_label.setText("Exert!")
            elif self.phase == "Exert2":
                self.heading_label.setText("Done.")
                self.countdown_label.setText("")
                self.timer.stop()
                return

            # Reset countdown for the next phase
            self.countdown_value = 2
            self.countdown_label.setText(f"{self.countdown_value}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CountdownApp()
    window.show()
    sys.exit(app.exec_())
