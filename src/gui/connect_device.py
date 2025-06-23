import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QPushButton, QTextEdit, QComboBox, QSpacerItem, QSizePolicy
)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt

class Connect(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Connection")
        self.setGeometry(100, 100, 600, 500)

        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout()
        self.central_widget.setLayout(self.main_layout)

        # Top section with title and controls
        top_layout = QVBoxLayout()
        
        # Title label
        title_label = QLabel("Select Device and Enter Port")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        top_layout.addWidget(title_label)

        # Layout to hold the buttons and labels
        config_layout = QHBoxLayout()

        # First comboBox (Device)
        self.comboBox = QComboBox()
        self.comboBox.addItem("OpenBCI")
        self.comboBox.addItem("Neurosity")
        self.boxLabel = QLabel("Device")
        self.boxLabel.setAlignment(Qt.AlignLeft)
        self.comboBox.setStyleSheet("font-size: 14px; padding: 5px;")
        layout1 = QVBoxLayout()
        layout1.addWidget(self.boxLabel)
        layout1.addWidget(self.comboBox)


        # Second textEdit (Port)
        self.port = QTextEdit()
        self.port.setFixedHeight(50)  # Set the height of the port box to be small
        self.port.setPlaceholderText("Enter port...")
        self.port.setStyleSheet("font-size: 14px; padding: 5px;")
        self.portlabel = QLabel("Port")
        self.portlabel.setAlignment(Qt.AlignLeft)
        layout2 = QVBoxLayout()
        layout2.addWidget(self.portlabel)
        layout2.addWidget(self.port)

        # Third button (Connect)
        self.button1 = QPushButton("Connect")
        self.button1.setStyleSheet("background-color: #4CAF50; color: white; font-size: 14px; padding: 10px; border-radius: 5px;")
        layout3 = QVBoxLayout()
        layout3.addWidget(self.button1)

        # Add individual layouts to the horizontal layout
        config_layout.addLayout(layout1)
        config_layout.addLayout(layout2)
        config_layout.addLayout(layout3)

        # Set the stretch factor of each layout to make them expand only vertically (not horizontally)
        config_layout.setStretchFactor(layout1, 1)
        config_layout.setStretchFactor(layout2, 1)
        config_layout.setStretchFactor(layout3, 0)

        # Add the button layout to the top section layout
        top_layout.addLayout(config_layout)

        # Add top section layout to the main layout
        self.main_layout.addLayout(top_layout)

        # Spacer to fill the rest of the space (for blank area)
        self.spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.main_layout.addItem(self.spacer)

        # Connect buttons to their respective methods
        self.button1.clicked.connect(self.connect_headset)

    def connect_headset(self):
        # Get the port from the QTextEdit and print it
        port_value = self.port.toPlainText()
        if port_value:
            print(f"Connecting to port: {port_value}")
        else:
            print("No port entered!")