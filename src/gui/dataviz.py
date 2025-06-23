import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QPushButton
)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt

class DataVIz(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data Vizulisation")
        self.setGeometry(100, 100, 600, 500)

        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout()
        self.central_widget.setLayout(self.main_layout)

        # Layout to hold the buttons and labels
        button_layout = QVBoxLayout()

        # First button and label
        self.button1 = QPushButton("Raw Data")
        self.label1 = QLabel("View plot of raw data...")
        self.label1.setAlignment(Qt.AlignCenter)
        layout1 = QVBoxLayout()
        layout1.addWidget(self.button1)
        layout1.addWidget(self.label1)

        # Second button and label
        self.button2 = QPushButton("Frequency Plot")
        self.label2 = QLabel("View the frequency plot...")
        self.label2.setAlignment(Qt.AlignCenter)
        layout2 = QVBoxLayout()
        layout2.addWidget(self.button2)
        layout2.addWidget(self.label2)

        # Third button and label
        self.button3 = QPushButton("Epochs")
        self.label3 = QLabel("View Epochs... they are...")
        self.label3.setAlignment(Qt.AlignCenter)
        layout3 = QVBoxLayout()
        layout3.addWidget(self.button3)
        layout3.addWidget(self.label3)

        # Add individual layouts to the horizontal layout
        button_layout.addLayout(layout1)
        button_layout.addLayout(layout2)
        button_layout.addLayout(layout3)

        # Add the button layout to the main layout
        self.main_layout.addLayout(button_layout)

        # Connect buttons to their respective methods (for now just print)
        self.button1.clicked.connect(self.view_rawdata)
        self.button2.clicked.connect(self.view_frequency)
        self.button3.clicked.connect(self.view_epoch)

    def view_rawdata(self):
        print("Loading data...")

    def view_frequency(self):
        print("frequency...")

    def view_epoch(self):
        print("Epoch..")