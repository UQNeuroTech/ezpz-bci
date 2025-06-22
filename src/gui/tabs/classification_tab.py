"""
Classification Tab - Real-time EEG classification interface
"""
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QGroupBox, QProgressBar,
    QComboBox, QTextEdit, QSlider, QCheckBox,
    QFileDialog, QMessageBox, QTableWidget, QTableWidgetItem
)
from PySide6.QtGui import QFont, QPalette

# Import the RealTimeClassifier worker
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'workers'))
from real_time_classifier import RealTimeClassifier

class ClassificationTab(QWidget):
    def __init__(self):
        super().__init__()
        self.is_classifying = False
        self.model_loaded = False
        self.device_connected = False
        
        # Initialize the real-time classifier worker
        self.classifier = RealTimeClassifier()
        self.connect_worker_signals()
        
        # Statistics tracking
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self.update_statistics_display)
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the classification tab UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Header
        header = self.create_header()
        layout.addWidget(header)
        
        # Setup section
        setup_section = self.create_setup_section()
        layout.addWidget(setup_section)
        
        # Control section
        control_section = self.create_control_section()
        layout.addWidget(control_section)
        
        # Real-time display section
        display_section = self.create_display_section()
        layout.addWidget(display_section)
        
        # Status and log section
        status_section = self.create_status_section()
        layout.addWidget(status_section)
        
        layout.addStretch()
        
    def create_header(self):
        """Create header section"""
        header_widget = QWidget()
        layout = QVBoxLayout(header_widget)
        
        title = QLabel("🧠 Live EEG Classification")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title.setFont(title_font)
        
        description = QLabel(
            "Real-time classification of EEG signals into computer commands.\n"
            "Load a trained model and start controlling your computer with your thoughts!"
        )
        description.setStyleSheet("color: #666; font-size: 12px;")
        
        layout.addWidget(title)
        layout.addWidget(description)
        
        return header_widget
        
    def create_setup_section(self):
        """Create setup and configuration section"""
        group = QGroupBox("Setup & Configuration")
        layout = QGridLayout(group)
        
        # Model selection
        layout.addWidget(QLabel("Trained Model:"), 0, 0)
        self.model_path_label = QLabel("No model loaded")
        self.model_path_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self.model_path_label, 0, 1)
        
        self.load_model_btn = QPushButton("📁 Load Model")
        self.load_model_btn.clicked.connect(self.load_model)
        layout.addWidget(self.load_model_btn, 0, 2)
        
        # Device selection
        layout.addWidget(QLabel("EEG Device:"), 1, 0)
        self.device_combo = QComboBox()
        self.device_combo.addItems(["OpenBCI Cyton", "Neurosity Crown", "Synthetic (Demo)"])
        layout.addWidget(self.device_combo, 1, 1)
        
        self.connect_btn = QPushButton("🔌 Connect")
        self.connect_btn.clicked.connect(self.connect_device)
        layout.addWidget(self.connect_btn, 1, 2)
        
        # Prediction settings
        layout.addWidget(QLabel("Confidence Threshold:"), 2, 0)
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(50, 95)
        self.confidence_slider.setValue(75)
        self.confidence_slider.valueChanged.connect(self.update_confidence_label)
        layout.addWidget(self.confidence_slider, 2, 1)
        
        self.confidence_label = QLabel("75%")
        layout.addWidget(self.confidence_label, 2, 2)
        
        # Prediction smoothing
        layout.addWidget(QLabel("Prediction Smoothing:"), 3, 0)
        self.smoothing_checkbox = QCheckBox("Enable (reduces false positives)")
        self.smoothing_checkbox.setChecked(True)
        layout.addWidget(self.smoothing_checkbox, 3, 1, 1, 2)
        
        return group
        
    def create_control_section(self):
        """Create classification control section"""
        group = QGroupBox("Classification Control")
        layout = QHBoxLayout(group)
        
        # System status indicators
        status_layout = QVBoxLayout()
        
        self.model_status = QLabel("❌ Model: Not Loaded")
        self.model_status.setStyleSheet("color: #d32f2f;")
        status_layout.addWidget(self.model_status)
        
        self.device_status = QLabel("❌ Device: Not Connected")
        self.device_status.setStyleSheet("color: #d32f2f;")
        status_layout.addWidget(self.device_status)
        
        layout.addLayout(status_layout)
        
        # Control buttons
        self.start_btn = QPushButton("▶️ Start Classification")
        self.start_btn.clicked.connect(self.start_classification)
        self.start_btn.setMinimumHeight(50)
        self.start_btn.setEnabled(False)
        layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("⏹️ Stop Classification")
        self.stop_btn.clicked.connect(self.stop_classification)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setMinimumHeight(50)
        layout.addWidget(self.stop_btn)
        
        # Practice mode
        self.practice_btn = QPushButton("🎯 Practice Mode")
        self.practice_btn.clicked.connect(self.toggle_practice_mode)
        self.practice_btn.setToolTip("Practice mode shows predictions without executing actions")
        layout.addWidget(self.practice_btn)
        
        return group
        
    def create_display_section(self):
        """Create real-time display section"""
        group = QGroupBox("Real-time Classification")
        layout = QVBoxLayout(group)
        
        # Current prediction display
        pred_layout = QHBoxLayout()
        
        # Large prediction display
        self.prediction_label = QLabel("Waiting...")
        self.prediction_label.setAlignment(Qt.AlignCenter)
        pred_font = QFont()
        pred_font.setPointSize(28)
        pred_font.setBold(True)
        self.prediction_label.setFont(pred_font)
        self.prediction_label.setMinimumHeight(100)
        self.prediction_label.setStyleSheet("""
            QLabel {
                border: 3px solid #ddd;
                border-radius: 10px;
                background-color: #f5f5f5;
                padding: 20px;
            }
        """)
        pred_layout.addWidget(self.prediction_label)
        
        # Confidence meter
        conf_layout = QVBoxLayout()
        conf_layout.addWidget(QLabel("Confidence"))
        
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        self.confidence_bar.setOrientation(Qt.Vertical)
        self.confidence_bar.setMinimumHeight(100)
        conf_layout.addWidget(self.confidence_bar)
        
        self.confidence_value = QLabel("0%")
        self.confidence_value.setAlignment(Qt.AlignCenter)
        conf_layout.addWidget(self.confidence_value)
        
        pred_layout.addLayout(conf_layout)
        layout.addLayout(pred_layout)
        
        # Class probabilities table
        self.prob_table = QTableWidget(3, 2)
        self.prob_table.setHorizontalHeaderLabels(["Class", "Probability"])
        self.prob_table.setMaximumHeight(120)
        
        # Initialize table
        classes = ["Rest", "Left Fist", "Right Fist"]
        for i, class_name in enumerate(classes):
            self.prob_table.setItem(i, 0, QTableWidgetItem(class_name))
            self.prob_table.setItem(i, 1, QTableWidgetItem("0.00%"))
            
        layout.addWidget(self.prob_table)
        
        return group
        
    def create_status_section(self):
        """Create status and activity log section"""
        group = QGroupBox("Activity & Status")
        layout = QVBoxLayout(group)
        
        # Statistics
        stats_layout = QGridLayout()
        
        stats_layout.addWidget(QLabel("Predictions Made:"), 0, 0)
        self.pred_count_label = QLabel("0")
        stats_layout.addWidget(self.pred_count_label, 0, 1)
        
        stats_layout.addWidget(QLabel("Actions Executed:"), 0, 2)
        self.action_count_label = QLabel("0")
        stats_layout.addWidget(self.action_count_label, 0, 3)
        
        stats_layout.addWidget(QLabel("Classification Rate:"), 1, 0)
        self.rate_label = QLabel("0 Hz")
        stats_layout.addWidget(self.rate_label, 1, 1)
        
        stats_layout.addWidget(QLabel("Session Time:"), 1, 2)
        self.session_time_label = QLabel("00:00")
        stats_layout.addWidget(self.session_time_label, 1, 3)
        
        layout.addLayout(stats_layout)
        
        # Activity log
        self.activity_log = QTextEdit()
        self.activity_log.setMaximumHeight(100)
        self.activity_log.setReadOnly(True)
        self.activity_log.append("Ready to start classification...")
        layout.addWidget(self.activity_log)
        
        return group
        
    def load_model(self):
        """Load a trained EEGNet model"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Trained Model",
            "",
            "PyTorch Models (*.pth);;All Files (*)"
        )
        
        if file_path:
            self.classifier.set_model_path(file_path)
            self.classifier.load_model()
            self.log_activity(f"Loading model: {file_path.split('/')[-1]}...")
                
    def connect_device(self):
        """Connect to EEG device"""
        device = self.device_combo.currentText()
        
        if not self.device_connected:
            self.log_activity(f"Connecting to {device}...")
            self.classifier.set_device_type(device)
            self.classifier.connect_device()
        else:
            # Disconnect
            self.device_connected = False
            self.device_status.setText("❌ Device: Not Connected")
            self.device_status.setStyleSheet("color: #d32f2f;")
            self.connect_btn.setText("🔌 Connect")
            self.log_activity("🔌 Device disconnected")
            self.update_start_button()
            
    def update_start_button(self):
        """Update start button availability"""
        self.start_btn.setEnabled(self.model_loaded and self.device_connected)
        
    def start_classification(self):
        """Start real-time classification"""
        if not self.is_classifying:
            self.is_classifying = True
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            
            self.prediction_label.setText("Classifying...")
            self.prediction_label.setStyleSheet("""
                QLabel {
                    border: 3px solid #4caf50;
                    border-radius: 10px;
                    background-color: #e8f5e8;
                    padding: 20px;
                    color: #2e7d32;
                }
            """)
            
            self.log_activity("🔴 Real-time classification started!")
            
            # Configure classifier settings
            confidence_threshold = self.confidence_slider.value()
            self.classifier.set_confidence_threshold(confidence_threshold)
            self.classifier.set_smoothing_enabled(self.smoothing_checkbox.isChecked())
            
            # Start classification
            self.classifier.start_classification()
            
            # Start statistics timer
            self.stats_timer.start(1000)  # Update every second
            
    def stop_classification(self):
        """Stop real-time classification"""
        if self.is_classifying:
            self.is_classifying = False
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            
            self.prediction_label.setText("Stopped")
            self.prediction_label.setStyleSheet("""
                QLabel {
                    border: 3px solid #ddd;
                    border-radius: 10px;
                    background-color: #f5f5f5;
                    padding: 20px;
                    color: #333;
                }
            """)
            
            self.log_activity("⏹️ Classification stopped")
            
            # Stop classifier and timer
            self.classifier.stop_classification()
            self.stats_timer.stop()
            
    def toggle_practice_mode(self):
        """Toggle practice mode"""
        # TODO: Implement practice mode toggle
        self.log_activity("🎯 Practice mode toggled")
        
    def update_confidence_label(self, value):
        """Update confidence threshold label"""
        self.confidence_label.setText(f"{value}%")
        
    def update_prediction(self, prediction, confidence, probabilities):
        """Update the prediction display"""
        self.prediction_label.setText(prediction)
        self.confidence_bar.setValue(int(confidence * 100))
        self.confidence_value.setText(f"{confidence:.1%}")
        
        # Update probabilities table
        for i, (label, prob) in enumerate(probabilities):
            self.prob_table.setItem(i, 1, QTableWidgetItem(f"{prob:.2%}"))
            
        # Color code prediction based on confidence
        if confidence > 0.8:
            color = "#4caf50"  # Green
            bg_color = "#e8f5e8"
        elif confidence > 0.6:
            color = "#ff9800"  # Orange
            bg_color = "#fff3e0"
        else:
            color = "#f44336"  # Red
            bg_color = "#ffebee"
            
        self.prediction_label.setStyleSheet(f"""
            QLabel {{
                border: 3px solid {color};
                border-radius: 10px;
                background-color: {bg_color};
                padding: 20px;
                color: {color};
            }}
        """)
        
    def log_activity(self, message):
        """Add message to activity log"""
        self.activity_log.append(f"• {message}")
        
    def update_statistics(self, pred_count, action_count, rate, session_time):
        """Update classification statistics"""
        self.pred_count_label.setText(str(pred_count))
        self.action_count_label.setText(str(action_count))
        self.rate_label.setText(f"{rate:.1f} Hz")
        
        # Format session time
        minutes = int(session_time // 60)
        seconds = int(session_time % 60)
        self.session_time_label.setText(f"{minutes:02d}:{seconds:02d}")
        
    def connect_worker_signals(self):
        """Connect signals from the real-time classifier worker"""
        self.classifier.prediction_updated.connect(self.on_prediction_updated)
        self.classifier.status_logged.connect(self.log_activity)
        self.classifier.classification_started.connect(self.on_classification_started)
        self.classifier.classification_stopped.connect(self.on_classification_stopped)
        self.classifier.device_connected.connect(self.on_device_connected)
        self.classifier.model_loaded.connect(self.on_model_loaded)
        
    def on_prediction_updated(self, prediction, confidence, probabilities):
        """Handle prediction updates from the classifier"""
        self.update_prediction(prediction, confidence, probabilities)
        
    def on_classification_started(self):
        """Handle classification start"""
        self.log_activity("✅ Classification thread started")
        
    def on_classification_stopped(self):
        """Handle classification stop"""
        self.log_activity("⏹️ Classification thread stopped")
        
    def on_device_connected(self, connected):
        """Handle device connection status changes"""
        self.device_connected = connected
        if connected:
            self.device_status.setText("✅ Device: Connected")
            self.device_status.setStyleSheet("color: #388e3c;")
            self.connect_btn.setText("🔌 Disconnect")
            self.log_activity("✅ Device connected successfully")
        else:
            self.device_status.setText("❌ Device: Not Connected")
            self.device_status.setStyleSheet("color: #d32f2f;")
            self.connect_btn.setText("🔌 Connect")
            self.log_activity("❌ Device connection failed")
        self.update_start_button()
        
    def on_model_loaded(self, success, message):
        """Handle model loading completion"""
        if success:
            self.model_path_label.setText(message.split(': ')[-1])
            self.model_path_label.setStyleSheet("color: #333;")
            self.model_loaded = True
            self.model_status.setText("✅ Model: Loaded")
            self.model_status.setStyleSheet("color: #388e3c;")
            self.log_activity(f"✅ {message}")
        else:
            self.model_loaded = False
            self.model_status.setText("❌ Model: Not Loaded")
            self.model_status.setStyleSheet("color: #d32f2f;")
            self.log_activity(f"❌ {message}")
            QMessageBox.warning(self, "Model Loading Failed", message)
        self.update_start_button()
        
    def update_statistics_display(self):
        """Update the statistics display periodically"""
        pred_count, action_count, rate, session_time = self.classifier.get_statistics()
        self.update_statistics(pred_count, action_count, rate, session_time)