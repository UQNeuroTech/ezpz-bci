"""
Data Collection Tab - Interface for collecting EEG training data
"""
import sys
import os

# Add parent directory to path for worker imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QGroupBox, QProgressBar,
    QSpinBox, QComboBox, QTextEdit, QFrame,
    QMessageBox, QSlider
)
from PySide6.QtGui import QFont, QPixmap

# Import the EEG data collection worker
from workers.eeg_data_collector import EEGDataCollector

class DataCollectionTab(QWidget):
    def __init__(self):
        super().__init__()
        self.is_collecting = False
        self.current_prompt = "None"
        
        # Initialize the EEG data collection worker
        self.data_collector = EEGDataCollector()
        self.connect_worker_signals()
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the data collection tab UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Header
        header = self.create_header()
        layout.addWidget(header)
        
        # Configuration section
        config_section = self.create_config_section()
        layout.addWidget(config_section)
        
        # Collection controls
        controls_section = self.create_controls_section()
        layout.addWidget(controls_section)
        
        # Visual prompt display
        prompt_section = self.create_prompt_section()
        layout.addWidget(prompt_section)
        
        # Progress and status
        status_section = self.create_status_section()
        layout.addWidget(status_section)
        
        layout.addStretch()
        
    def create_header(self):
        """Create header section"""
        header_widget = QWidget()
        layout = QVBoxLayout(header_widget)
        
        title = QLabel("📊 EEG Data Collection")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title.setFont(title_font)
        
        description = QLabel(
            "Collect EEG training data by following the visual prompts.\n"
            "Perform the indicated mental tasks when prompted."
        )
        description.setStyleSheet("color: #666; font-size: 12px;")
        
        layout.addWidget(title)
        layout.addWidget(description)
        
        return header_widget
        
    def create_config_section(self):
        """Create configuration section"""
        group = QGroupBox("Collection Settings")
        layout = QGridLayout(group)
        
        # Device selection
        layout.addWidget(QLabel("EEG Device:"), 0, 0)
        self.device_combo = QComboBox()
        self.device_combo.addItems(["OpenBCI Cyton", "Neurosity Crown", "Synthetic (Demo)"])
        layout.addWidget(self.device_combo, 0, 1)
        
        # Number of trials per class
        layout.addWidget(QLabel("Trials per Class:"), 1, 0)
        self.trials_spin = QSpinBox()
        self.trials_spin.setRange(5, 50)
        self.trials_spin.setValue(10)
        layout.addWidget(self.trials_spin, 1, 1)
        
        # Trial duration
        layout.addWidget(QLabel("Trial Duration (s):"), 2, 0)
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(1, 10)
        self.duration_spin.setValue(3)
        layout.addWidget(self.duration_spin, 2, 1)
        
        # Rest duration
        layout.addWidget(QLabel("Rest Duration (s):"), 3, 0)
        self.rest_spin = QSpinBox()
        self.rest_spin.setRange(1, 5)
        self.rest_spin.setValue(2)
        layout.addWidget(self.rest_spin, 3, 1)
        
        return group
        
    def create_controls_section(self):
        """Create control buttons section"""
        group = QGroupBox("Collection Controls")
        layout = QHBoxLayout(group)
        
        # Test connection button
        self.test_btn = QPushButton("🔌 Test Connection")
        self.test_btn.clicked.connect(self.test_connection)
        layout.addWidget(self.test_btn)
        
        # Start collection button
        self.start_btn = QPushButton("▶️ Start Collection")
        self.start_btn.clicked.connect(self.start_collection)
        self.start_btn.setMinimumHeight(40)
        layout.addWidget(self.start_btn)
        
        # Stop collection button
        self.stop_btn = QPushButton("⏹️ Stop Collection")
        self.stop_btn.clicked.connect(self.stop_collection)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setMinimumHeight(40)
        layout.addWidget(self.stop_btn)
        
        # Pause button
        self.pause_btn = QPushButton("⏸️ Pause")
        self.pause_btn.clicked.connect(self.pause_collection)
        self.pause_btn.setEnabled(False)
        layout.addWidget(self.pause_btn)
        
        return group
        
    def create_prompt_section(self):
        """Create visual prompt display section"""
        group = QGroupBox("Current Task")
        layout = QVBoxLayout(group)
        
        # Large prompt display
        self.prompt_label = QLabel("Ready to Start")
        self.prompt_label.setAlignment(Qt.AlignCenter)
        prompt_font = QFont()
        prompt_font.setPointSize(36)
        prompt_font.setBold(True)
        self.prompt_label.setFont(prompt_font)
        self.prompt_label.setMinimumHeight(150)
        self.prompt_label.setStyleSheet("""
            QLabel {
                border: 3px solid #ddd;
                border-radius: 10px;
                background-color: #f5f5f5;
                padding: 20px;
            }
        """)
        
        layout.addWidget(self.prompt_label)
        
        # Task instructions
        self.instruction_label = QLabel("Click 'Start Collection' to begin")
        self.instruction_label.setAlignment(Qt.AlignCenter)
        self.instruction_label.setStyleSheet("color: #666; font-size: 14px;")
        layout.addWidget(self.instruction_label)
        
        return group
        
    def create_status_section(self):
        """Create status and progress section"""
        group = QGroupBox("Collection Progress")
        layout = QVBoxLayout(group)
        
        # Overall progress
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(QLabel("Overall Progress:"))
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("0/0 trials completed")
        progress_layout.addWidget(self.progress_label)
        
        layout.addLayout(progress_layout)
        
        # Current trial timer
        timer_layout = QHBoxLayout()
        timer_layout.addWidget(QLabel("Current Trial:"))
        self.trial_progress = QProgressBar()
        self.trial_progress.setRange(0, 100)
        self.trial_progress.setValue(0)
        timer_layout.addWidget(self.trial_progress)
        
        self.timer_label = QLabel("0s")
        timer_layout.addWidget(self.timer_label)
        
        layout.addLayout(timer_layout)
        
        # Status log
        self.status_log = QTextEdit()
        self.status_log.setMaximumHeight(100)
        self.status_log.setReadOnly(True)
        self.status_log.append("Ready to collect data...")
        layout.addWidget(self.status_log)
        
        return group
        
    def test_connection(self):
        """Test EEG device connection"""
        device = self.device_combo.currentText()
        self.log_status(f"Testing connection to {device}...")
        
        # Set device type and test connection using worker
        self.data_collector.device_type = device
        success = self.data_collector.test_connection()
        
        if success:
            QMessageBox.information(self, "Connection Test", f"Successfully connected to {device}!")
        else:
            QMessageBox.warning(self, "Connection Failed", f"Failed to connect to {device}. Check device and try again.")
        
    def start_collection(self):
        """Start data collection process"""
        if not self.is_collecting:
            self.is_collecting = True
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.pause_btn.setEnabled(True)
            
            # Set collection parameters
            device = self.device_combo.currentText()
            trials_per_class = self.trials_spin.value()
            trial_duration = self.duration_spin.value()
            rest_duration = self.rest_spin.value()
            
            self.data_collector.set_parameters(device, trials_per_class, trial_duration, rest_duration)
            
            # Calculate total trials
            total_trials = trials_per_class * 3  # Rest, Left Fist, Right Fist
            
            self.progress_bar.setRange(0, total_trials)
            self.progress_label.setText(f"0/{total_trials} trials completed")
            
            self.log_status("🔴 Data collection started!")
            self.update_prompt("Get Ready...", "Collection will begin in 3 seconds")
            
            # Start actual data collection thread
            self.data_collector.start_collection()
            
    def stop_collection(self):
        """Stop data collection process"""
        if self.is_collecting:
            self.is_collecting = False
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.pause_btn.setEnabled(False)
            
            self.update_prompt("Collection Stopped", "Click 'Start Collection' to begin again")
            self.log_status("⏹️ Data collection stopped")
            
            # Stop data collection thread
            self.data_collector.stop_collection()
            
    def pause_collection(self):
        """Pause/resume data collection"""
        self.data_collector.pause_collection()
        
    def update_prompt(self, main_text, instruction_text):
        """Update the visual prompt display"""
        self.prompt_label.setText(main_text)
        self.instruction_label.setText(instruction_text)
        
        # Color coding for different tasks
        if "Rest" in main_text:
            self.prompt_label.setStyleSheet("""
                QLabel {
                    border: 3px solid #4caf50;
                    border-radius: 10px;
                    background-color: #e8f5e8;
                    padding: 20px;
                    color: #2e7d32;
                }
            """)
        elif "Left" in main_text:
            self.prompt_label.setStyleSheet("""
                QLabel {
                    border: 3px solid #2196f3;
                    border-radius: 10px;
                    background-color: #e3f2fd;
                    padding: 20px;
                    color: #1565c0;
                }
            """)
        elif "Right" in main_text:
            self.prompt_label.setStyleSheet("""
                QLabel {
                    border: 3px solid #ff9800;
                    border-radius: 10px;
                    background-color: #fff3e0;
                    padding: 20px;
                    color: #ef6c00;
                }
            """)
        else:
            self.prompt_label.setStyleSheet("""
                QLabel {
                    border: 3px solid #ddd;
                    border-radius: 10px;
                    background-color: #f5f5f5;
                    padding: 20px;
                    color: #333;
                }
            """)
            
    def log_status(self, message):
        """Add message to status log"""
        self.status_log.append(f"• {message}")
        
    def update_progress(self, completed_trials, total_trials):
        """Update progress indicators"""
        self.progress_bar.setValue(completed_trials)
        self.progress_label.setText(f"{completed_trials}/{total_trials} trials completed")
        
    def update_trial_progress(self, progress_percent, time_remaining):
        """Update current trial progress"""
        self.trial_progress.setValue(progress_percent)
        self.timer_label.setText(f"{time_remaining}s")
        
    def connect_worker_signals(self):
        """Connect signals from the EEG data collection worker"""
        self.data_collector.prompt_updated.connect(self.update_prompt)
        self.data_collector.progress_updated.connect(self.update_progress)
        self.data_collector.trial_progress_updated.connect(self.update_trial_progress)
        self.data_collector.status_logged.connect(self.log_status)
        self.data_collector.collection_finished.connect(self.on_collection_finished)
        self.data_collector.connection_status_changed.connect(self.on_connection_status_changed)
        
    def on_collection_finished(self, success, message):
        """Handle collection completion"""
        self.is_collecting = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        
        if success:
            self.update_prompt("Collection Complete!", f"✅ {message}")
            QMessageBox.information(self, "Collection Complete", f"Data collection finished successfully!\n{message}")
        else:
            self.update_prompt("Collection Failed", f"❌ {message}")
            QMessageBox.warning(self, "Collection Failed", f"Data collection failed:\n{message}")
            
    def on_connection_status_changed(self, connected):
        """Handle connection status changes"""
        # This will be used to update UI connection indicators
        pass
        
    def connect_worker_signals(self):
        """Connect signals from the EEG data collection worker"""
        self.data_collector.prompt_updated.connect(self.update_prompt)
        self.data_collector.progress_updated.connect(self.update_progress)
        self.data_collector.trial_progress_updated.connect(self.update_trial_progress)
        self.data_collector.status_logged.connect(self.log_status)
        self.data_collector.collection_finished.connect(self.on_collection_finished)
        self.data_collector.connection_status_changed.connect(self.on_connection_status_changed)
        
    def on_collection_finished(self, success, message):
        """Handle collection completion"""
        self.is_collecting = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        
        if success:
            self.update_prompt("Collection Complete!", f"✅ {message}")
            QMessageBox.information(self, "Collection Complete", f"Data collection finished successfully!\n{message}")
        else:
            self.update_prompt("Collection Failed", f"❌ {message}")
            QMessageBox.warning(self, "Collection Failed", f"Data collection failed:\n{message}")
            
    def on_connection_status_changed(self, connected):
        """Handle connection status changes"""
        # This will be used to update UI connection indicators
        pass
        
    def connect_worker_signals(self):
        """Connect signals from the EEG data collection worker"""
        self.data_collector.prompt_updated.connect(self.update_prompt)
        self.data_collector.progress_updated.connect(self.update_progress)
        self.data_collector.trial_progress_updated.connect(self.update_trial_progress)
        self.data_collector.status_logged.connect(self.log_status)
        self.data_collector.collection_finished.connect(self.on_collection_finished)
        self.data_collector.connection_status_changed.connect(self.on_connection_status_changed)
        
    def on_collection_finished(self, success, message):
        """Handle collection completion"""
        self.is_collecting = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        
        if success:
            self.update_prompt("Collection Complete!", f"✅ {message}")
            QMessageBox.information(self, "Collection Complete", f"Data collection finished successfully!\n{message}")
        else:
            self.update_prompt("Collection Failed", f"❌ {message}")
            QMessageBox.warning(self, "Collection Failed", f"Data collection failed:\n{message}")
            
    def on_connection_status_changed(self, connected):
        """Handle connection status changes"""
        # This will be used to update UI connection indicators
        pass 