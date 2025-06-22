"""
Home Tab - Overview and status of the EEG-BCI system
"""
import time
from PySide6.QtCore import Qt, QTimer, Signal

# Optional psutil import for system monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("psutil not available - system monitoring will use simulated data")
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QGroupBox, QProgressBar,
    QTextEdit, QFrame
)
from PySide6.QtGui import QFont, QPixmap

class HomeTab(QWidget):
    # Signals for tab switching
    switch_to_tab = Signal(int)  # Signal to switch to a specific tab
    
    def __init__(self):
        super().__init__()
        
        # System monitoring
        self.system_timer = QTimer()
        self.system_timer.timeout.connect(self.update_system_metrics)
        self.system_timer.start(2000)  # Update every 2 seconds
        
        # Session tracking
        self.session_start_time = time.time()
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the home tab UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        
        # Header section
        header = self.create_header()
        layout.addWidget(header)
        
        # Status overview section
        status_section = self.create_status_section()
        layout.addWidget(status_section)
        
        # System metrics section
        metrics_section = self.create_system_metrics()
        layout.addWidget(metrics_section)
        
        # Quick actions section
        actions_section = self.create_quick_actions()
        layout.addWidget(actions_section)
        
        # Recent activity section
        activity_section = self.create_activity_section()
        layout.addWidget(activity_section)
        
        layout.addStretch()
        
    def create_header(self):
        """Create the header section with title and description"""
        header_widget = QWidget()
        layout = QVBoxLayout(header_widget)
        
        # Title
        title = QLabel("🧠 ezpz-BCI Control Center")
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)
        
        # Description
        description = QLabel(
            "Transform your EEG signals into computer commands.\n"
            "Train models, collect data, and control your computer with your thoughts."
        )
        description.setAlignment(Qt.AlignCenter)
        description.setStyleSheet("color: #666; font-size: 14px;")
        
        layout.addWidget(title)
        layout.addWidget(description)
        
        return header_widget
        
    def create_status_section(self):
        """Create system status overview"""
        group = QGroupBox("System Status")
        layout = QGridLayout(group)
        
        # EEG Device Status
        layout.addWidget(QLabel("EEG Device:"), 0, 0)
        self.device_status = QLabel("❌ Not Connected")
        self.device_status.setStyleSheet("color: #d32f2f;")
        layout.addWidget(self.device_status, 0, 1)
        
        # Model Status
        layout.addWidget(QLabel("Trained Model:"), 1, 0)
        self.model_status = QLabel("❌ No Model Loaded")
        self.model_status.setStyleSheet("color: #d32f2f;")
        layout.addWidget(self.model_status, 1, 1)
        
        # Classification Status
        layout.addWidget(QLabel("Live Classification:"), 2, 0)
        self.classification_status = QLabel("⏸️ Stopped")
        self.classification_status.setStyleSheet("color: #f57c00;")
        layout.addWidget(self.classification_status, 2, 1)
        
        # Last Prediction
        layout.addWidget(QLabel("Last Prediction:"), 3, 0)
        self.last_prediction = QLabel("None")
        layout.addWidget(self.last_prediction, 3, 1)
        
        return group
        
    def create_system_metrics(self):
        """Create system performance metrics overview"""
        group = QGroupBox("System Performance")
        layout = QGridLayout(group)
        
        # CPU Usage
        layout.addWidget(QLabel("CPU Usage:"), 0, 0)
        self.cpu_bar = QProgressBar()
        self.cpu_bar.setRange(0, 100)
        self.cpu_bar.setValue(0)
        layout.addWidget(self.cpu_bar, 0, 1)
        self.cpu_label = QLabel("0%")
        layout.addWidget(self.cpu_label, 0, 2)
        
        # Memory Usage
        layout.addWidget(QLabel("Memory Usage:"), 1, 0)
        self.memory_bar = QProgressBar()
        self.memory_bar.setRange(0, 100)
        self.memory_bar.setValue(0)
        layout.addWidget(self.memory_bar, 1, 1)
        self.memory_label = QLabel("0%")
        layout.addWidget(self.memory_label, 1, 2)
        
        # Session Time
        layout.addWidget(QLabel("Session Time:"), 2, 0)
        self.session_time_label = QLabel("00:00:00")
        layout.addWidget(self.session_time_label, 2, 1, 1, 2)
        
        return group
        
    def create_quick_actions(self):
        """Create quick action buttons"""
        group = QGroupBox("Quick Actions")
        layout = QHBoxLayout(group)
        
        # Data Collection
        collect_btn = QPushButton("📊 Collect Training Data")
        collect_btn.setMinimumHeight(50)
        collect_btn.clicked.connect(self.start_data_collection)
        layout.addWidget(collect_btn)
        
        # Train Model
        train_btn = QPushButton("🤖 Train Model")
        train_btn.setMinimumHeight(50)
        train_btn.clicked.connect(self.start_training)
        layout.addWidget(train_btn)
        
        # Start Classification
        classify_btn = QPushButton("🧠 Start Classification")
        classify_btn.setMinimumHeight(50)
        classify_btn.clicked.connect(self.start_classification)
        layout.addWidget(classify_btn)
        
        return group
        
    def create_activity_section(self):
        """Create recent activity log"""
        group = QGroupBox("Recent Activity")
        layout = QVBoxLayout(group)
        
        self.activity_log = QTextEdit()
        self.activity_log.setMaximumHeight(150)
        self.activity_log.setReadOnly(True)
        self.activity_log.append("System started - Welcome to ezpz-BCI!")
        
        layout.addWidget(self.activity_log)
        
        return group
        
    def start_data_collection(self):
        """Switch to data collection tab"""
        self.log_activity("Switching to Data Collection tab...")
        self.switch_to_tab.emit(1)  # Data Collection tab index
        
    def start_training(self):
        """Switch to training tab"""
        self.log_activity("Switching to Training tab...")
        self.switch_to_tab.emit(2)  # Training tab index
        
    def start_classification(self):
        """Switch to classification tab"""
        self.log_activity("Switching to Live Classification tab...")
        self.switch_to_tab.emit(3)  # Classification tab index
        
    def log_activity(self, message):
        """Add message to activity log"""
        self.activity_log.append(f"• {message}")
        
    def update_device_status(self, connected):
        """Update EEG device connection status"""
        if connected:
            self.device_status.setText("✅ Connected")
            self.device_status.setStyleSheet("color: #388e3c;")
        else:
            self.device_status.setText("❌ Not Connected")
            self.device_status.setStyleSheet("color: #d32f2f;")
            
    def update_model_status(self, loaded):
        """Update model loading status"""
        if loaded:
            self.model_status.setText("✅ Model Loaded")
            self.model_status.setStyleSheet("color: #388e3c;")
        else:
            self.model_status.setText("❌ No Model Loaded")
            self.model_status.setStyleSheet("color: #d32f2f;")
            
    def update_classification_status(self, active):
        """Update classification status"""
        if active:
            self.classification_status.setText("▶️ Running")
            self.classification_status.setStyleSheet("color: #388e3c;")
        else:
            self.classification_status.setText("⏸️ Stopped")
            self.classification_status.setStyleSheet("color: #f57c00;")
            
    def update_last_prediction(self, prediction):
        """Update the last prediction display"""
        self.last_prediction.setText(prediction)
        
    def update_system_metrics(self):
        """Update system performance metrics"""
        try:
            if PSUTIL_AVAILABLE:
                # Get real CPU and memory usage
                cpu_percent = psutil.cpu_percent(interval=None)
                memory_percent = psutil.virtual_memory().percent
            else:
                # Use simulated data
                import random
                cpu_percent = random.uniform(10, 60)
                memory_percent = random.uniform(30, 70)
            
            # Update CPU
            self.cpu_bar.setValue(int(cpu_percent))
            self.cpu_label.setText(f"{cpu_percent:.1f}%")
            
            # Update Memory
            self.memory_bar.setValue(int(memory_percent))
            self.memory_label.setText(f"{memory_percent:.1f}%")
            
            # Update session time
            session_time = time.time() - self.session_start_time
            hours = int(session_time // 3600)
            minutes = int((session_time % 3600) // 60)
            seconds = int(session_time % 60)
            self.session_time_label.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
            
        except Exception as e:
            self.log_activity(f"❌ Error updating system metrics: {str(e)}")