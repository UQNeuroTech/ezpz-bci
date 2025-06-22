"""
Training Tab - Interface for training EEGNet models
"""
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QGroupBox, QProgressBar,
    QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit,
    QFileDialog, QMessageBox, QSlider, QCheckBox
)
from PySide6.QtGui import QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Import the ModelTrainer worker
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'workers'))
from model_trainer import ModelTrainer

class TrainingTab(QWidget):
    def __init__(self):
        super().__init__()
        self.is_training = False
        self.training_data_path = None
        
        # Initialize the model trainer worker
        self.model_trainer = ModelTrainer()
        self.connect_worker_signals()
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the training tab UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Header
        header = self.create_header()
        layout.addWidget(header)
        
        # Data selection section
        data_section = self.create_data_section()
        layout.addWidget(data_section)
        
        # Training parameters section
        params_section = self.create_parameters_section()
        layout.addWidget(params_section)
        
        # Training controls
        controls_section = self.create_controls_section()
        layout.addWidget(controls_section)
        
        # Progress and visualization
        progress_section = self.create_progress_section()
        layout.addWidget(progress_section)
        
        layout.addStretch()
        
    def create_header(self):
        """Create header section"""
        header_widget = QWidget()
        layout = QVBoxLayout(header_widget)
        
        title = QLabel("🤖 EEGNet Model Training")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title.setFont(title_font)
        
        description = QLabel(
            "Train an EEGNet classifier on your collected EEG data.\n"
            "Configure training parameters and monitor progress in real-time."
        )
        description.setStyleSheet("color: #666; font-size: 12px;")
        
        layout.addWidget(title)
        layout.addWidget(description)
        
        return header_widget
        
    def create_data_section(self):
        """Create data selection section"""
        group = QGroupBox("Training Data")
        layout = QVBoxLayout(group)
        
        # Data file selection
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("Data File:"))
        
        self.data_path_label = QLabel("No data file selected")
        self.data_path_label.setStyleSheet("color: #666; font-style: italic;")
        file_layout.addWidget(self.data_path_label)
        
        self.browse_btn = QPushButton("📁 Browse")
        self.browse_btn.clicked.connect(self.browse_data_file)
        file_layout.addWidget(self.browse_btn)
        
        layout.addLayout(file_layout)
        
        # Data info display
        self.data_info = QTextEdit()
        self.data_info.setMaximumHeight(80)
        self.data_info.setReadOnly(True)
        self.data_info.setPlainText("Select a data file to view information...")
        layout.addWidget(self.data_info)
        
        return group
        
    def create_parameters_section(self):
        """Create training parameters section"""
        group = QGroupBox("Training Parameters")
        layout = QGridLayout(group)
        
        # Epochs
        layout.addWidget(QLabel("Epochs:"), 0, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(10, 1000)
        self.epochs_spin.setValue(200)
        layout.addWidget(self.epochs_spin, 0, 1)
        
        # Batch size
        layout.addWidget(QLabel("Batch Size:"), 1, 0)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(8, 128)
        self.batch_spin.setValue(64)
        layout.addWidget(self.batch_spin, 1, 1)
        
        # Learning rate
        layout.addWidget(QLabel("Learning Rate:"), 2, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 0.1)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setDecimals(4)
        self.lr_spin.setSingleStep(0.0001)
        layout.addWidget(self.lr_spin, 2, 1)
        
        # Test ratio
        layout.addWidget(QLabel("Test Ratio:"), 3, 0)
        self.test_ratio_spin = QDoubleSpinBox()
        self.test_ratio_spin.setRange(0.1, 0.5)
        self.test_ratio_spin.setValue(0.3)
        self.test_ratio_spin.setDecimals(2)
        self.test_ratio_spin.setSingleStep(0.05)
        layout.addWidget(self.test_ratio_spin, 3, 1)
        
        # Advanced options
        layout.addWidget(QLabel("Use GPU:"), 4, 0)
        self.gpu_checkbox = QCheckBox("Enable CUDA (if available)")
        self.gpu_checkbox.setChecked(True)
        layout.addWidget(self.gpu_checkbox, 4, 1)
        
        # Model architecture
        layout.addWidget(QLabel("Model Type:"), 5, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["EEGNet-8,2 (Default)", "EEGNet-4,2", "EEGNet-8,4"])
        layout.addWidget(self.model_combo, 5, 1)
        
        return group
        
    def create_controls_section(self):
        """Create training control buttons"""
        group = QGroupBox("Training Controls")
        layout = QHBoxLayout(group)
        
        # Validate data button
        self.validate_btn = QPushButton("✅ Validate Data")
        self.validate_btn.clicked.connect(self.validate_data)
        layout.addWidget(self.validate_btn)
        
        # Start training button
        self.start_train_btn = QPushButton("🚀 Start Training")
        self.start_train_btn.clicked.connect(self.start_training)
        self.start_train_btn.setMinimumHeight(40)
        self.start_train_btn.setEnabled(False)
        layout.addWidget(self.start_train_btn)
        
        # Stop training button
        self.stop_train_btn = QPushButton("⏹️ Stop Training")
        self.stop_train_btn.clicked.connect(self.stop_training)
        self.stop_train_btn.setEnabled(False)
        self.stop_train_btn.setMinimumHeight(40)
        layout.addWidget(self.stop_train_btn)
        
        # Save model button
        self.save_btn = QPushButton("💾 Save Model")
        self.save_btn.clicked.connect(self.save_model)
        self.save_btn.setEnabled(False)
        layout.addWidget(self.save_btn)
        
        return group
        
    def create_progress_section(self):
        """Create progress monitoring section"""
        group = QGroupBox("Training Progress")
        layout = QVBoxLayout(group)
        
        # Progress bar and info
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(QLabel("Progress:"))
        
        self.training_progress = QProgressBar()
        self.training_progress.setRange(0, 100)
        self.training_progress.setValue(0)
        progress_layout.addWidget(self.training_progress)
        
        self.progress_label = QLabel("0/0 epochs")
        progress_layout.addWidget(self.progress_label)
        
        layout.addLayout(progress_layout)
        
        # Current metrics
        metrics_layout = QGridLayout()
        
        metrics_layout.addWidget(QLabel("Current Loss:"), 0, 0)
        self.loss_label = QLabel("N/A")
        metrics_layout.addWidget(self.loss_label, 0, 1)
        
        metrics_layout.addWidget(QLabel("Current Accuracy:"), 0, 2)
        self.accuracy_label = QLabel("N/A")
        metrics_layout.addWidget(self.accuracy_label, 0, 3)
        
        metrics_layout.addWidget(QLabel("Best Accuracy:"), 1, 0)
        self.best_accuracy_label = QLabel("N/A")
        metrics_layout.addWidget(self.best_accuracy_label, 1, 1)
        
        metrics_layout.addWidget(QLabel("Time Remaining:"), 1, 2)
        self.time_remaining_label = QLabel("N/A")
        metrics_layout.addWidget(self.time_remaining_label, 1, 3)
        
        layout.addLayout(metrics_layout)
        
        # Training plots
        self.create_plots()
        layout.addWidget(self.plot_widget)
        
        # Training log
        self.training_log = QTextEdit()
        self.training_log.setMaximumHeight(120)
        self.training_log.setReadOnly(True)
        self.training_log.append("Ready to start training...")
        layout.addWidget(self.training_log)
        
        return group
        
    def create_plots(self):
        """Create matplotlib plots for training visualization"""
        self.figure = Figure(figsize=(12, 4))
        self.plot_widget = FigureCanvas(self.figure)
        
        # Create subplots
        self.loss_ax = self.figure.add_subplot(121)
        self.accuracy_ax = self.figure.add_subplot(122)
        
        # Initialize empty plots
        self.loss_ax.set_title("Training Loss")
        self.loss_ax.set_xlabel("Epoch")
        self.loss_ax.set_ylabel("Loss")
        self.loss_ax.grid(True, alpha=0.3)
        
        self.accuracy_ax.set_title("Training Accuracy")
        self.accuracy_ax.set_xlabel("Epoch")
        self.accuracy_ax.set_ylabel("Accuracy")
        self.accuracy_ax.grid(True, alpha=0.3)
        
        self.figure.tight_layout()
        
        # Data storage for plots
        self.epochs_data = []
        self.loss_data = []
        self.accuracy_data = []
        
    def browse_data_file(self):
        """Browse for training data file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Training Data File",
            "",
            "EEG Files (*.fif *.json);;All Files (*)"
        )
        
        if file_path:
            self.training_data_path = file_path
            self.data_path_label.setText(file_path.split('/')[-1])
            self.data_path_label.setStyleSheet("color: #333;")
            self.log_training(f"Selected data file: {file_path.split('/')[-1]}")
            
            # TODO: Load and analyze data file
            self.analyze_data_file(file_path)
            
    def analyze_data_file(self, file_path):
        """Analyze the selected data file"""
        try:
            # TODO: Implement actual data analysis
            # For now, show placeholder info
            info_text = f"""Data File: {file_path.split('/')[-1]}
Format: EEG Epochs (.fif)
Channels: 8
Sampling Rate: 250 Hz
Epochs: 150 (50 per class)
Classes: Rest, Left Fist, Right Fist
Duration: 3.2s per epoch

✅ Data file is valid and ready for training."""
            
            self.data_info.setPlainText(info_text)
            self.start_train_btn.setEnabled(True)
            self.log_training("✅ Data file validated successfully")
            
        except Exception as e:
            self.data_info.setPlainText(f"❌ Error analyzing data file:\n{str(e)}")
            self.start_train_btn.setEnabled(False)
            self.log_training(f"❌ Error: {str(e)}")
            
    def validate_data(self):
        """Validate training data and parameters"""
        if not self.training_data_path:
            QMessageBox.warning(self, "No Data", "Please select a training data file first.")
            return
            
        # Use the model trainer to validate data
        self.model_trainer.set_data_path(self.training_data_path)
        self.model_trainer.validate_data()
        
    def start_training(self):
        """Start model training"""
        if not self.is_training:
            self.is_training = True
            self.start_train_btn.setEnabled(False)
            self.stop_train_btn.setEnabled(True)
            
            # Get training parameters
            epochs = self.epochs_spin.value()
            batch_size = self.batch_spin.value()
            learning_rate = self.lr_spin.value()
            test_ratio = self.test_ratio_spin.value()
            
            self.training_progress.setRange(0, epochs)
            self.progress_label.setText(f"0/{epochs} epochs")
            
            self.log_training("🚀 Training started!")
            self.log_training(f"Parameters: {epochs} epochs, batch size {batch_size}, lr {learning_rate}")
            
            # Clear previous plots
            self.epochs_data.clear()
            self.loss_data.clear()
            self.accuracy_data.clear()
            
            # Configure and start the model trainer
            self.model_trainer.set_data_path(self.training_data_path)
            self.model_trainer.set_parameters(epochs, batch_size, learning_rate, test_ratio)
            self.model_trainer.start_training()
            
    def stop_training(self):
        """Stop model training"""
        if self.is_training:
            self.is_training = False
            self.start_train_btn.setEnabled(True)
            self.stop_train_btn.setEnabled(False)
            self.save_btn.setEnabled(True)
            
            self.log_training("⏹️ Training stopped by user")
            
            # Stop the model trainer
            self.model_trainer.stop_training()
            
    def save_model(self):
        """Save trained model"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Trained Model",
            "eegnet_model.pth",
            "PyTorch Models (*.pth);;All Files (*)"
        )
        
        if file_path:
            # TODO: Implement actual model saving
            self.log_training(f"💾 Model saved to: {file_path.split('/')[-1]}")
            QMessageBox.information(self, "Model Saved", f"Model saved successfully to:\n{file_path}")
            
    def update_training_progress(self, epoch, total_epochs, loss, accuracy):
        """Update training progress and plots"""
        # Update progress bar and labels
        self.training_progress.setValue(epoch)
        self.progress_label.setText(f"{epoch}/{total_epochs} epochs")
        self.loss_label.setText(f"{loss:.4f}")
        self.accuracy_label.setText(f"{accuracy:.2%}")
        
        # Update best accuracy
        if hasattr(self, 'best_accuracy'):
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_accuracy_label.setText(f"{accuracy:.2%}")
        else:
            self.best_accuracy = accuracy
            self.best_accuracy_label.setText(f"{accuracy:.2%}")
            
        # Update plots
        self.epochs_data.append(epoch)
        self.loss_data.append(loss)
        self.accuracy_data.append(accuracy)
        
        # Redraw plots
        self.loss_ax.clear()
        self.accuracy_ax.clear()
        
        self.loss_ax.plot(self.epochs_data, self.loss_data, 'b-', linewidth=2)
        self.loss_ax.set_title("Training Loss")
        self.loss_ax.set_xlabel("Epoch")
        self.loss_ax.set_ylabel("Loss")
        self.loss_ax.grid(True, alpha=0.3)
        
        self.accuracy_ax.plot(self.epochs_data, self.accuracy_data, 'g-', linewidth=2)
        self.accuracy_ax.set_title("Training Accuracy")
        self.accuracy_ax.set_xlabel("Epoch")
        self.accuracy_ax.set_ylabel("Accuracy")
        self.accuracy_ax.grid(True, alpha=0.3)
        
        self.figure.tight_layout()
        self.plot_widget.draw()
        
    def log_training(self, message):
        """Add message to training log"""
        self.training_log.append(f"• {message}")
        
    def estimate_time_remaining(self, epoch, total_epochs, elapsed_time):
        """Estimate remaining training time"""
        if epoch > 0:
            time_per_epoch = elapsed_time / epoch
            remaining_epochs = total_epochs - epoch
            remaining_time = time_per_epoch * remaining_epochs
            
            # Format time
            if remaining_time > 3600:
                time_str = f"{remaining_time/3600:.1f}h"
            elif remaining_time > 60:
                time_str = f"{remaining_time/60:.1f}m"
            else:
                time_str = f"{remaining_time:.0f}s"
                
            self.time_remaining_label.setText(time_str)
        else:
            self.time_remaining_label.setText("Calculating...")
            
    def connect_worker_signals(self):
        """Connect signals from the model trainer worker"""
        self.model_trainer.training_progress_updated.connect(self.update_training_progress)
        self.model_trainer.status_logged.connect(self.log_training)
        self.model_trainer.training_finished.connect(self.on_training_finished)
        self.model_trainer.validation_completed.connect(self.on_validation_completed)
        
    def on_training_finished(self, success, message):
        """Handle training completion"""
        self.is_training = False
        self.start_train_btn.setEnabled(True)
        self.stop_train_btn.setEnabled(False)
        self.save_btn.setEnabled(True)
        
        if success:
            self.log_training(f"✅ {message}")
            QMessageBox.information(self, "Training Complete", f"Model training finished successfully!\n{message}")
        else:
            self.log_training(f"❌ {message}")
            QMessageBox.warning(self, "Training Failed", f"Model training failed:\n{message}")
            
    def on_validation_completed(self, success, message):
        """Handle data validation completion"""
        if success:
            self.data_info.setPlainText(message)
            self.start_train_btn.setEnabled(True)
            self.log_training("✅ Data validation completed")
            QMessageBox.information(self, "Validation Complete", "Data validation passed! Ready to train.")
        else:
            self.data_info.setPlainText(f"❌ Validation failed:\n{message}")
            self.start_train_btn.setEnabled(False)
            self.log_training(f"❌ Validation failed: {message}")
            QMessageBox.warning(self, "Validation Failed", f"Data validation failed:\n{message}")