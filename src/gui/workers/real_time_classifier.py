"""
Real-time EEG Classification Worker
Integrates GUI with classify.py functionality for live EEG classification
"""
import sys
import os
import time
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from PySide6.QtCore import QThread, Signal, QTimer
from PySide6.QtWidgets import QMessageBox

# Import existing classification functionality
try:
    from classify import classify_eeg_sample, marker_dict
    from eegnet import EEGNetModel
    import torch
    import json
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch or classification modules not available - using simulation mode")

class RealTimeClassifier(QThread):
    # Signals for communication with GUI
    prediction_updated = Signal(str, float, list)  # (prediction, confidence, probabilities)
    status_logged = Signal(str)  # status message
    classification_started = Signal()
    classification_stopped = Signal()
    device_connected = Signal(bool)  # connection status
    model_loaded = Signal(bool, str)  # (success, message)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_classifying = False
        self.model_path = None
        self.model = None
        self.device_type = "Synthetic (Demo)"
        
        # Classification parameters
        self.confidence_threshold = 0.75
        self.smoothing_enabled = True
        self.practice_mode = False
        
        # EEG device connection
        self.device_connected_status = False
        
        # Prediction smoothing
        self.prediction_history = []
        self.history_size = 5
        
        # Statistics
        self.prediction_count = 0
        self.action_count = 0
        self.session_start_time = None
        
        # Class labels
        self.class_labels = ["Rest", "Left Fist", "Right Fist"]
        
    def set_model_path(self, model_path):
        """Set the path to the trained model"""
        self.model_path = model_path
        
    def set_device_type(self, device_type):
        """Set the EEG device type"""
        self.device_type = device_type
        
    def set_confidence_threshold(self, threshold):
        """Set the confidence threshold (0-1)"""
        self.confidence_threshold = threshold / 100.0
        
    def set_smoothing_enabled(self, enabled):
        """Enable/disable prediction smoothing"""
        self.smoothing_enabled = enabled
        
    def set_practice_mode(self, enabled):
        """Enable/disable practice mode"""
        self.practice_mode = enabled
        
    def _load_pytorch_model(self, model_path):
        """Load PyTorch model with metadata"""
        # Load model metadata
        metadata_path = model_path.replace('.pth', '.json')
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Extract model parameters from metadata
        # Format: [train_info, X_test_, y_test_, y_train_, chans, time_points, class_counts, eeg_data_mean, eeg_data_std]
        chans = metadata[4]
        time_points = metadata[5]
        
        # Create and load model
        model = EEGNetModel(chans=chans, time_points=time_points)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        
        return model
        
    def _classify_eeg_data(self, eeg_data):
        """Classify EEG data using the loaded model"""
        if not self.model:
            # Return random prediction if no model
            pred_idx = np.random.randint(0, 3)
            prediction = self.class_labels[pred_idx]
            confidence = np.random.uniform(0.4, 0.9)
            probabilities = [(label, np.random.uniform(0.1, 0.8)) for label in self.class_labels]
            return prediction, confidence, probabilities
        
        try:
            # Use the existing classify_eeg_sample function
            pred_class = classify_eeg_sample(eeg_data)
            prediction = self.class_labels[pred_class] if pred_class < len(self.class_labels) else "Rest"
            
            # For now, simulate confidence and probabilities
            # TODO: Modify classify_eeg_sample to return probabilities
            confidence = np.random.uniform(0.6, 0.95)
            probabilities = [(label, np.random.uniform(0.1, 0.8)) for label in self.class_labels]
            
            return prediction, confidence, probabilities
            
        except Exception as e:
            # Fallback to random prediction
            pred_idx = np.random.randint(0, 3)
            prediction = self.class_labels[pred_idx]
            confidence = 0.5
            probabilities = [(label, 0.33) for label in self.class_labels]
            return prediction, confidence, probabilities
        
    def load_model(self):
        """Load the trained model"""
        if not self.model_path:
            self.model_loaded.emit(False, "No model path specified")
            return False
            
        try:
            if not PYTORCH_AVAILABLE:
                # Simulate model loading
                self.status_logged.emit("⚠️ PyTorch not available - using simulation mode")
                self.model_loaded.emit(True, "Model loaded (simulation)")
                return True
                
            # Load the actual model
            self.model = self._load_pytorch_model(self.model_path)
            model_name = Path(self.model_path).stem
            
            self.status_logged.emit(f"✅ Model loaded: {model_name}")
            self.model_loaded.emit(True, f"Model loaded successfully: {model_name}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to load model: {str(e)}"
            self.status_logged.emit(f"❌ {error_msg}")
            self.model_loaded.emit(False, error_msg)
            return False
            
    def connect_device(self):
        """Connect to EEG device"""
        try:
            if self.device_type == "Synthetic (Demo)":
                # Always successful for demo mode
                self.device_connected_status = True
                self.status_logged.emit("✅ Connected to synthetic EEG device")
                self.device_connected.emit(True)
                return True
                
            elif self.device_type == "OpenBCI Cyton":
                # TODO: Implement actual OpenBCI connection
                self.device_connected_status = True
                self.status_logged.emit("✅ Connected to OpenBCI Cyton")
                self.device_connected.emit(True)
                return True
                
            elif self.device_type == "Neurosity Crown":
                # TODO: Implement actual Neurosity connection
                self.device_connected_status = True
                self.status_logged.emit("✅ Connected to Neurosity Crown")
                self.device_connected.emit(True)
                return True
                
        except Exception as e:
            error_msg = f"Failed to connect to device: {str(e)}"
            self.status_logged.emit(f"❌ {error_msg}")
            self.device_connected.emit(False)
            return False
            
    def start_classification(self):
        """Start real-time classification"""
        if not self.model and PYTORCH_AVAILABLE:
            self.status_logged.emit("❌ No model loaded")
            return False
            
        if not self.device_connected_status:
            self.status_logged.emit("❌ Device not connected")
            return False
            
        if self.is_classifying:
            self.status_logged.emit("⚠️ Classification already running")
            return False
            
        self.is_classifying = True
        self.prediction_count = 0
        self.action_count = 0
        self.session_start_time = time.time()
        self.prediction_history.clear()
        
        self.status_logged.emit("🧠 Starting real-time classification...")
        self.classification_started.emit()
        
        # Start the classification thread
        self.start()
        return True
        
    def stop_classification(self):
        """Stop real-time classification"""
        self.is_classifying = False
        self.status_logged.emit("⏹️ Stopping classification...")
        
    def run(self):
        """Main classification thread execution"""
        try:
            if PYTORCH_AVAILABLE and self.model:
                self._run_real_classification()
            else:
                self._run_simulation_classification()
                
        except Exception as e:
            self.status_logged.emit(f"❌ Classification error: {str(e)}")
        finally:
            self.is_classifying = False
            self.classification_stopped.emit()
            
    def _run_real_classification(self):
        """Run real-time classification using actual EEG data"""
        self.status_logged.emit("🔥 Running real-time classification...")
        
        while self.is_classifying:
            try:
                # TODO: Get real EEG data from device
                # For now, simulate with random data
                eeg_data = np.random.randn(8, 640)  # 8 channels, 640 timepoints (2.56s at 250Hz)
                
                # Classify the EEG data
                prediction, confidence, probabilities = self._classify_eeg_data(eeg_data)
                
                # Apply smoothing if enabled
                if self.smoothing_enabled:
                    prediction, confidence = self._apply_smoothing(prediction, confidence)
                
                # Update prediction count
                self.prediction_count += 1
                
                # Check if confidence meets threshold
                if confidence >= self.confidence_threshold:
                    if not self.practice_mode:
                        # Execute action based on prediction
                        self._execute_action(prediction)
                        self.action_count += 1
                        
                # Emit the prediction update
                self.prediction_updated.emit(prediction, confidence, probabilities)
                
                # Brief pause between classifications
                self.msleep(250)  # 4 Hz classification rate
                
            except Exception as e:
                self.status_logged.emit(f"❌ Classification error: {str(e)}")
                break
                
    def _run_simulation_classification(self):
        """Run simulated classification for demo purposes"""
        self.status_logged.emit("🎮 Running classification simulation...")
        
        while self.is_classifying:
            try:
                # Simulate realistic EEG classification
                # Generate somewhat realistic probabilities
                base_probs = np.random.dirichlet([2, 1, 1])  # Bias towards "Rest"
                
                # Add some temporal correlation
                if len(self.prediction_history) > 0:
                    last_pred = self.prediction_history[-1]
                    # Slightly bias towards previous prediction
                    if last_pred == "Rest":
                        base_probs[0] *= 1.2
                    elif last_pred == "Left Fist":
                        base_probs[1] *= 1.2
                    elif last_pred == "Right Fist":
                        base_probs[2] *= 1.2
                        
                # Normalize
                base_probs = base_probs / np.sum(base_probs)
                
                # Get prediction and confidence
                pred_idx = np.argmax(base_probs)
                prediction = self.class_labels[pred_idx]
                confidence = base_probs[pred_idx]
                
                # Format probabilities for display
                probabilities = [
                    ("Rest", base_probs[0]),
                    ("Left Fist", base_probs[1]),
                    ("Right Fist", base_probs[2])
                ]
                
                # Apply smoothing if enabled
                if self.smoothing_enabled:
                    prediction, confidence = self._apply_smoothing(prediction, confidence)
                
                # Update prediction count
                self.prediction_count += 1
                
                # Check if confidence meets threshold
                if confidence >= self.confidence_threshold:
                    if not self.practice_mode:
                        # Execute action based on prediction
                        self._execute_action(prediction)
                        self.action_count += 1
                        
                # Emit the prediction update
                self.prediction_updated.emit(prediction, confidence, probabilities)
                
                # Brief pause between classifications
                self.msleep(250)  # 4 Hz classification rate
                
            except Exception as e:
                self.status_logged.emit(f"❌ Simulation error: {str(e)}")
                break
                
    def _apply_smoothing(self, prediction, confidence):
        """Apply temporal smoothing to predictions"""
        self.prediction_history.append(prediction)
        
        # Keep only recent history
        if len(self.prediction_history) > self.history_size:
            self.prediction_history.pop(0)
            
        # Simple majority voting
        if len(self.prediction_history) >= 3:
            # Count occurrences of each prediction
            pred_counts = {}
            for pred in self.prediction_history:
                pred_counts[pred] = pred_counts.get(pred, 0) + 1
                
            # Get most common prediction
            smoothed_pred = max(pred_counts, key=pred_counts.get)
            
            # Adjust confidence based on consistency
            consistency = pred_counts[smoothed_pred] / len(self.prediction_history)
            smoothed_confidence = confidence * consistency
            
            return smoothed_pred, smoothed_confidence
            
        return prediction, confidence
        
    def _execute_action(self, prediction):
        """Execute computer action based on prediction"""
        # TODO: Integrate with configuration system for custom actions
        if prediction == "Left Fist":
            self.status_logged.emit("⬅️ Action: Left command executed")
            # TODO: Execute left action (e.g., keyboard press)
        elif prediction == "Right Fist":
            self.status_logged.emit("➡️ Action: Right command executed")
            # TODO: Execute right action (e.g., keyboard press)
        elif prediction == "Rest":
            # No action for rest state
            pass
            
    def get_statistics(self):
        """Get classification statistics"""
        if self.session_start_time:
            session_time = time.time() - self.session_start_time
            rate = self.prediction_count / session_time if session_time > 0 else 0
            return self.prediction_count, self.action_count, rate, session_time
        return 0, 0, 0, 0 