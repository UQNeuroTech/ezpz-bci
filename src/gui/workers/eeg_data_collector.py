"""
EEG Data Collection Worker
Integrates GUI with collect_data_openbci.py functionality
"""
import sys
import os
import time
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from PySide6.QtCore import QThread, Signal, QTimer
from PySide6.QtWidgets import QMessageBox

# Import existing EEG collection functionality
try:
    from collect_data_openbci import generate_prompt_order, add_nothing_prompts, marker_dict
    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
    BRAINFLOW_AVAILABLE = True
except ImportError:
    BRAINFLOW_AVAILABLE = False
    print("BrainFlow not available - using simulation mode")

class EEGDataCollector(QThread):
    # Signals for communication with GUI
    prompt_updated = Signal(str, str)  # (main_text, instruction_text)
    progress_updated = Signal(int, int)  # (completed_trials, total_trials)
    trial_progress_updated = Signal(int, int)  # (progress_percent, time_remaining)
    status_logged = Signal(str)  # status message
    collection_finished = Signal(bool, str)  # (success, message)
    connection_status_changed = Signal(bool)  # connected status
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_collecting = False
        self.is_paused = False
        self.board = None
        self.device_connected = False
        
        # Collection parameters
        self.device_type = "OpenBCI Cyton"
        self.trials_per_class = 10
        self.trial_duration = 3
        self.rest_duration = 2
        
        # Data storage
        self.eeg_samples = []
        self.eeg_markers = []
        
    def set_parameters(self, device_type, trials_per_class, trial_duration, rest_duration):
        """Set collection parameters"""
        self.device_type = device_type
        self.trials_per_class = trials_per_class
        self.trial_duration = trial_duration
        self.rest_duration = rest_duration
        
    def test_connection(self):
        """Test EEG device connection"""
        if not BRAINFLOW_AVAILABLE:
            self.status_logged.emit("⚠️ BrainFlow not available - using simulation mode")
            self.connection_status_changed.emit(True)
            self.device_connected = True
            return True
            
        try:
            if self.device_type == "Synthetic (Demo)":
                board_id = BoardIds.SYNTHETIC_BOARD
                params = BrainFlowInputParams()
            else:
                board_id = BoardIds.CYTON_BOARD
                params = BrainFlowInputParams()
                params.serial_port = "/dev/ttyUSB0"  # Default port
                
            # Test connection
            test_board = BoardShim(board_id, params)
            test_board.prepare_session()
            test_board.release_session()
            
            self.device_connected = True
            self.connection_status_changed.emit(True)
            self.status_logged.emit(f"✅ Successfully connected to {self.device_type}")
            return True
            
        except Exception as e:
            self.device_connected = False
            self.connection_status_changed.emit(False)
            self.status_logged.emit(f"❌ Connection failed: {str(e)}")
            return False
            
    def start_collection(self):
        """Start the data collection process"""
        if not self.device_connected:
            self.status_logged.emit("❌ Device not connected")
            return
            
        self.is_collecting = True
        self.is_paused = False
        self.eeg_samples.clear()
        self.eeg_markers.clear()
        
        self.status_logged.emit("🔴 Starting data collection...")
        self.start()
        
    def stop_collection(self):
        """Stop the data collection process"""
        self.is_collecting = False
        self.status_logged.emit("⏹️ Stopping data collection...")
        
    def pause_collection(self):
        """Pause/resume data collection"""
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.status_logged.emit("⏸️ Collection paused")
        else:
            self.status_logged.emit("▶️ Collection resumed")
            
    def run(self):
        """Main collection thread execution"""
        try:
            if BRAINFLOW_AVAILABLE and self.device_type != "Synthetic (Demo)":
                self._run_real_collection()
            else:
                self._run_simulation_collection()
                
        except Exception as e:
            self.status_logged.emit(f"❌ Collection error: {str(e)}")
            self.collection_finished.emit(False, str(e))
            
    def _run_real_collection(self):
        """Run real EEG data collection using BrainFlow"""
        # Generate prompt sequence
        prompt_order = generate_prompt_order(self.trials_per_class)
        prompt_order = add_nothing_prompts(prompt_order)
        total_trials = len(prompt_order)
        
        # Setup board
        if self.device_type == "Synthetic (Demo)":
            board_id = BoardIds.SYNTHETIC_BOARD
            params = BrainFlowInputParams()
        else:
            board_id = BoardIds.CYTON_BOARD
            params = BrainFlowInputParams()
            params.serial_port = "/dev/ttyUSB0"
            
        self.board = BoardShim(board_id, params)
        self.board.prepare_session()
        self.board.start_stream()
        
        try:
            for trial_idx, marker in enumerate(prompt_order):
                if not self.is_collecting:
                    break
                    
                # Wait if paused
                while self.is_paused and self.is_collecting:
                    self.msleep(100)
                    
                if not self.is_collecting:
                    break
                    
                # Update prompt
                prompt_text = marker_dict[marker]
                instruction = f"Trial {trial_idx + 1}/{total_trials}"
                self.prompt_updated.emit(prompt_text, instruction)
                self.progress_updated.emit(trial_idx, total_trials)
                
                # Collect data for trial duration
                trial_start = time.time()
                trial_samples = []
                
                while time.time() - trial_start < self.trial_duration:
                    if not self.is_collecting:
                        break
                        
                    # Get EEG data
                    data = self.board.get_board_data()
                    channels = self.board.get_eeg_channels(board_id)
                    eeg_sample = [data[i].tolist() for i in channels]
                    trial_samples.extend(eeg_sample)
                    
                    # Update trial progress
                    elapsed = time.time() - trial_start
                    progress = int((elapsed / self.trial_duration) * 100)
                    remaining = max(0, int(self.trial_duration - elapsed))
                    self.trial_progress_updated.emit(progress, remaining)
                    
                    self.msleep(100)  # 10Hz update rate
                    
                # Store trial data
                self.eeg_samples.append(trial_samples)
                self.eeg_markers.append(marker)
                
                # Rest period
                if trial_idx < len(prompt_order) - 1:
                    self.prompt_updated.emit("Rest", f"Next trial in {self.rest_duration}s")
                    self.msleep(self.rest_duration * 1000)
                    
        finally:
            self.board.stop_stream()
            self.board.release_session()
            
        # Save collected data
        self._save_data()
        self.collection_finished.emit(True, f"Collected {len(self.eeg_samples)} trials")
        
    def _run_simulation_collection(self):
        """Run simulated data collection for demo purposes"""
        import random
        
        # Generate prompt sequence
        prompt_order = generate_prompt_order(self.trials_per_class) if BRAINFLOW_AVAILABLE else [1, 2, 3] * self.trials_per_class
        total_trials = len(prompt_order)
        
        marker_dict_sim = {0: "Nothing", 1: "Rest", 2: "Left Fist", 3: "Right Fist"}
        
        for trial_idx, marker in enumerate(prompt_order):
            if not self.is_collecting:
                break
                
            # Wait if paused
            while self.is_paused and self.is_collecting:
                self.msleep(100)
                
            if not self.is_collecting:
                break
                
            # Update prompt
            prompt_text = marker_dict_sim.get(marker, "Unknown")
            instruction = f"Trial {trial_idx + 1}/{total_trials} (Simulation)"
            self.prompt_updated.emit(prompt_text, instruction)
            self.progress_updated.emit(trial_idx, total_trials)
            
            # Simulate trial duration with progress updates
            for i in range(self.trial_duration * 10):  # 10 updates per second
                if not self.is_collecting:
                    break
                    
                progress = int((i / (self.trial_duration * 10)) * 100)
                remaining = max(0, int(self.trial_duration - (i / 10)))
                self.trial_progress_updated.emit(progress, remaining)
                
                self.msleep(100)
                
            # Generate fake EEG data
            fake_data = [[random.random() for _ in range(250)] for _ in range(8)]  # 8 channels, 1 second at 250Hz
            self.eeg_samples.append(fake_data)
            self.eeg_markers.append(marker)
            
            # Rest period
            if trial_idx < len(prompt_order) - 1:
                self.prompt_updated.emit("Rest", f"Next trial in {self.rest_duration}s")
                self.msleep(self.rest_duration * 1000)
                
        self.collection_finished.emit(True, f"Simulated collection: {len(self.eeg_samples)} trials")
        
    def _save_data(self):
        """Save collected EEG data"""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            
            # Save samples
            samples_path = data_dir / f"eeg_samples_{timestamp}.json"
            with open(samples_path, 'w') as f:
                json.dump(self.eeg_samples, f)
                
            # Save markers
            markers_path = data_dir / f"eeg_markers_{timestamp}.json"
            with open(markers_path, 'w') as f:
                json.dump(self.eeg_markers, f)
                
            self.status_logged.emit(f"💾 Data saved: {samples_path.name}")
            
        except Exception as e:
            self.status_logged.emit(f"❌ Save error: {str(e)}")
            
    def get_collected_data(self):
        """Get the collected data"""
        return self.eeg_samples, self.eeg_markers