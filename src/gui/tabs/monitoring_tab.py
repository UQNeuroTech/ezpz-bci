"""
Monitoring Tab - Real-time EEG signal visualization and system monitoring
"""
import numpy as np
import time
from collections import deque

# Optional psutil import for system monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("psutil not available - system monitoring will use simulated data")
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QGroupBox, QProgressBar,
    QComboBox, QTextEdit, QSlider, QCheckBox,
    QSpinBox, QTabWidget
)
from PySide6.QtGui import QFont

# Import matplotlib for real-time plotting
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation

class MonitoringTab(QWidget):
    def __init__(self):
        super().__init__()
        self.monitoring_active = False
        
        # EEG data buffers for real-time plotting
        self.sampling_rate = 250  # Hz
        self.n_channels = 8
        self.time_window = 5  # seconds
        self.buffer_size = self.sampling_rate * self.time_window
        
        # Initialize data buffers
        self.eeg_buffers = [deque(maxlen=self.buffer_size) for _ in range(self.n_channels)]
        self.time_buffer = deque(maxlen=self.buffer_size)
        
        # Channel names
        self.channel_names = ['C3', 'C4', 'Cz', 'F3', 'F4', 'P3', 'P4', 'Pz']
        
        # Signal quality tracking
        self.signal_qualities = [85] * self.n_channels  # Initialize with good quality
        
        # Performance metrics
        self.cpu_usage = 0
        self.memory_usage = 0
        self.gpu_usage = 0
        
        # Timers for updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_displays)
        
        self.performance_timer = QTimer()
        self.performance_timer.timeout.connect(self.update_performance_metrics_real_time)
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the monitoring tab UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Header
        header = self.create_header()
        layout.addWidget(header)
        
        # Create sub-tabs for different monitoring views
        self.sub_tabs = QTabWidget()
        
        # EEG Signals tab
        signals_tab = self.create_signals_tab()
        self.sub_tabs.addTab(signals_tab, "📊 Live Signals")
        
        # System Performance tab
        performance_tab = self.create_performance_tab()
        self.sub_tabs.addTab(performance_tab, "⚡ Performance")
        
        # Model Analysis tab
        analysis_tab = self.create_analysis_tab()
        self.sub_tabs.addTab(analysis_tab, "🔍 Model Analysis")
        
        layout.addWidget(self.sub_tabs)
        
    def create_header(self):
        """Create header section"""
        header_widget = QWidget()
        layout = QVBoxLayout(header_widget)
        
        title = QLabel("📈 System Monitoring")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title.setFont(title_font)
        
        description = QLabel(
            "Monitor real-time EEG signals, system performance, and model behavior.\n"
            "Visualize data quality and classification confidence in real-time."
        )
        description.setStyleSheet("color: #666; font-size: 12px;")
        
        layout.addWidget(title)
        layout.addWidget(description)
        
        return header_widget
        
    def create_signals_tab(self):
        """Create EEG signals monitoring tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Control section
        control_group = QGroupBox("Signal Display Controls")
        control_layout = QGridLayout(control_group)
        
        # Time window
        control_layout.addWidget(QLabel("Time Window:"), 0, 0)
        self.time_window_spin = QSpinBox()
        self.time_window_spin.setRange(1, 30)
        self.time_window_spin.setValue(5)
        self.time_window_spin.setSuffix(" seconds")
        control_layout.addWidget(self.time_window_spin, 0, 1)
        
        # Channel selection
        control_layout.addWidget(QLabel("Channels:"), 0, 2)
        self.channel_combo = QComboBox()
        self.channel_combo.addItems(["All Channels", "C3, C4, Cz", "Custom Selection"])
        control_layout.addWidget(self.channel_combo, 0, 3)
        
        # Amplitude scaling
        control_layout.addWidget(QLabel("Amplitude Scale:"), 1, 0)
        self.amplitude_slider = QSlider(Qt.Horizontal)
        self.amplitude_slider.setRange(1, 10)
        self.amplitude_slider.setValue(5)
        control_layout.addWidget(self.amplitude_slider, 1, 1)
        
        # Filters
        control_layout.addWidget(QLabel("Filters:"), 1, 2)
        self.filter_checkbox = QCheckBox("Apply 0.5-40Hz bandpass")
        self.filter_checkbox.setChecked(True)
        control_layout.addWidget(self.filter_checkbox, 1, 3)
        
        layout.addWidget(control_group)
        
        # Real-time EEG plot
        plot_group = QGroupBox("Live EEG Signals")
        plot_layout = QVBoxLayout(plot_group)
        
        # Create matplotlib figure and canvas
        self.eeg_figure = Figure(figsize=(12, 8))
        self.eeg_canvas = FigureCanvas(self.eeg_figure)
        self.eeg_canvas.setMinimumHeight(400)
        
        # Initialize EEG plots
        self.setup_eeg_plots()
        
        plot_layout.addWidget(self.eeg_canvas)
        
        # Plot control buttons
        plot_controls = QHBoxLayout()
        
        self.start_monitoring_btn = QPushButton("▶️ Start Monitoring")
        self.start_monitoring_btn.clicked.connect(self.start_monitoring)
        plot_controls.addWidget(self.start_monitoring_btn)
        
        self.stop_monitoring_btn = QPushButton("⏹️ Stop Monitoring")
        self.stop_monitoring_btn.clicked.connect(self.stop_monitoring)
        self.stop_monitoring_btn.setEnabled(False)
        plot_controls.addWidget(self.stop_monitoring_btn)
        
        self.clear_btn = QPushButton("🗑️ Clear")
        self.clear_btn.clicked.connect(self.clear_plots)
        plot_controls.addWidget(self.clear_btn)
        
        plot_controls.addStretch()
        plot_layout.addLayout(plot_controls)
        
        layout.addWidget(plot_group)
        
        # Signal quality indicators
        quality_group = QGroupBox("Signal Quality")
        quality_layout = QGridLayout(quality_group)
        
        self.quality_bars = []
        self.quality_labels = []
        
        for i, channel in enumerate(self.channel_names):
            row = i // 4
            col = (i % 4) * 2
            
            quality_layout.addWidget(QLabel(f"{channel}:"), row, col)
            quality_bar = QProgressBar()
            quality_bar.setRange(0, 100)
            quality_bar.setValue(self.signal_qualities[i])
            
            # Color coding for signal quality
            if self.signal_qualities[i] >= 80:
                quality_bar.setStyleSheet("QProgressBar::chunk { background-color: #4caf50; }")
            elif self.signal_qualities[i] >= 60:
                quality_bar.setStyleSheet("QProgressBar::chunk { background-color: #ff9800; }")
            else:
                quality_bar.setStyleSheet("QProgressBar::chunk { background-color: #f44336; }")
                
            quality_layout.addWidget(quality_bar, row, col + 1)
            self.quality_bars.append(quality_bar)
            
        layout.addWidget(quality_group)
        
        return widget
        
    def create_performance_tab(self):
        """Create system performance monitoring tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # System metrics
        metrics_group = QGroupBox("System Metrics")
        metrics_layout = QGridLayout(metrics_group)
        
        # CPU Usage
        metrics_layout.addWidget(QLabel("CPU Usage:"), 0, 0)
        self.cpu_bar = QProgressBar()
        self.cpu_bar.setRange(0, 100)
        self.cpu_bar.setValue(25)
        metrics_layout.addWidget(self.cpu_bar, 0, 1)
        self.cpu_label = QLabel("25%")
        metrics_layout.addWidget(self.cpu_label, 0, 2)
        
        # Memory Usage
        metrics_layout.addWidget(QLabel("Memory Usage:"), 1, 0)
        self.memory_bar = QProgressBar()
        self.memory_bar.setRange(0, 100)
        self.memory_bar.setValue(45)
        metrics_layout.addWidget(self.memory_bar, 1, 1)
        self.memory_label = QLabel("45%")
        metrics_layout.addWidget(self.memory_label, 1, 2)
        
        # GPU Usage (if available)
        metrics_layout.addWidget(QLabel("GPU Usage:"), 2, 0)
        self.gpu_bar = QProgressBar()
        self.gpu_bar.setRange(0, 100)
        self.gpu_bar.setValue(0)
        metrics_layout.addWidget(self.gpu_bar, 2, 1)
        self.gpu_label = QLabel("N/A")
        metrics_layout.addWidget(self.gpu_label, 2, 2)
        
        layout.addWidget(metrics_group)
        
        # Processing performance
        processing_group = QGroupBox("Processing Performance")
        processing_layout = QGridLayout(processing_group)
        
        # Data throughput
        processing_layout.addWidget(QLabel("Data Throughput:"), 0, 0)
        self.throughput_label = QLabel("250 samples/sec")
        processing_layout.addWidget(self.throughput_label, 0, 1)
        
        # Classification latency
        processing_layout.addWidget(QLabel("Classification Latency:"), 1, 0)
        self.latency_label = QLabel("12 ms")
        processing_layout.addWidget(self.latency_label, 1, 1)
        
        # Buffer status
        processing_layout.addWidget(QLabel("Buffer Status:"), 2, 0)
        self.buffer_bar = QProgressBar()
        self.buffer_bar.setRange(0, 100)
        self.buffer_bar.setValue(30)
        processing_layout.addWidget(self.buffer_bar, 2, 1)
        
        # Dropped samples
        processing_layout.addWidget(QLabel("Dropped Samples:"), 3, 0)
        self.dropped_label = QLabel("0 (0.0%)")
        processing_layout.addWidget(self.dropped_label, 3, 1)
        
        layout.addWidget(processing_group)
        
        # Connection status
        connection_group = QGroupBox("Connection Status")
        connection_layout = QGridLayout(connection_group)
        
        # Device connection
        connection_layout.addWidget(QLabel("EEG Device:"), 0, 0)
        self.device_status_label = QLabel("❌ Disconnected")
        self.device_status_label.setStyleSheet("color: #d32f2f;")
        connection_layout.addWidget(self.device_status_label, 0, 1)
        
        # Signal strength
        connection_layout.addWidget(QLabel("Signal Strength:"), 1, 0)
        self.signal_strength_bar = QProgressBar()
        self.signal_strength_bar.setRange(0, 100)
        self.signal_strength_bar.setValue(0)
        connection_layout.addWidget(self.signal_strength_bar, 1, 1)
        
        # Connection uptime
        connection_layout.addWidget(QLabel("Uptime:"), 2, 0)
        self.uptime_label = QLabel("00:00:00")
        connection_layout.addWidget(self.uptime_label, 2, 1)
        
        layout.addWidget(connection_group)
        
        # Performance log
        log_group = QGroupBox("Performance Log")
        log_layout = QVBoxLayout(log_group)
        
        self.performance_log = QTextEdit()
        self.performance_log.setMaximumHeight(120)
        self.performance_log.setReadOnly(True)
        self.performance_log.append("System monitoring initialized...")
        log_layout.addWidget(self.performance_log)
        
        layout.addWidget(log_group)
        
        return widget
        
    def create_analysis_tab(self):
        """Create model analysis and debugging tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Model information
        model_group = QGroupBox("Current Model Information")
        model_layout = QGridLayout(model_group)
        
        model_layout.addWidget(QLabel("Model Type:"), 0, 0)
        self.model_type_label = QLabel("EEGNet-8,2")
        model_layout.addWidget(self.model_type_label, 0, 1)
        
        model_layout.addWidget(QLabel("Input Channels:"), 1, 0)
        self.input_channels_label = QLabel("8")
        model_layout.addWidget(self.input_channels_label, 1, 1)
        
        model_layout.addWidget(QLabel("Classes:"), 2, 0)
        self.classes_label = QLabel("Rest, Left Fist, Right Fist")
        model_layout.addWidget(self.classes_label, 2, 1)
        
        model_layout.addWidget(QLabel("Training Accuracy:"), 3, 0)
        self.train_accuracy_label = QLabel("87.5%")
        model_layout.addWidget(self.train_accuracy_label, 3, 1)
        
        layout.addWidget(model_group)
        
        # Real-time predictions analysis
        predictions_group = QGroupBox("Prediction Analysis")
        predictions_layout = QVBoxLayout(predictions_group)
        
        # Confidence distribution placeholder
        self.confidence_plot_placeholder = QLabel("Confidence Distribution Plot\n(Shows distribution of prediction confidences)")
        self.confidence_plot_placeholder.setAlignment(Qt.AlignCenter)
        self.confidence_plot_placeholder.setMinimumHeight(200)
        self.confidence_plot_placeholder.setStyleSheet("""
            QLabel {
                border: 2px dashed #ccc;
                border-radius: 5px;
                background-color: #f9f9f9;
                color: #666;
                font-size: 12px;
            }
        """)
        predictions_layout.addWidget(self.confidence_plot_placeholder)
        
        layout.addWidget(predictions_group)
        
        # Feature analysis
        features_group = QGroupBox("Feature Analysis")
        features_layout = QVBoxLayout(features_group)
        
        # Feature importance placeholder
        self.features_plot_placeholder = QLabel("Feature Importance Visualization\n(Shows which EEG features are most important for classification)")
        self.features_plot_placeholder.setAlignment(Qt.AlignCenter)
        self.features_plot_placeholder.setMinimumHeight(150)
        self.features_plot_placeholder.setStyleSheet("""
            QLabel {
                border: 2px dashed #ccc;
                border-radius: 5px;
                background-color: #f9f9f9;
                color: #666;
                font-size: 12px;
            }
        """)
        features_layout.addWidget(self.features_plot_placeholder)
        
        layout.addWidget(features_group)
        
        # Debug information
        debug_group = QGroupBox("Debug Information")
        debug_layout = QVBoxLayout(debug_group)
        
        self.debug_log = QTextEdit()
        self.debug_log.setMaximumHeight(100)
        self.debug_log.setReadOnly(True)
        self.debug_log.append("Debug logging enabled...")
        debug_layout.addWidget(self.debug_log)
        
        layout.addWidget(debug_group)
        
        return widget
        
    def update_signal_quality(self, channel_qualities):
        """Update signal quality indicators"""
        # TODO: Implement signal quality updates
        pass
        
    def update_system_metrics(self, cpu, memory, gpu=None):
        """Update system performance metrics"""
        self.cpu_bar.setValue(cpu)
        self.cpu_label.setText(f"{cpu}%")
        
        self.memory_bar.setValue(memory)
        self.memory_label.setText(f"{memory}%")
        
        if gpu is not None:
            self.gpu_bar.setValue(gpu)
            self.gpu_label.setText(f"{gpu}%")
            
    def update_processing_metrics(self, throughput, latency, buffer_usage, dropped_samples):
        """Update processing performance metrics"""
        self.throughput_label.setText(f"{throughput} samples/sec")
        self.latency_label.setText(f"{latency} ms")
        self.buffer_bar.setValue(buffer_usage)
        self.dropped_label.setText(f"{dropped_samples} ({dropped_samples/1000:.1f}%)")
        
    def update_connection_status(self, connected, signal_strength, uptime):
        """Update connection status"""
        if connected:
            self.device_status_label.setText("✅ Connected")
            self.device_status_label.setStyleSheet("color: #388e3c;")
            self.signal_strength_bar.setValue(signal_strength)
        else:
            self.device_status_label.setText("❌ Disconnected")
            self.device_status_label.setStyleSheet("color: #d32f2f;")
            self.signal_strength_bar.setValue(0)
            
        # Format uptime
        hours = uptime // 3600
        minutes = (uptime % 3600) // 60
        seconds = uptime % 60
        self.uptime_label.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
        
    def log_performance(self, message):
        """Add message to performance log"""
        self.performance_log.append(f"• {message}")
        
    def log_debug(self, message):
        """Add message to debug log"""
        self.debug_log.append(f"• {message}")
        
    def setup_eeg_plots(self):
        """Initialize the EEG plotting axes"""
        self.eeg_figure.clear()
        
        # Create subplots for each channel
        self.eeg_axes = []
        for i in range(self.n_channels):
            ax = self.eeg_figure.add_subplot(self.n_channels, 1, i + 1)
            ax.set_ylabel(f'{self.channel_names[i]}\n(μV)', fontsize=8)
            ax.set_xlim(0, self.time_window)
            ax.set_ylim(-100, 100)  # Typical EEG amplitude range
            ax.grid(True, alpha=0.3)
            
            # Remove x-axis labels except for bottom plot
            if i < self.n_channels - 1:
                ax.set_xticks([])
            else:
                ax.set_xlabel('Time (seconds)')
                
            self.eeg_axes.append(ax)
            
        self.eeg_figure.tight_layout()
        self.eeg_canvas.draw()
        
    def start_monitoring(self):
        """Start real-time EEG monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.start_monitoring_btn.setEnabled(False)
            self.stop_monitoring_btn.setEnabled(True)
            
            # Clear buffers
            self.clear_buffers()
            
            # Start update timers
            self.update_timer.start(40)  # 25 FPS for smooth visualization
            self.performance_timer.start(1000)  # 1 Hz for performance metrics
            
            self.log_performance("🔴 Real-time monitoring started")
            
    def stop_monitoring(self):
        """Stop real-time EEG monitoring"""
        if self.monitoring_active:
            self.monitoring_active = False
            self.start_monitoring_btn.setEnabled(True)
            self.stop_monitoring_btn.setEnabled(False)
            
            # Stop timers
            self.update_timer.stop()
            self.performance_timer.stop()
            
            self.log_performance("⏹️ Real-time monitoring stopped")
            
    def clear_plots(self):
        """Clear all EEG plots"""
        self.clear_buffers()
        self.setup_eeg_plots()
        self.log_performance("🗑️ Plots cleared")
        
    def clear_buffers(self):
        """Clear all data buffers"""
        for buffer in self.eeg_buffers:
            buffer.clear()
        self.time_buffer.clear()
        
    def generate_synthetic_eeg(self):
        """Generate synthetic EEG data for demonstration"""
        current_time = time.time()
        
        # Generate realistic EEG-like signals
        eeg_data = []
        for i in range(self.n_channels):
            # Base frequency components (alpha, beta, theta)
            alpha = 15 * np.sin(2 * np.pi * 10 * current_time + i * np.pi/4)  # 10 Hz alpha
            beta = 8 * np.sin(2 * np.pi * 20 * current_time + i * np.pi/8)   # 20 Hz beta
            theta = 20 * np.sin(2 * np.pi * 6 * current_time + i * np.pi/6)  # 6 Hz theta
            
            # Add some noise and artifacts
            noise = np.random.normal(0, 5)
            
            # Occasional "blink" artifacts
            if np.random.random() < 0.01 and i < 2:  # Frontal channels
                artifact = np.random.normal(0, 50)
            else:
                artifact = 0
                
            signal = alpha + beta + theta + noise + artifact
            eeg_data.append(signal)
            
        return eeg_data
        
    def update_displays(self):
        """Update all real-time displays"""
        if not self.monitoring_active:
            return
            
        # Generate new EEG data
        new_eeg_data = self.generate_synthetic_eeg()
        current_time = len(self.time_buffer) / self.sampling_rate
        
        # Add data to buffers
        for i, value in enumerate(new_eeg_data):
            self.eeg_buffers[i].append(value)
        self.time_buffer.append(current_time)
        
        # Update signal quality based on signal characteristics
        self.update_signal_quality_from_data(new_eeg_data)
        
        # Update EEG plots
        self.update_eeg_plots()
        
    def update_eeg_plots(self):
        """Update the EEG signal plots"""
        if len(self.time_buffer) < 2:
            return
            
        time_data = list(self.time_buffer)
        current_time = time_data[-1] if time_data else 0
        
        for i, ax in enumerate(self.eeg_axes):
            if len(self.eeg_buffers[i]) > 0:
                signal_data = list(self.eeg_buffers[i])
                
                ax.clear()
                ax.plot(time_data, signal_data, 'b-', linewidth=0.8)
                ax.set_ylabel(f'{self.channel_names[i]}\n(μV)', fontsize=8)
                ax.set_xlim(max(0, current_time - self.time_window), current_time + 0.1)
                ax.set_ylim(-100, 100)
                ax.grid(True, alpha=0.3)
                
                # Remove x-axis labels except for bottom plot
                if i < self.n_channels - 1:
                    ax.set_xticks([])
                else:
                    ax.set_xlabel('Time (seconds)')
                    
        self.eeg_canvas.draw()
        
    def update_signal_quality_from_data(self, eeg_data):
        """Update signal quality based on current EEG data characteristics"""
        for i, signal in enumerate(eeg_data):
            # Simple quality metric based on signal amplitude and noise
            amplitude = abs(signal)
            
            if amplitude < 5:  # Very low signal
                quality = max(20, self.signal_qualities[i] - 2)
            elif amplitude > 200:  # Artifact/saturation
                quality = max(10, self.signal_qualities[i] - 5)
            elif 5 <= amplitude <= 50:  # Good signal range
                quality = min(95, self.signal_qualities[i] + 1)
            else:  # Moderate signal
                quality = max(50, min(80, self.signal_qualities[i]))
                
            self.signal_qualities[i] = quality
            
            # Update quality bar
            self.quality_bars[i].setValue(quality)
            
            # Update color coding
            if quality >= 80:
                self.quality_bars[i].setStyleSheet("QProgressBar::chunk { background-color: #4caf50; }")
            elif quality >= 60:
                self.quality_bars[i].setStyleSheet("QProgressBar::chunk { background-color: #ff9800; }")
            else:
                self.quality_bars[i].setStyleSheet("QProgressBar::chunk { background-color: #f44336; }")
                
    def update_performance_metrics_real_time(self):
        """Update system performance metrics in real-time"""
        try:
            if PSUTIL_AVAILABLE:
                # Get real system metrics using psutil
                self.cpu_usage = psutil.cpu_percent(interval=None)
                self.memory_usage = psutil.virtual_memory().percent
            else:
                # Use simulated data
                import random
                self.cpu_usage = random.uniform(10, 60)
                self.memory_usage = random.uniform(30, 70)
            
            # Update CPU
            self.cpu_bar.setValue(int(self.cpu_usage))
            self.cpu_label.setText(f"{self.cpu_usage:.1f}%")
            
            # Update Memory
            self.memory_bar.setValue(int(self.memory_usage))
            self.memory_label.setText(f"{self.memory_usage:.1f}%")
            
            # GPU usage (placeholder - would need specific GPU monitoring)
            self.gpu_usage = 0  # TODO: Implement GPU monitoring
            self.gpu_bar.setValue(self.gpu_usage)
            self.gpu_label.setText("N/A" if self.gpu_usage == 0 else f"{self.gpu_usage}%")
            
            # Update processing metrics
            if self.monitoring_active and self.time_buffer:
                throughput = len(self.time_buffer) / max(1, self.time_buffer[-1] if self.time_buffer else 1)
                self.throughput_label.setText(f"{throughput:.0f} samples/sec")
                
                # Simulate realistic latency
                latency = np.random.normal(15, 3)  # 15ms ± 3ms
                self.latency_label.setText(f"{latency:.1f} ms")
                
                # Buffer usage
                buffer_usage = (len(self.time_buffer) / self.buffer_size) * 100
                self.buffer_bar.setValue(int(buffer_usage))
                
        except Exception as e:
            self.log_performance(f"❌ Error updating metrics: {str(e)}")
            
    def connect_to_external_data_source(self, data_source):
        """Connect to external EEG data source (e.g., from classification tab)"""
        # This method allows other tabs to feed real EEG data to the monitoring
        # For now, we'll use synthetic data, but this provides the interface
        pass