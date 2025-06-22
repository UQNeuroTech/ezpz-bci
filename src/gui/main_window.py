"""
Enhanced Main Window for ezpz-BCI GUI
Integrates all functionality: data collection, training, classification, configuration, and monitoring
"""
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QMenuBar, QStatusBar, QLabel, QPushButton,
    QMessageBox, QSystemTrayIcon, QMenu
)
from PySide6.QtGui import QFont, QIcon, QAction

# Import all tab classes
from tabs.home_tab import HomeTab
from tabs.data_collection_tab import DataCollectionTab
from tabs.training_tab import TrainingTab
from tabs.classification_tab import ClassificationTab
from tabs.monitoring_tab import MonitoringTab
from configForm import HotKeyMapper  # Keep the existing config form

class MainWindow(QMainWindow):
    # Signals for inter-tab communication
    tab_switch_requested = Signal(int)
    status_update = Signal(str, str)  # (component, status)
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ezpz-BCI Control Center")
        self.setMinimumSize(1000, 700)
        
        # Initialize UI components
        self.init_ui()
        self.init_menu_bar()
        self.init_status_bar()
        self.init_system_tray()
        
        # Connect signals
        self.connect_signals()
        
        # Initialize timers for updates
        self.init_timers()
        
    def init_ui(self):
        """Initialize the main user interface"""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)
        self.tabs.setMovable(False)
        
        # Create and add all tabs
        self.home_tab = HomeTab()
        self.data_collection_tab = DataCollectionTab()
        self.training_tab = TrainingTab()
        self.classification_tab = ClassificationTab()
        self.config_tab = HotKeyMapper()  # Enhanced existing config
        self.monitoring_tab = MonitoringTab()
        
        # Add tabs to tab widget
        self.tabs.addTab(self.home_tab, "🏠 Home")
        self.tabs.addTab(self.data_collection_tab, "📊 Data Collection")
        self.tabs.addTab(self.training_tab, "🤖 Training")
        self.tabs.addTab(self.classification_tab, "🧠 Live Classification")
        self.tabs.addTab(self.config_tab, "⚙️ Configuration")
        self.tabs.addTab(self.monitoring_tab, "📈 Monitoring")
        
        # Add tab widget to layout
        layout.addWidget(self.tabs)
        
        # Apply styling
        self.apply_styling()
        
    def init_menu_bar(self):
        """Initialize the menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        # New session action
        new_session_action = QAction("&New Session", self)
        new_session_action.setShortcut("Ctrl+N")
        new_session_action.triggered.connect(self.new_session)
        file_menu.addAction(new_session_action)
        
        # Load session action
        load_session_action = QAction("&Load Session", self)
        load_session_action.setShortcut("Ctrl+O")
        load_session_action.triggered.connect(self.load_session)
        file_menu.addAction(load_session_action)
        
        # Save session action
        save_session_action = QAction("&Save Session", self)
        save_session_action.setShortcut("Ctrl+S")
        save_session_action.triggered.connect(self.save_session)
        file_menu.addAction(save_session_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("&Tools")
        
        # Device connection action
        connect_device_action = QAction("&Connect Device", self)
        connect_device_action.triggered.connect(self.quick_connect_device)
        tools_menu.addAction(connect_device_action)
        
        # Quick training action
        quick_train_action = QAction("&Quick Train", self)
        quick_train_action.triggered.connect(self.quick_train)
        tools_menu.addAction(quick_train_action)
        
        # Emergency stop action
        emergency_stop_action = QAction("&Emergency Stop", self)
        emergency_stop_action.setShortcut("Ctrl+Shift+S")
        emergency_stop_action.triggered.connect(self.emergency_stop)
        tools_menu.addAction(emergency_stop_action)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        
        # Full screen action
        fullscreen_action = QAction("&Full Screen", self)
        fullscreen_action.setShortcut("F11")
        fullscreen_action.triggered.connect(self.toggle_fullscreen)
        view_menu.addAction(fullscreen_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        # About action
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
        # User guide action
        guide_action = QAction("&User Guide", self)
        guide_action.triggered.connect(self.show_user_guide)
        help_menu.addAction(guide_action)
        
    def init_status_bar(self):
        """Initialize the status bar"""
        self.status_bar = self.statusBar()
        
        # System status indicators
        self.device_status_label = QLabel("Device: Disconnected")
        self.device_status_label.setStyleSheet("color: #d32f2f;")
        self.status_bar.addPermanentWidget(self.device_status_label)
        
        self.model_status_label = QLabel("Model: Not Loaded")
        self.model_status_label.setStyleSheet("color: #d32f2f;")
        self.status_bar.addPermanentWidget(self.model_status_label)
        
        self.classification_status_label = QLabel("Classification: Stopped")
        self.classification_status_label.setStyleSheet("color: #f57c00;")
        self.status_bar.addPermanentWidget(self.classification_status_label)
        
        # Default status message
        self.status_bar.showMessage("Ready - Welcome to ezpz-BCI!")
        
    def init_system_tray(self):
        """Initialize system tray icon (if supported)"""
        if QSystemTrayIcon.isSystemTrayAvailable():
            self.tray_icon = QSystemTrayIcon(self)
            
            # Create tray menu
            tray_menu = QMenu()
            
            show_action = tray_menu.addAction("Show ezpz-BCI")
            show_action.triggered.connect(self.show)
            
            tray_menu.addSeparator()
            
            quit_action = tray_menu.addAction("Quit")
            quit_action.triggered.connect(self.close)
            
            self.tray_icon.setContextMenu(tray_menu)
            self.tray_icon.activated.connect(self.tray_icon_activated)
            
            # TODO: Set actual icon
            # self.tray_icon.setIcon(QIcon("icon.png"))
            self.tray_icon.show()
            
    def connect_signals(self):
        """Connect signals between components"""
        # Tab switching
        self.tab_switch_requested.connect(self.switch_to_tab)
        
        # Status updates
        self.status_update.connect(self.update_status_display)
        
        # Home tab quick action signals
        self.home_tab.switch_to_tab.connect(self.switch_to_tab)
        
        # Inter-tab communication
        # TODO: Connect specific tab signals
        
    def init_timers(self):
        """Initialize update timers"""
        # Status update timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_system_status)
        self.status_timer.start(1000)  # Update every second
        
        # Performance monitoring timer
        self.performance_timer = QTimer()
        self.performance_timer.timeout.connect(self.update_performance_metrics)
        self.performance_timer.start(5000)  # Update every 5 seconds
        
    def apply_styling(self):
        """Apply custom styling to the application"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            
            QTabWidget::pane {
                border: 1px solid #c0c0c0;
                background-color: white;
            }
            
            QTabWidget::tab-bar {
                alignment: center;
            }
            
            QTabBar::tab {
                background-color: #e0e0e0;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 2px solid #2196f3;
            }
            
            QTabBar::tab:hover {
                background-color: #f0f0f0;
            }
            
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            
            QPushButton {
                background-color: #2196f3;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            
            QPushButton:hover {
                background-color: #1976d2;
            }
            
            QPushButton:pressed {
                background-color: #0d47a1;
            }
            
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        
    def switch_to_tab(self, tab_index):
        """Switch to specified tab"""
        if 0 <= tab_index < self.tabs.count():
            self.tabs.setCurrentIndex(tab_index)
            
    def update_status_display(self, component, status):
        """Update status display for a component"""
        if component == "device":
            if "connected" in status.lower():
                self.device_status_label.setText(f"Device: {status}")
                self.device_status_label.setStyleSheet("color: #388e3c;")
                self.home_tab.update_device_status(True)
            else:
                self.device_status_label.setText(f"Device: {status}")
                self.device_status_label.setStyleSheet("color: #d32f2f;")
                self.home_tab.update_device_status(False)
                
        elif component == "model":
            if "loaded" in status.lower():
                self.model_status_label.setText(f"Model: {status}")
                self.model_status_label.setStyleSheet("color: #388e3c;")
                self.home_tab.update_model_status(True)
            else:
                self.model_status_label.setText(f"Model: {status}")
                self.model_status_label.setStyleSheet("color: #d32f2f;")
                self.home_tab.update_model_status(False)
                
        elif component == "classification":
            if "running" in status.lower():
                self.classification_status_label.setText(f"Classification: {status}")
                self.classification_status_label.setStyleSheet("color: #388e3c;")
                self.home_tab.update_classification_status(True)
            else:
                self.classification_status_label.setText(f"Classification: {status}")
                self.classification_status_label.setStyleSheet("color: #f57c00;")
                self.home_tab.update_classification_status(False)
                
    def update_system_status(self):
        """Update system status periodically"""
        # TODO: Implement actual status checking
        pass
        
    def update_performance_metrics(self):
        """Update performance metrics periodically"""
        # TODO: Implement actual performance monitoring
        pass
        
    # Menu action handlers
    def new_session(self):
        """Start a new session"""
        reply = QMessageBox.question(
            self, "New Session", 
            "Are you sure you want to start a new session? This will reset all current data.",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # TODO: Implement session reset
            self.status_bar.showMessage("New session started")
            
    def load_session(self):
        """Load a saved session"""
        # TODO: Implement session loading
        self.status_bar.showMessage("Session loading not yet implemented")
        
    def save_session(self):
        """Save current session"""
        # TODO: Implement session saving
        self.status_bar.showMessage("Session saving not yet implemented")
        
    def quick_connect_device(self):
        """Quick device connection"""
        # Switch to classification tab and trigger connection
        self.switch_to_tab(3)  # Classification tab
        self.classification_tab.connect_device()
        
    def quick_train(self):
        """Quick training shortcut"""
        # Switch to training tab
        self.switch_to_tab(2)  # Training tab
        self.status_bar.showMessage("Switched to training tab")
        
    def emergency_stop(self):
        """Emergency stop all operations"""
        reply = QMessageBox.warning(
            self, "Emergency Stop", 
            "This will immediately stop all running operations. Continue?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # TODO: Implement emergency stop
            self.status_bar.showMessage("Emergency stop activated!")
            
    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()
            
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self, "About ezpz-BCI",
            "<h3>ezpz-BCI Control Center</h3>"
            "<p>Version 1.0.0</p>"
            "<p>Brain-Computer Interface system for EEG-based computer control.</p>"
            "<p>Built with PySide6 and PyTorch.</p>"
            "<p><b>BrainHack 2025 Brisbane</b></p>"
            "<p>Free and open source software.</p>"
        )
        
    def show_user_guide(self):
        """Show user guide"""
        # TODO: Implement user guide
        QMessageBox.information(
            self, "User Guide",
            "User guide will be available in a future version.\n\n"
            "For now, explore the tabs to get started:\n"
            "1. Collect training data\n"
            "2. Train your model\n"
            "3. Configure key mappings\n"
            "4. Start live classification!"
        )
        
    def tray_icon_activated(self, reason):
        """Handle system tray icon activation"""
        if reason == QSystemTrayIcon.DoubleClick:
            if self.isVisible():
                self.hide()
            else:
                self.show()
                self.raise_()
                self.activateWindow()
                
    def closeEvent(self, event):
        """Handle application close event"""
        # Check if any operations are running
        # TODO: Implement proper shutdown checks
        
        reply = QMessageBox.question(
            self, "Exit ezpz-BCI",
            "Are you sure you want to exit?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # TODO: Cleanup operations
            event.accept()
        else:
            event.ignore()


def main():
    """Main application entry point"""
    app = QApplication([])
    app.setApplicationName("ezpz-BCI")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("BrainHack 2025")
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Start event loop
    return app.exec()


if __name__ == "__main__":
    main() 