from PySide6.QtCore    import Qt, Slot
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QFormLayout,
    QComboBox, QPushButton, QTableWidget, QTableWidgetItem, QKeySequenceEdit,
    QMessageBox, QTabWidget, QLabel
)
import sys, random
from configForm import HotKeyMapper

class Home(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Home")
    

class MainWindow(QWidget):
    """Wrapper window that provides a horizontal tab banner."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Home")

        root = QVBoxLayout(self)

        # Create a QTabWidget (tabs on top by default)
        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.North)      # North = horizontal banner
        tabs.setMovable(False)                     # optional

        # 2Add pages
        tabs.addTab(Home(), "Home")
        tabs.addTab(HotKeyMapper(), "Config")
        tabs.addTab(QLabel("Settings coming soon…"), "Settings")
        tabs.addTab(QLabel("About page…"),        "About")

        # Drop the tab widget into the root layout
        root.addWidget(tabs)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(500, 300)
    window.show()
    sys.exit(app.exec())




    



#if __name__ == "__main__":
#    app = QApplication(sys.argv)
#    app.setAttribute(Qt.AA_EnableHighDpiScaling)

#    w = HotKeyMapper()
#    w.resize(500, 400)
#    w.show()

#    sys.exit(app.exec())
