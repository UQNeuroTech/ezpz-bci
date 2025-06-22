from PySide6.QtCore    import Qt, Slot
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QFormLayout,
    QComboBox, QPushButton, QTableWidget, QTableWidgetItem, QKeySequenceEdit,
    QMessageBox, QTabWidget, QLabel, QToolButton
)
from training_page import CountdownApp
from pathlib import Path
from PySide6.QtGui import QPixmap
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

        self.toggle = QToolButton()
        self.toggle.setCheckable(True)            # makes it act like an on/off switch
        self.toggle.setChecked(False)             # default = OFF
        self.toggle.setText("OFF")                # label so users know the state

        # Colour rules: blue when off, green when on
        self.toggle.setStyleSheet("""
            QToolButton {
                background-color: #FF0000;          /* blue */
                color: white;
                padding: 4px 10px;
                border: none;
                border-radius: 6px;
                font-weight: bold;
            }
            QToolButton:checked {
                background-color: #4CAF50;          /* green */
            }
        """)

        root = QVBoxLayout(self)

        # Create a QTabWidget (tabs on top by default)
        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.North)      # North = horizontal banner
        tabs.setMovable(False)                     # optional

        #add logo
        logo = QLabel()

        pix = QPixmap("./src/gui/images/EASY_BCI_logo.png"); 

        logo.setPixmap(pix)

        logo.setPixmap(pix.scaledToHeight(24, Qt.SmoothTransformation))

        if pix.isNull():
            print("Logo failed to load check the path!")
        
        tabs.setCornerWidget(logo, Qt.TopLeftCorner)   # or Qt.TopRightCorner

        # Add pages
        tabs.addTab(Home(), "Home")
        tabs.addTab(HotKeyMapper(), "Config")
        tabs.addTab(CountdownApp(), "Train")
        tabs.addTab(QLabel("About page…"),        "About")
        self.toggle.toggled.connect(self.update_label)

        # Stick it in the banner’s right corner
        tabs.setCornerWidget(self.toggle, Qt.TopRightCorner)

        # Drop the tab widget into the root layout
        root.addWidget(tabs)
    def update_label(self, checked: bool) -> None:
        self.toggle.setText("ON" if checked else "OFF")

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
