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
import json
from configForm import HotKeyMapper



class Home(QWidget):
    def __init__(self, cfg_path: Path | str = "./src/gui/db/config.json"):
        super().__init__()
        self.setWindowTitle("Home")

        layout = QVBoxLayout(self)

        self.table = QTableWidget(0, 2)               # rows will be added
        self.table.setHorizontalHeaderLabels(["Shortcut", "Command"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)

        layout.addWidget(self.table)

        self._load_mappings(Path(cfg_path))

        if self.table.rowCount() == 0:
            layout.addWidget(QLabel("No mappings found yet."))

    def _load_mappings(self, path: Path) -> None:
        """Read JSON and fill the table."""
        if not path.exists() or path.stat().st_size == 0:
            return                                              # nothing to load

        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            return                                              # corrupted file

        if isinstance(data, list):
            merged = {}
            for entry in data:
                if isinstance(entry, dict):
                    merged.update(entry)
            data = merged

        if not isinstance(data, dict):
            return                                              # unexpected format

        # Fill the table
        for shortcut, command in data.items():
            r = self.table.rowCount()
            self.table.insertRow(r)
            self.table.setItem(r, 0, QTableWidgetItem(shortcut))
            self.table.setItem(r, 1, QTableWidgetItem(command))

    

class MainWindow(QWidget):
    """Wrapper window that provides a horizontal tab banner."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Home")

        #on off button
        self.toggle = QToolButton()
        self.toggle.setCheckable(True)            # makes it act like an on/off switch
        self.toggle.setChecked(False)             # default = OFF
        self.toggle.setText("OFF")                # label so users know the state

        # Colour rules: blue when off, green when on style
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

        #tab louout
        # Create a QTabWidget (tabs on top by default)
        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.North)      # North = horizontal banner
        tabs.setMovable(False)                     # optional

        #logo
        logo = QLabel()
        pix = QPixmap("./src/gui/images/EASY_BCI_logo.png"); 
        logo.setPixmap(pix)
        logo.setPixmap(pix.scaledToHeight(24, Qt.SmoothTransformation))
        if pix.isNull():
            print("Logo failed to load check the path!")
        tabs.setCornerWidget(logo, Qt.TopLeftCorner) 

        # Add pages
        tabs.addTab(Home(), "Home")
        tabs.addTab(HotKeyMapper(), "Config")
        tabs.addTab(CountdownApp(), "Collect")
        tabs.addTab(QLabel("About page…"),        "About")

        # on/off in tab bar
        self.toggle.toggled.connect(self.update_label)
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
