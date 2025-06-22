from PySide6.QtCore    import Qt, Slot, QFileSystemWatcher
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QFormLayout,
    QComboBox, QPushButton, QTableWidget, QTableWidgetItem, QKeySequenceEdit,
    QMessageBox, QTabWidget, QLabel, QToolButton
)
from src.gui.training_page import TrainingPage
from pathlib import Path
from PySide6.QtGui import QPixmap
import sys, random
import json
from src.gui.configForm import HotKeyMapper
from src.gui.collection_page import CountdownApp
from src.gui.info_page import InfoPage


class Home(QWidget):
    def __init__(self, cfg_path: Path | str = "./src/gui/db/config.json"):
        super().__init__()
        self.setWindowTitle("Home")
        self.cfg_path = Path(cfg_path)

        layout = QVBoxLayout(self)

        # three columns: Shortcut | Command | (Remove button)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Shortcut", "Command", ""])
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)

        layout.addWidget(self.table)

        self._load_mappings()

        if self.table.rowCount() == 0:
            layout.addWidget(QLabel("No mappings found yet."))

    # UI helpers
    def _add_row(self, shortcut: str, command: str) -> None:
        r = self.table.rowCount()
        self.table.insertRow(r)

        self.table.setItem(r, 0, QTableWidgetItem(shortcut))
        self.table.setItem(r, 1, QTableWidgetItem(command))

        # Remove-row button
        btn = QPushButton("Remove")
        btn.clicked.connect(lambda _, sc=shortcut: self._remove_mapping(sc))
        self.table.setCellWidget(r, 2, btn)

    def _load_mappings(self) -> None:
        """Load JSON file and populate table."""
        if not self.cfg_path.exists() or self.cfg_path.stat().st_size == 0:
            return

        try:
            data = json.loads(self.cfg_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return

        # Allow either dict or legacy list format
        if isinstance(data, list):
            merged = {}
            for d in data:
                if isinstance(d, dict):
                    merged.update(d)
            data = merged

        if not isinstance(data, dict):
            return

        for shortcut, command in data.items():
            self._add_row(shortcut, command)

    # Remove logic
    @Slot(str)
    def _remove_mapping(self, shortcut: str) -> None:
        # 1. Ask for confirmation (optional)
        if QMessageBox.question(
            self, "Delete mapping",
            f"Remove shortcut '{shortcut}'?\n(This can’t be undone)",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        ) != QMessageBox.Yes:
            return

        # 2. Update JSON on disk
        if self.cfg_path.exists() and self.cfg_path.stat().st_size:
            try:
                data = json.loads(self.cfg_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = {}
        else:
            data = {}

        # Normalise list→dict if needed
        if isinstance(data, list):
            tmp = {}
            for d in data:
                if isinstance(d, dict):
                    tmp.update(d)
            data = tmp

        if shortcut in data:
            data.pop(shortcut)
            tmp_file = self.cfg_path.with_suffix(".tmp")
            tmp_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
            tmp_file.replace(self.cfg_path)

        # 3. Remove row from UI
        for row in range(self.table.rowCount()):
            if self.table.item(row, 0).text() == shortcut:
                self.table.removeRow(row)
                break


class MainWindow(QWidget):
    """Wrapper window that provides a horizontal tab banner."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EZPZ-BCI")

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
        tabs.addTab(TrainingPage(), "Train")
        tabs.addTab(CountdownApp(), "Collect")
        tabs.addTab(InfoPage(),        "Info")

        # on/off in tab bar
        self.toggle.toggled.connect(self.update_label)
        tabs.setCornerWidget(self.toggle, Qt.TopRightCorner)

        # Drop the tab widget into the root layout
        root.addWidget(tabs)
    def update_label(self, checked: bool) -> None:
        self.toggle.setText("ON" if checked else "OFF")

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(700, 500)
    window.show()
    sys.exit(app.exec())




    



#if __name__ == "__main__":
#    app = QApplication(sys.argv)
#    app.setAttribute(Qt.AA_EnableHighDpiScaling)

#    w = HotKeyMapper()
#    w.resize(500, 400)
#    w.show()

#    sys.exit(app.exec())
