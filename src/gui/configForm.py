"""
keyboard_mapper.py – A tiny GUI for mapping commands to Ctrl+Alt+<letter> shortcuts
Tested with Python 3.11 + PySide6 6.7
"""
import os
import json
import sys, random
from pathlib import Path
from PySide6.QtCore    import Qt, Slot
from PySide6.QtGui     import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QFormLayout,
    QComboBox, QPushButton, QTableWidget, QTableWidgetItem, QKeySequenceEdit,
    QMessageBox
)

# -------------------------------------------------------------------- #
# 1.  Define the commands your hot-keys will launch                     #
# -------------------------------------------------------------------- #
path = Path('./db/config.json')
def say_hi():
    print("hi")

def say_bye():
    print("bye")

def random_number():
    print("test")

COMMANDS = {
    "Say Hi":        say_hi,
    "Say Bye":       say_bye,
    "Random number": random_number,
}

# -------------------------------------------------------------------- #
# 2.  Main widget                                                      #
# -------------------------------------------------------------------- #
class HotKeyMapper(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hot-Key Mapper")

        # ----- form: choose command + capture shortcut ----------------
        self.command_box = QComboBox()
        self.command_box.addItems(COMMANDS.keys())

        self.key_edit = QKeySequenceEdit()

        self.add_btn = QPushButton("Add Mapping")
        self.add_btn.clicked.connect(self.add_mapping)

        form = QFormLayout()
        form.addRow("Command:", self.command_box)
        form.addRow("Shortcut:", self.key_edit)
        form.addRow("", self.add_btn)

        # ----- table: show current mappings ---------------------------
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Command", "Shortcut"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)

        # ----- overall layout -----------------------------------------
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.table)

        # storage for live QShortcuts
        self._shortcuts = {}   # {QKeySequence().toString(): QShortcut}

    # ---------------------------------------------------------------- #
    # 3.  Add / replace a mapping                                      #
    # ---------------------------------------------------------------- #
    @Slot()
    def add_mapping(self):
        cmd_name   = self.command_box.currentText()
        key_seq    = self.key_edit.keySequence()

        if key_seq.isEmpty():
            QMessageBox.warning(self, "No shortcut",
                                "Press a key combination first.")
            return

        key_str = key_seq.toString(QKeySequence.NativeText)

        # If the shortcut already exists, remove the old QShortcut
        if key_str in self._shortcuts:
            self._shortcuts.pop(key_str).deleteLater()
            self._replace_row(key_str, cmd_name)
        else:
            self._insert_row(cmd_name, key_str)

        add_to_file(key_str, cmd_name, path)
        # Create a live QShortcut bound to this window
        sc = QShortcut(key_seq, self)
        sc.activated.connect(COMMANDS[cmd_name])
        self._shortcuts[key_str] = sc

        # reset editor for the next entry
        self.key_edit.clear()

    # ----- helpers to keep the table in sync -------------------------
    def _insert_row(self, cmd, key):
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(cmd))
        self.table.setItem(r, 1, QTableWidgetItem(key))

    def _replace_row(self, key, new_cmd):
        for row in range(self.table.rowCount()):
            if self.table.item(row, 1).text() == key:
                self.table.item(row, 0).setText(new_cmd)
                break

    
def add_to_file(key, cmd, path):
    
    if path.exists() and path.stat().st_size:
        try:
            with path.open("r", encoding="utf-8") as f:
                data: dict[str, str] = json.load(f)
        except json.JSONDecodeError:
            data = {}           # corrupted file → reset
    else:
        data = {}

    # 2. Insert / overwrite
    data[key] = cmd

    # 3. Rewrite atomically
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)

# -------------------------------------------------------------------- #
# 4.  Run it                                                           #
# -------------------------------------------------------------------- ##
#if __name__ == "__main__":
#    app = QApplication(sys.argv)
#    app.setAttribute(Qt.AA_EnableHighDpiScaling)

#    w = HotKeyMapper()
#    w.resize(500, 400)
#    w.show()

#    sys.exit(app.exec())
