"""
keyboard_mapper.py – A tiny GUI for mapping commands to Ctrl+Alt+<letter> shortcuts
Tested with Python 3.11 + PySide6 6.7
"""
import os
import json
import sys, random
from pathlib import Path
from PySide6.QtCore    import Qt, Slot, QFileSystemWatcher
from PySide6.QtGui     import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QFormLayout,
    QComboBox, QPushButton, QTableWidget, QTableWidgetItem, QKeySequenceEdit,
    QMessageBox
)


path = Path('./data/config.json')


# -------------------------------------------------------------------- #
# 2.  Main widget                                                      #
# -------------------------------------------------------------------- #
class HotKeyMapper(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hot-Key Mapper")

        # ----- form: choose command + capture shortcut ----------------
        self.commands = load_commands(Path("./data/categories.json"))
        print(self.commands)
        self.command_box = QComboBox()
        self.command_box.addItems(self.commands) 

        self.key_edit = QKeySequenceEdit()

        self.add_btn = QPushButton("Add Mapping")
        self.add_btn.clicked.connect(self.add_mapping)

        form = QFormLayout()
        form.addRow("Command:", self.command_box)
        form.addRow("Shortcut:", self.key_edit)
        form.addRow("", self.add_btn)

        # table: show current mappings 
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Command", "Shortcut"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)

        #overall layout
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.table)

        self.refresh()

        self.watcher = QFileSystemWatcher(["./data/categories.json"])
        self.watcher.fileChanged.connect(self.on_file_changed)

        # storage for live QShortcuts
        self._shortcuts = {}   # {QKeySequence().toString(): QShortcut}
    def refresh(self):
        self.command_box.addItems(load_commands(Path("./data/categories.json")))
    #Add / replace a mapping
    @Slot()
    def on_file_changed(self, changed_path):
        """
        QFileSystemWatcher stops watching a file if it is *removed* and
        *recreated* (common when editors do “safe save”).  So we add it back:
        """
        if changed_path not in self.watcher.files():
            self.watcher.addPath(changed_path)

        self.refresh()                     # re-run your logic
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

        sc.activated.connect(lambda name=cmd_name: print(f"Triggered: {name}"))
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
    
    path.parent.mkdir(parents=True, exist_ok=True)

    #load existing JSON
    if path.exists() and path.stat().st_size:
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            # ⇢ if the file *was* a list, squash it into one dict
            if isinstance(data, list):
                merged = {}
                for entry in data:
                    if isinstance(entry, dict):
                        merged.update(entry)
                data = merged
        except json.JSONDecodeError:
            data = {}
    else:
        data = {}

    data[key] = cmd
    #update and save
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)

def load_commands(cfg_path):
    """Read cfg_path and return a dict {Pretty Name: handler}."""
    if not cfg_path.exists():
        print("Config file missing using no commands.")
        return {}

    with cfg_path.open(encoding="utf-8") as f:
        data = json.load(f)
    list_of_commands = []

    
    for i in data.get("categories", []):
        list_of_commands.append(i)
    return list_of_commands



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
