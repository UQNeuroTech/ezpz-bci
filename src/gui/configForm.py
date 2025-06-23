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

        # collect command mappings
        self.commands = load_commands(Path("./data/categories.json"))
        self.command_box = QComboBox()
        self.command_box.addItems(self.commands) 
        self.key_edit = QKeySequenceEdit()
        self.add_btn = QPushButton("Add Mapping")
        self.add_btn.clicked.connect(self.add_mapping)

        #display command mappings
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

        #on file change refresgh table
        self.refresh()
        self.watcher = QFileSystemWatcher(["./data/categories.json"])
        self.watcher.fileChanged.connect(self.on_file_changed)

        # storage for live QShortcuts
        self._shortcuts = {}   # {QKeySequence().toString(): QShortcut}

    def refresh(self):
        """
        Refresh the comand box options for the drop down
        """
        self.command_box.addItems(load_commands(Path("./data/categories.json")))

    @Slot()
    def on_file_changed(self, changed_path):
        """
        QFileSystemWatcher stops watching a file if it is removed and
        recreated. Makes QFileSystemWatcher persistant
        """
        if changed_path not in self.watcher.files():
            self.watcher.addPath(changed_path)

        self.refresh()                     # re-run your logic

    def add_mapping(self):
        """
        add a button press to the map to command
        """

        #get the current info from the form
        cmd_name   = self.command_box.currentText()
        key_seq    = self.key_edit.keySequence()

        #make sure input is not empty
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

        #add mapping to config file
        add_to_file(key_str, cmd_name, path)

        # Create a live QShortcut bound to this window
        sc = QShortcut(key_seq, self)

        sc.activated.connect(lambda name=cmd_name: print(f"Triggered: {name}"))
        self._shortcuts[key_str] = sc

        # reset editor for the next entry
        self.key_edit.clear()

    def _insert_row(self, cmd, key):
        """
        insert a new row in the table
        """
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(cmd))
        self.table.setItem(r, 1, QTableWidgetItem(key))

    def _replace_row(self, key, new_cmd):
        """
        replace a row in the display tble
        """
        for row in range(self.table.rowCount()):
            if self.table.item(row, 1).text() == key:
                self.table.item(row, 0).setText(new_cmd)
                break

    
def add_to_file(key, cmd, path):
    """
    Add a cmd to the config file
    """
    
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
    """Read cfg_path and return a list of trained commands"""
    if not cfg_path.exists():
        print("Config file missing using no commands.")
        return {}

    with cfg_path.open(encoding="utf-8") as f:
        data = json.load(f)
    list_of_commands = []

    
    for i in data.get("categories", []):
        list_of_commands.append(i)
    return list_of_commands
