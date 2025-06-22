#!/usr/bin/env python3
"""
Launch script for ezpz-BCI GUI
Simple entry point to start the enhanced GUI application
"""

import sys
import os

# Add the parent directory to the path so we can import from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gui.main_window import main

if __name__ == "__main__":
    print("🧠 Starting ezpz-BCI Control Center...")
    print("📋 Phase 1: Enhanced GUI Architecture Complete")
    print("✨ Features: Home, Data Collection, Training, Classification, Configuration, Monitoring")
    print("")
    
    # Launch the GUI
    sys.exit(main())