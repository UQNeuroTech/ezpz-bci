#!/usr/bin/env python

"""
Starter script that ensures the project root is in the Python path.
This makes imports work properly during development.
"""

import os
import sys

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Now import and run the main application
from src.gui.home import main

if __name__ == "__main__":
    main()
