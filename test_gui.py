#!/usr/bin/env python3
"""
Test script for EG-QGEM GUI Interface
=====================================

This script validates that the GUI components can be imported and instantiated
without requiring a display.
"""

import sys
import os
import tempfile

# Set up Qt for headless operation
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

try:
    # Test imports
    print("Testing GUI imports...")
    from gui_interface import EGQGEMMainWindow, SimulationWorker, PlotCanvas
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import QThread

    print("✓ All imports successful")

    # Test application creation
    print("\nTesting application creation...")
    app = QApplication(sys.argv)
    print("✓ QApplication created")

    # Test main window creation
    print("\nTesting main window creation...")
    window = EGQGEMMainWindow()
    print("✓ Main window created")

    # Test worker thread creation
    print("\nTesting worker thread creation...")
    test_params = {
        'n_subsystems': 10,
        'steps': 20,
        'pattern': 'local',
        'dimension': 3
    }
    worker = SimulationWorker('spacetime', test_params)
    print("✓ Worker thread created")

    # Test plot canvas creation
    print("\nTesting plot canvas creation...")
    canvas = PlotCanvas(window)
    print("✓ Plot canvas created")

    print("\n🎉 All GUI components validated successfully!")
    print("\nTo run the GUI with a display, use:")
    print("python gui_interface.py")

except Exception as e:
    print(f"❌ Error testing GUI: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
