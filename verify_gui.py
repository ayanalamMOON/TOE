#!/usr/bin/env python3
"""
EG-QGEM GUI Verification Script
==============================

This script verifies that all GUI components are properly installed and configured.
"""

import sys
import os
import importlib.util

def check_module(module_name, display_name=None):
    """Check if a module can be imported."""
    if display_name is None:
        display_name = module_name

    try:
        __import__(module_name)
        print(f"✅ {display_name}")
        return True
    except ImportError as e:
        print(f"❌ {display_name}: {e}")
        return False
    except Exception as e:
        print(f"⚠️  {display_name}: {e}")
        return False

def check_file_exists(filepath, description):
    """Check if a file exists."""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} (not found)")
        return False

def main():
    """Run all verification checks."""
    print("🔍 EG-QGEM GUI Verification")
    print("=" * 40)

    # Check Python version
    print(f"🐍 Python version: {sys.version.split()[0]}")
    print()

    # Check core dependencies
    print("📦 Core Dependencies:")
    core_ok = True
    core_ok &= check_module("numpy", "NumPy")
    core_ok &= check_module("scipy", "SciPy")
    core_ok &= check_module("matplotlib", "Matplotlib")
    core_ok &= check_module("plotly", "Plotly")
    core_ok &= check_module("qiskit", "Qiskit")
    print()

    # Check GUI dependencies
    print("🖥️  GUI Dependencies:")
    gui_ok = True
    gui_ok &= check_module("PyQt5", "PyQt5")
    gui_ok &= check_module("PyQt5.QtWidgets", "PyQt5 Widgets")
    gui_ok &= check_module("PyQt5.QtCore", "PyQt5 Core")
    gui_ok &= check_module("PyQt5.QtGui", "PyQt5 GUI")
    gui_ok &= check_module("seaborn", "Seaborn")
    print()

    # Check EG-QGEM modules
    print("🔬 EG-QGEM Modules:")
    egqgem_ok = True
    egqgem_ok &= check_module("theory.constants", "Theory Constants")
    egqgem_ok &= check_module("simulations.spacetime_emergence", "Spacetime Emergence")
    egqgem_ok &= check_module("simulations.black_hole_simulator", "Black Hole Simulator")
    egqgem_ok &= check_module("experiments.predictions", "Experimental Predictions")
    egqgem_ok &= check_module("visualization.plotting", "Visualization")
    print()

    # Check GUI files
    print("📁 GUI Files:")
    files_ok = True
    files_ok &= check_file_exists("gui_interface.py", "Main GUI Interface")
    files_ok &= check_file_exists("launch_gui.py", "GUI Launcher")
    files_ok &= check_file_exists("research_interface.py", "CLI Interface")
    files_ok &= check_file_exists("requirements.txt", "Requirements File")
    print()

    # Check PyQt5 version
    print("🔧 Version Information:")
    try:
        from PyQt5.QtCore import PYQT_VERSION_STR, QT_VERSION_STR
        print(f"✅ PyQt5 version: {PYQT_VERSION_STR}")
        print(f"✅ Qt version: {QT_VERSION_STR}")
    except Exception as e:
        print(f"⚠️  Version check failed: {e}")
    print()

    # Display environment info
    print("🌍 Environment:")
    print(f"✅ Display: {os.environ.get('DISPLAY', 'Not set')}")
    print(f"✅ QT Platform: {os.environ.get('QT_QPA_PLATFORM', 'Default')}")
    print()

    # Summary
    print("📊 Verification Summary:")
    print("=" * 25)

    all_ok = core_ok and gui_ok and egqgem_ok and files_ok

    if all_ok:
        print("🎉 All components verified successfully!")
        print()
        print("✅ Ready to run EG-QGEM GUI interface")
        print("👉 Run: python launch_gui.py")
    else:
        print("⚠️  Some components need attention:")
        if not core_ok:
            print("   • Install missing core dependencies")
        if not gui_ok:
            print("   • Install PyQt5: pip install PyQt5")
        if not egqgem_ok:
            print("   • Check EG-QGEM module imports")
        if not files_ok:
            print("   • Verify GUI files are present")

    print()
    return all_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
