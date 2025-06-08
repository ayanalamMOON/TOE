#!/usr/bin/env python3
"""
Simplified EG-QGEM GUI Test
==========================

A lightweight version to test GUI functionality without heavy imports.
"""

import sys
import os
import json
import time
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Set Qt platform for headless environments
if 'QT_QPA_PLATFORM' not in os.environ:
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QPushButton, QSpinBox, QDoubleSpinBox,
    QComboBox, QTextEdit, QProgressBar, QSplitter, QGroupBox,
    QFormLayout, QCheckBox, QSlider, QFrame, QScrollArea,
    QMessageBox, QFileDialog, QStatusBar, QMenuBar, QAction
)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt
from PyQt5.QtGui import QFont, QPixmap, QPalette, QColor, QIcon


class SimplePlotCanvas(FigureCanvas):
    """Matplotlib canvas for embedding plots in Qt."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.figure)
        self.setParent(parent)

        # Initialize with a sample plot
        self.axes = self.figure.add_subplot(111)
        self.axes.plot([1, 2, 3, 4], [1, 4, 2, 3])
        self.axes.set_title('Sample Plot')
        self.figure.tight_layout()


class SimpleWorker(QThread):
    """Simple worker thread for testing."""

    progress_updated = pyqtSignal(int)
    log_updated = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def run(self):
        """Run a simple simulation."""
        for i in range(101):
            time.sleep(0.01)  # Simulate work
            self.progress_updated.emit(i)
            if i % 20 == 0:
                self.log_updated.emit(f"Progress: {i}%")

        self.log_updated.emit("Simulation completed!")
        self.finished_signal.emit()


class SimpleMainWindow(QMainWindow):
    """Simplified main window for testing."""

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.worker = None

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("EG-QGEM GUI Test")
        self.setGeometry(100, 100, 1000, 700)

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Create splitter
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Left panel: Controls
        self.create_control_panel(splitter)

        # Right panel: Results
        self.create_results_panel(splitter)

        # Set splitter proportions
        splitter.setSizes([300, 700])

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def create_control_panel(self, parent):
        """Create the control panel."""
        control_widget = QWidget()
        parent.addWidget(control_widget)
        layout = QVBoxLayout(control_widget)

        # Title
        title = QLabel("EG-QGEM Controls")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(title)

        # Parameters group
        params_group = QGroupBox("Simulation Parameters")
        params_layout = QFormLayout(params_group)

        self.n_subsystems = QSpinBox()
        self.n_subsystems.setRange(10, 100)
        self.n_subsystems.setValue(50)
        params_layout.addRow("Subsystems:", self.n_subsystems)

        self.steps = QSpinBox()
        self.steps.setRange(50, 500)
        self.steps.setValue(100)
        params_layout.addRow("Evolution Steps:", self.steps)

        layout.addWidget(params_group)

        # Buttons
        self.run_button = QPushButton("Run Simulation")
        self.run_button.clicked.connect(self.run_simulation)
        layout.addWidget(self.run_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_simulation)
        layout.addWidget(self.stop_button)

        # Progress
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # Log
        log_label = QLabel("Log:")
        layout.addWidget(log_label)

        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(200)
        layout.addWidget(self.log_text)

        layout.addStretch()

    def create_results_panel(self, parent):
        """Create the results panel."""
        results_widget = QWidget()
        parent.addWidget(results_widget)
        layout = QVBoxLayout(results_widget)

        # Title
        title = QLabel("Results & Visualization")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(title)

        # Plot canvas
        self.plot_canvas = SimplePlotCanvas(self, width=8, height=6)
        layout.addWidget(self.plot_canvas)

    def run_simulation(self):
        """Start simulation."""
        if self.worker and self.worker.isRunning():
            return

        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.log_text.clear()

        self.worker = SimpleWorker()
        self.worker.progress_updated.connect(self.progress_bar.setValue)
        self.worker.log_updated.connect(self.log_text.append)
        self.worker.finished_signal.connect(self.simulation_finished)
        self.worker.start()

        self.status_bar.showMessage("Simulation running...")

    def stop_simulation(self):
        """Stop simulation."""
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
        self.simulation_finished()

    def simulation_finished(self):
        """Handle simulation completion."""
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_bar.showMessage("Simulation completed")

        # Update plot with new data
        x = np.linspace(0, 10, 100)
        y = np.sin(x) * np.exp(-x/5)

        self.plot_canvas.axes.clear()
        self.plot_canvas.axes.plot(x, y, 'b-', linewidth=2)
        self.plot_canvas.axes.set_title('Simulation Results')
        self.plot_canvas.axes.set_xlabel('Time')
        self.plot_canvas.axes.set_ylabel('Amplitude')
        self.plot_canvas.figure.tight_layout()
        self.plot_canvas.draw()


def main():
    """Main function."""
    print("Starting EG-QGEM GUI Test...")

    app = QApplication(sys.argv)
    app.setApplicationName("EG-QGEM GUI Test")

    window = SimpleMainWindow()
    print("Window created successfully!")

    # In headless mode, just validate creation
    if os.environ.get('QT_QPA_PLATFORM') == 'offscreen':
        print("✓ GUI components validated successfully!")
        print("✓ Main window created")
        print("✓ Plot canvas initialized")
        print("✓ Worker thread available")
        print("\nTo run with display:")
        print("export DISPLAY=:0  # or appropriate display")
        print("python simple_gui_test.py")
        return

    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
