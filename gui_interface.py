"""
EG-QGEM Research GUI Interface
=============================

PyQt5-based graphical user interface for running EG-QGEM simulations
and visualizing results in real-time.
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
import seaborn as sns

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QPushButton, QSpinBox, QDoubleSpinBox,
    QComboBox, QTextEdit, QProgressBar, QSplitter, QGroupBox,
    QFormLayout, QCheckBox, QSlider, QFrame, QScrollArea,
    QMessageBox, QFileDialog, QStatusBar, QMenuBar, QAction
)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt
from PyQt5.QtGui import QFont, QPixmap, QPalette, QColor, QIcon

# Import EG-QGEM modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from research_interface import EGQGEMResearchInterface
from simulations.spacetime_emergence import SpacetimeEmergenceSimulator, run_emergence_simulation
from simulations.black_hole_simulator import BlackHoleSimulator
from experiments.predictions import generate_experimental_predictions
from visualization.plotting import SpacetimeVisualizer, BlackHoleVisualizer, ExperimentVisualizer

class SimulationWorker(QThread):
    """Worker thread for running simulations without blocking GUI."""

    progress_updated = pyqtSignal(int)
    log_updated = pyqtSignal(str)
    result_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, simulation_type, parameters):
        super().__init__()
        self.simulation_type = simulation_type
        self.parameters = parameters
        self.interface = EGQGEMResearchInterface()

    def run(self):
        """Execute the simulation based on type."""
        try:
            self.log_updated.emit(f"Starting {self.simulation_type} simulation...")

            if self.simulation_type == "spacetime":
                self.run_spacetime_simulation()
            elif self.simulation_type == "blackhole":
                self.run_blackhole_simulation()
            elif self.simulation_type == "experiments":
                self.run_experiments()
            elif self.simulation_type == "full":
                self.run_full_analysis()

        except Exception as e:
            self.error_occurred.emit(f"Error in {self.simulation_type}: {str(e)}")

    def run_spacetime_simulation(self):
        """Run spacetime emergence simulation."""
        config = {
            'n_subsystems': self.parameters.get('n_subsystems', 50),
            'evolution_steps': self.parameters.get('steps', 100),
            'entanglement_pattern': self.parameters.get('pattern', 'local'),
            'dimension': self.parameters.get('dimension', 3)
        }

        self.progress_updated.emit(10)
        self.log_updated.emit("Initializing spacetime emergence...")

        # Run simulation
        simulator, evolution_data = run_emergence_simulation(
            n_subsystems=config['n_subsystems'],
            steps=config['evolution_steps'],
            pattern=config['entanglement_pattern']
        )

        self.progress_updated.emit(80)
        self.log_updated.emit("Computing results...")

        # Prepare results
        results = {
            'type': 'spacetime',
            'simulator': simulator,
            'evolution_data': evolution_data,
            'summary': simulator.get_simulation_summary(),
            'config': config
        }

        self.progress_updated.emit(100)
        self.log_updated.emit("Spacetime emergence simulation completed!")
        self.result_ready.emit(results)

    def run_blackhole_simulation(self):
        """Run black hole simulation."""
        mass_solar = self.parameters.get('mass_solar_masses', 10)
        spin = self.parameters.get('spin', 0.0)

        self.progress_updated.emit(10)
        self.log_updated.emit(f"Creating {mass_solar} solar mass black hole...")

        # Initialize black hole
        mass_kg = mass_solar * 1.989e30
        bh = BlackHoleSimulator(mass=mass_kg, spin=spin)

        self.progress_updated.emit(30)
        self.log_updated.emit("Computing Hawking radiation...")

        # Run simulations
        radiation_data = bh.simulate_hawking_radiation(time_steps=50)

        self.progress_updated.emit(60)
        self.log_updated.emit("Computing information scrambling...")

        scrambling_data = bh.compute_information_scrambling()

        self.progress_updated.emit(90)
        self.log_updated.emit("Analyzing firewall resolution...")

        firewall_data = bh.analyze_firewall_resolution()

        results = {
            'type': 'blackhole',
            'simulator': bh,
            'radiation_data': radiation_data,
            'scrambling_data': scrambling_data,
            'firewall_data': firewall_data,
            'config': {'mass_solar_masses': mass_solar, 'spin': spin}
        }

        self.progress_updated.emit(100)
        self.log_updated.emit("Black hole simulation completed!")
        self.result_ready.emit(results)

    def run_experiments(self):
        """Run experimental predictions."""
        self.progress_updated.emit(20)
        self.log_updated.emit("Generating experimental predictions...")

        config = {
            'mass_range': [1e-18, 1e-12],
            'distance_range': [1e-3, 1e3],
            'energy_range': [1e-20, 1e-10],
            'time_range': [1e-15, 1e-3]
        }

        self.progress_updated.emit(50)
        predictions = generate_experimental_predictions(config)

        results = {
            'type': 'experiments',
            'predictions': predictions,
            'config': config
        }

        self.progress_updated.emit(100)
        self.log_updated.emit("Experimental predictions generated!")
        self.result_ready.emit(results)

    def run_full_analysis(self):
        """Run complete analysis."""
        self.log_updated.emit("Running comprehensive EG-QGEM analysis...")

        # Run spacetime
        self.progress_updated.emit(10)
        spacetime_params = {'n_subsystems': 30, 'steps': 50}
        self.parameters = spacetime_params
        self.run_spacetime_simulation()

        # Run black hole
        self.progress_updated.emit(40)
        blackhole_params = {'mass_solar_masses': 5, 'spin': 0.5}
        self.parameters = blackhole_params
        self.run_blackhole_simulation()

        # Run experiments
        self.progress_updated.emit(70)
        self.run_experiments()

        self.progress_updated.emit(100)
        self.log_updated.emit("Full analysis completed!")


class PlotCanvas(FigureCanvas):
    """Matplotlib canvas for embedding plots in PyQt5."""

    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        plt.style.use('seaborn-v0_8')

    def clear_plot(self):
        """Clear the current plot."""
        self.fig.clear()
        self.draw()

    def plot_spacetime_results(self, results):
        """Plot spacetime emergence results."""
        self.fig.clear()
        simulator = results['simulator']

        # Create subplots
        gs = self.fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # 1. Entanglement network
        ax1 = self.fig.add_subplot(gs[0, 0])
        positions = simulator.positions[:, :2]  # 2D projection
        ax1.scatter(positions[:, 0], positions[:, 1],
                   c=simulator.curvature_field, cmap='viridis', s=50)
        ax1.set_title('Entanglement Network')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')

        # 2. Entanglement evolution
        ax2 = self.fig.add_subplot(gs[0, 1])
        evolution_data = results['evolution_data']
        if evolution_data:
            steps = list(range(len(evolution_data)))
            entanglement = [data['total_entanglement'] for data in evolution_data]
            ax2.plot(steps, entanglement, 'b-', linewidth=2)
            ax2.set_title('Entanglement Evolution')
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Total Entanglement')
            ax2.grid(True, alpha=0.3)

        # 3. Curvature distribution
        ax3 = self.fig.add_subplot(gs[1, 0])
        ax3.hist(simulator.curvature_field, bins=20, alpha=0.7, color='orange')
        ax3.set_title('Curvature Distribution')
        ax3.set_xlabel('Curvature')
        ax3.set_ylabel('Frequency')

        # 4. Summary statistics
        ax4 = self.fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        summary = results['summary']
        stats_text = f"""Simulation Summary:

• Subsystems: {summary['n_subsystems']}
• Total Entanglement: {summary['total_entanglement']:.3f}
• Connectivity: {summary['connectivity']:.3f}
• Avg Curvature: {summary['avg_curvature']:.4f}
• Network Edges: {summary.get('network_edges', 'N/A')}"""

        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')

        self.fig.suptitle('Spacetime Emergence Simulation Results', fontsize=14, fontweight='bold')
        self.draw()

    def plot_blackhole_results(self, results):
        """Plot black hole simulation results."""
        self.fig.clear()
        simulator = results['simulator']
        radiation_data = results['radiation_data']
        scrambling_data = results['scrambling_data']

        # Create subplots
        gs = self.fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # 1. Mass evolution
        ax1 = self.fig.add_subplot(gs[0, 0])
        times = radiation_data['times'] / (365.25 * 24 * 3600)  # Convert to years
        mass_ratio = radiation_data['mass_evolution'] / simulator.mass
        ax1.plot(times, mass_ratio, 'r-', linewidth=2)
        ax1.set_title('Black Hole Mass Evolution')
        ax1.set_xlabel('Time (years)')
        ax1.set_ylabel('M(t)/M₀')
        ax1.grid(True, alpha=0.3)

        # 2. Hawking temperature
        ax2 = self.fig.add_subplot(gs[0, 1])
        ax2.semilogy(times, radiation_data['temperature_evolution'], 'g-', linewidth=2)
        ax2.set_title('Hawking Temperature')
        ax2.set_xlabel('Time (years)')
        ax2.set_ylabel('Temperature (K)')
        ax2.grid(True, alpha=0.3)

        # 3. Information scrambling
        ax3 = self.fig.add_subplot(gs[1, 0])
        ax3.plot(scrambling_data['times'], scrambling_data['otoc'], 'b-', linewidth=2)
        ax3.axvline(scrambling_data['scrambling_time'], color='red', linestyle='--',
                   label=f"Scrambling time: {scrambling_data['scrambling_time']:.2f}")
        ax3.set_title('Information Scrambling (OTOC)')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Out-of-Time Correlator')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Summary statistics
        ax4 = self.fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        config = results['config']
        stats_text = f"""Black Hole Properties:

• Mass: {config['mass_solar_masses']} M☉
• Spin: {config['spin']}
• Schwarzschild radius: {simulator.rs/1000:.2f} km
• Hawking temp: {simulator.compute_hawking_temperature():.2e} K
• Scrambling time: {scrambling_data['scrambling_time']:.2f}
• Lyapunov exp: {scrambling_data['lyapunov_exponent']:.2f}"""

        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')

        self.fig.suptitle('Black Hole Simulation Results', fontsize=14, fontweight='bold')
        self.draw()

    def plot_experiment_results(self, results):
        """Plot experimental predictions."""
        self.fig.clear()
        predictions = results['predictions']

        # Create subplots
        gs = self.fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # 1. Quantum gravity experiment
        ax1 = self.fig.add_subplot(gs[0, 0])
        qg_data = predictions['quantum_gravity']
        ax1.plot(qg_data['times'] * 1000, qg_data['visibility_evolution'], 'b-', linewidth=2)
        ax1.set_title('Quantum Gravity Experiment')
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Interference Visibility')
        ax1.grid(True, alpha=0.3)

        # 2. Gravitational waves
        ax2 = self.fig.add_subplot(gs[0, 1])
        gw_data = predictions['gravitational_waves']
        ax2.semilogy(gw_data['times'], gw_data['amplitudes'], 'r-', linewidth=2, label='Standard GW')
        ax2.semilogy(gw_data['times'], gw_data['entanglement_signals'], 'g--', linewidth=2, label='Entanglement signature')
        ax2.set_title('Gravitational Wave Signatures')
        ax2.set_xlabel('Time to merger (s)')
        ax2.set_ylabel('Strain amplitude')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. CMB power spectrum
        ax3 = self.fig.add_subplot(gs[1, 0])
        cmb_data = predictions['cosmology']['cmb_power_spectrum']
        ax3.loglog(cmb_data['l_values'], cmb_data['C_l_standard'], 'k-', linewidth=2, label='Standard ΛCDM')
        ax3.loglog(cmb_data['l_values'], cmb_data['C_l_modified'], 'r-', linewidth=2, label='EG-QGEM modified')
        ax3.set_title('CMB Power Spectrum')
        ax3.set_xlabel('Multipole l')
        ax3.set_ylabel('C_l')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Detectability summary
        ax4 = self.fig.add_subplot(gs[1, 1])
        detectability = {
            'Quantum\nGravity': qg_data['entanglement_rate'] > 1e-6,
            'GW\nEntanglement': gw_data['detectability'],
            'CMB\nFeatures': cmb_data['detectability'],
            'Decoherence': predictions['decoherence']['measurable']
        }

        names = list(detectability.keys())
        values = [1 if v else 0 for v in detectability.values()]
        colors = ['green' if v else 'red' for v in values]

        bars = ax4.bar(names, values, color=colors, alpha=0.7)
        ax4.set_title('Experimental Detectability')
        ax4.set_ylabel('Detectable')
        ax4.set_ylim(0, 1.2)

        # Add text labels
        for bar, detectable in zip(bars, detectability.values()):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    'YES' if detectable else 'NO',
                    ha='center', va='bottom', fontweight='bold')

        self.fig.suptitle('Experimental Predictions', fontsize=14, fontweight='bold')
        self.draw()


class EGQGEMMainWindow(QMainWindow):
    """Main application window for EG-QGEM research GUI."""

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.current_results = {}
        self.worker = None

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("EG-QGEM Research Interface")
        self.setGeometry(100, 100, 1400, 900)

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Left panel: Controls
        self.create_control_panel(splitter)

        # Right panel: Results and visualization
        self.create_results_panel(splitter)

        # Set splitter proportions
        splitter.setSizes([400, 1000])

        # Create status bar
        self.statusBar().showMessage("Ready to run EG-QGEM simulations")

        # Create menu bar
        self.create_menu_bar()

        # Apply dark theme
        self.apply_dark_theme()

    def create_control_panel(self, parent):
        """Create the control panel for simulation parameters."""
        control_widget = QWidget()
        parent.addWidget(control_widget)
        layout = QVBoxLayout(control_widget)

        # Title
        title = QLabel("EG-QGEM Control Panel")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Simulation type selection
        sim_group = QGroupBox("Simulation Type")
        sim_layout = QVBoxLayout(sim_group)

        self.sim_type = QComboBox()
        self.sim_type.addItems([
            "Spacetime Emergence",
            "Black Hole Physics",
            "Experimental Predictions",
            "Full Analysis"
        ])
        self.sim_type.currentTextChanged.connect(self.on_simulation_type_changed)
        sim_layout.addWidget(self.sim_type)
        layout.addWidget(sim_group)

        # Parameter panels (will be shown/hidden based on simulation type)
        self.create_spacetime_params(layout)
        self.create_blackhole_params(layout)
        self.create_experiment_params(layout)

        # Run button
        self.run_button = QPushButton("🚀 Run Simulation")
        self.run_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.run_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.run_button.clicked.connect(self.run_simulation)
        layout.addWidget(self.run_button)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Log area
        log_group = QGroupBox("Simulation Log")
        log_layout = QVBoxLayout(log_group)

        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(200)
        self.log_text.setFont(QFont("Consolas", 9))
        log_layout.addWidget(self.log_text)

        # Clear log button
        clear_log_btn = QPushButton("Clear Log")
        clear_log_btn.clicked.connect(self.log_text.clear)
        log_layout.addWidget(clear_log_btn)

        layout.addWidget(log_group)

        # Stretch to push everything to top
        layout.addStretch()

        # Show initial parameter panel
        self.on_simulation_type_changed("Spacetime Emergence")

    def create_spacetime_params(self, parent_layout):
        """Create spacetime simulation parameter controls."""
        self.spacetime_group = QGroupBox("Spacetime Parameters")
        layout = QFormLayout(self.spacetime_group)

        self.n_subsystems = QSpinBox()
        self.n_subsystems.setRange(10, 200)
        self.n_subsystems.setValue(50)
        layout.addRow("Number of Subsystems:", self.n_subsystems)

        self.evolution_steps = QSpinBox()
        self.evolution_steps.setRange(10, 500)
        self.evolution_steps.setValue(100)
        layout.addRow("Evolution Steps:", self.evolution_steps)

        self.entanglement_pattern = QComboBox()
        self.entanglement_pattern.addItems(["local", "random", "small_world"])
        layout.addRow("Entanglement Pattern:", self.entanglement_pattern)

        self.dimension = QSpinBox()
        self.dimension.setRange(2, 4)
        self.dimension.setValue(3)
        layout.addRow("Spatial Dimension:", self.dimension)

        parent_layout.addWidget(self.spacetime_group)

    def create_blackhole_params(self, parent_layout):
        """Create black hole simulation parameter controls."""
        self.blackhole_group = QGroupBox("Black Hole Parameters")
        layout = QFormLayout(self.blackhole_group)

        self.mass_solar = QDoubleSpinBox()
        self.mass_solar.setRange(0.1, 100.0)
        self.mass_solar.setValue(10.0)
        self.mass_solar.setSuffix(" M☉")
        layout.addRow("Mass:", self.mass_solar)

        self.spin = QDoubleSpinBox()
        self.spin.setRange(0.0, 0.99)
        self.spin.setValue(0.0)
        self.spin.setDecimals(2)
        layout.addRow("Spin (a/M):", self.spin)

        self.charge = QDoubleSpinBox()
        self.charge.setRange(0.0, 0.99)
        self.charge.setValue(0.0)
        self.charge.setDecimals(2)
        layout.addRow("Charge (Q/M):", self.charge)

        parent_layout.addWidget(self.blackhole_group)

    def create_experiment_params(self, parent_layout):
        """Create experimental prediction parameter controls."""
        self.experiment_group = QGroupBox("Experiment Parameters")
        layout = QFormLayout(self.experiment_group)

        self.mass_range_min = QDoubleSpinBox()
        self.mass_range_min.setRange(1e-20, 1e-10)
        self.mass_range_min.setValue(1e-18)
        self.mass_range_min.setDecimals(20)
        # Scientific notation display not available in this PyQt5 version
        layout.addRow("Min Mass (kg):", self.mass_range_min)

        self.mass_range_max = QDoubleSpinBox()
        self.mass_range_max.setRange(1e-15, 1e-5)
        self.mass_range_max.setValue(1e-12)
        self.mass_range_max.setDecimals(20)
        # Scientific notation display not available in this PyQt5 version
        layout.addRow("Max Mass (kg):", self.mass_range_max)

        parent_layout.addWidget(self.experiment_group)

    def create_results_panel(self, parent):
        """Create the results and visualization panel."""
        results_widget = QWidget()
        parent.addWidget(results_widget)
        layout = QVBoxLayout(results_widget)

        # Results tab widget
        self.results_tabs = QTabWidget()
        layout.addWidget(self.results_tabs)

        # Visualization tab
        self.viz_tab = QWidget()
        self.results_tabs.addTab(self.viz_tab, "📊 Visualization")
        viz_layout = QVBoxLayout(self.viz_tab)

        # Plot canvas
        self.plot_canvas = PlotCanvas(self.viz_tab, width=10, height=8)
        viz_layout.addWidget(self.plot_canvas)

        # Plot controls
        plot_controls = QHBoxLayout()

        self.save_plot_btn = QPushButton("💾 Save Plot")
        self.save_plot_btn.clicked.connect(self.save_plot)
        plot_controls.addWidget(self.save_plot_btn)

        self.clear_plot_btn = QPushButton("🗑️ Clear Plot")
        self.clear_plot_btn.clicked.connect(self.plot_canvas.clear_plot)
        plot_controls.addWidget(self.clear_plot_btn)

        plot_controls.addStretch()
        viz_layout.addLayout(plot_controls)

        # Data tab
        self.data_tab = QWidget()
        self.results_tabs.addTab(self.data_tab, "📋 Data")
        data_layout = QVBoxLayout(self.data_tab)

        self.data_text = QTextEdit()
        self.data_text.setFont(QFont("Consolas", 10))
        data_layout.addWidget(self.data_text)

        # Export controls
        export_controls = QHBoxLayout()

        self.export_json_btn = QPushButton("📄 Export JSON")
        self.export_json_btn.clicked.connect(self.export_json)
        export_controls.addWidget(self.export_json_btn)

        self.export_report_btn = QPushButton("📊 Generate Report")
        self.export_report_btn.clicked.connect(self.generate_report)
        export_controls.addWidget(self.export_report_btn)

        export_controls.addStretch()
        data_layout.addLayout(export_controls)

    def create_menu_bar(self):
        """Create the application menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu('File')

        load_action = QAction('Load Configuration', self)
        load_action.triggered.connect(self.load_configuration)
        file_menu.addAction(load_action)

        save_action = QAction('Save Configuration', self)
        save_action.triggered.connect(self.save_configuration)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        exit_action = QAction('Exit', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menubar.addMenu('View')

        toggle_log_action = QAction('Toggle Log Panel', self)
        toggle_log_action.triggered.connect(self.toggle_log_panel)
        view_menu.addAction(toggle_log_action)

        # Help menu
        help_menu = menubar.addMenu('Help')

        about_action = QAction('About EG-QGEM', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def apply_dark_theme(self):
        """Apply a dark theme to the application."""
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ToolTipBase, QColor(0, 0, 0))
        dark_palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.Text, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
        dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))

        self.setPalette(dark_palette)

    def on_simulation_type_changed(self, sim_type):
        """Handle simulation type change."""
        # Hide all parameter groups
        self.spacetime_group.setVisible(False)
        self.blackhole_group.setVisible(False)
        self.experiment_group.setVisible(False)

        # Show relevant parameter group
        if sim_type == "Spacetime Emergence":
            self.spacetime_group.setVisible(True)
        elif sim_type == "Black Hole Physics":
            self.blackhole_group.setVisible(True)
        elif sim_type == "Experimental Predictions":
            self.experiment_group.setVisible(True)
        # Full Analysis shows no specific parameters

    def run_simulation(self):
        """Start the selected simulation."""
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Warning", "A simulation is already running!")
            return

        # Disable run button and show progress
        self.run_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # Clear previous results
        self.plot_canvas.clear_plot()
        self.data_text.clear()

        # Get simulation parameters
        sim_type_map = {
            "Spacetime Emergence": "spacetime",
            "Black Hole Physics": "blackhole",
            "Experimental Predictions": "experiments",
            "Full Analysis": "full"
        }

        sim_type = sim_type_map[self.sim_type.currentText()]
        parameters = self.get_current_parameters()

        # Create and start worker thread
        self.worker = SimulationWorker(sim_type, parameters)
        self.worker.progress_updated.connect(self.progress_bar.setValue)
        self.worker.log_updated.connect(self.add_log_message)
        self.worker.result_ready.connect(self.on_simulation_completed)
        self.worker.error_occurred.connect(self.on_simulation_error)
        self.worker.finished.connect(self.on_worker_finished)

        self.worker.start()

        self.add_log_message(f"Started {self.sim_type.currentText()} simulation...")

    def get_current_parameters(self):
        """Get current simulation parameters from UI."""
        sim_type = self.sim_type.currentText()

        if sim_type == "Spacetime Emergence":
            return {
                'n_subsystems': self.n_subsystems.value(),
                'steps': self.evolution_steps.value(),
                'pattern': self.entanglement_pattern.currentText(),
                'dimension': self.dimension.value()
            }
        elif sim_type == "Black Hole Physics":
            return {
                'mass_solar_masses': self.mass_solar.value(),
                'spin': self.spin.value(),
                'charge': self.charge.value()
            }
        elif sim_type == "Experimental Predictions":
            return {
                'mass_range_min': self.mass_range_min.value(),
                'mass_range_max': self.mass_range_max.value()
            }
        else:  # Full Analysis
            return {}

    def add_log_message(self, message):
        """Add a message to the log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

        # Auto-scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def on_simulation_completed(self, results):
        """Handle simulation completion."""
        self.current_results = results
        self.add_log_message("Simulation completed successfully!")

        # Update visualization
        if results['type'] == 'spacetime':
            self.plot_canvas.plot_spacetime_results(results)
        elif results['type'] == 'blackhole':
            self.plot_canvas.plot_blackhole_results(results)
        elif results['type'] == 'experiments':
            self.plot_canvas.plot_experiment_results(results)

        # Update data display
        self.update_data_display(results)

        self.statusBar().showMessage("Simulation completed successfully")

    def on_simulation_error(self, error_message):
        """Handle simulation error."""
        self.add_log_message(f"ERROR: {error_message}")
        QMessageBox.critical(self, "Simulation Error", f"An error occurred:\n\n{error_message}")
        self.statusBar().showMessage("Simulation failed")

    def on_worker_finished(self):
        """Handle worker thread completion."""
        self.run_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)

    def update_data_display(self, results):
        """Update the data display tab with results."""
        data_text = "SIMULATION RESULTS\n"
        data_text += "=" * 50 + "\n\n"

        if results['type'] == 'spacetime':
            summary = results['summary']
            data_text += "SPACETIME EMERGENCE SIMULATION:\n"
            data_text += f"• Subsystems: {summary['n_subsystems']}\n"
            data_text += f"• Dimension: {summary['dimension']}\n"
            data_text += f"• Total Entanglement: {summary['total_entanglement']:.4f}\n"
            data_text += f"• Average Entanglement: {summary['avg_entanglement']:.4f}\n"
            data_text += f"• Network Connectivity: {summary['connectivity']:.4f}\n"
            data_text += f"• Average Curvature: {summary['avg_curvature']:.6f}\n"

        elif results['type'] == 'blackhole':
            config = results['config']
            simulator = results['simulator']
            data_text += "BLACK HOLE SIMULATION:\n"
            data_text += f"• Mass: {config['mass_solar_masses']} solar masses\n"
            data_text += f"• Spin: {config['spin']}\n"
            data_text += f"• Schwarzschild radius: {simulator.rs/1000:.2f} km\n"
            data_text += f"• Hawking temperature: {simulator.compute_hawking_temperature():.2e} K\n"

            scrambling_data = results['scrambling_data']
            data_text += f"• Scrambling time: {scrambling_data['scrambling_time']:.2f}\n"
            data_text += f"• Lyapunov exponent: {scrambling_data['lyapunov_exponent']:.2f}\n"

        elif results['type'] == 'experiments':
            predictions = results['predictions']
            data_text += "EXPERIMENTAL PREDICTIONS:\n"

            qg_data = predictions['quantum_gravity']
            data_text += f"• Quantum gravity entanglement rate: {qg_data['entanglement_rate']:.2e} Hz\n"

            gw_data = predictions['gravitational_waves']
            data_text += f"• GW entanglement detectable: {gw_data['detectability']}\n"

            cmb_data = predictions['cosmology']['cmb_power_spectrum']
            data_text += f"• CMB features detectable: {cmb_data['detectability']}\n"

            decoherence_data = predictions['decoherence']
            data_text += f"• Gravitational decoherence measurable: {decoherence_data['measurable']}\n"
            data_text += f"• Gravitational coherence time: {decoherence_data['gravitational_time']:.2e} s\n"

        self.data_text.setText(data_text)

    def save_plot(self):
        """Save the current plot to file."""
        if not hasattr(self.plot_canvas, 'fig') or not self.plot_canvas.fig.get_axes():
            QMessageBox.warning(self, "Warning", "No plot to save!")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Plot", f"eg_qgem_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            "PNG files (*.png);;PDF files (*.pdf);;SVG files (*.svg)"
        )

        if filename:
            self.plot_canvas.fig.savefig(filename, dpi=300, bbox_inches='tight')
            self.add_log_message(f"Plot saved to {filename}")

    def export_json(self):
        """Export current results to JSON file."""
        if not self.current_results:
            QMessageBox.warning(self, "Warning", "No results to export!")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Results", f"eg_qgem_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON files (*.json)"
        )

        if filename:
            # Create exportable data (remove non-serializable objects)
            export_data = {
                'type': self.current_results['type'],
                'config': self.current_results['config'],
                'timestamp': datetime.now().isoformat()
            }

            if self.current_results['type'] == 'spacetime':
                export_data['summary'] = self.current_results['summary']
            elif self.current_results['type'] == 'blackhole':
                export_data['scrambling_data'] = self.current_results['scrambling_data']
            elif self.current_results['type'] == 'experiments':
                export_data['predictions'] = self.current_results['predictions']

            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)

            self.add_log_message(f"Results exported to {filename}")

    def generate_report(self):
        """Generate a comprehensive research report."""
        if not self.current_results:
            QMessageBox.warning(self, "Warning", "No results to generate report from!")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Generate Report", f"eg_qgem_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "Text files (*.txt)"
        )

        if filename:
            with open(filename, 'w') as f:
                f.write("ENTANGLED GEOMETRODYNAMICS & QUANTUM-GRAVITATIONAL ENTANGLEMENT METRIC\n")
                f.write("RESEARCH REPORT\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                # Write simulation-specific content
                results = self.current_results

                if results['type'] == 'spacetime':
                    f.write("SPACETIME EMERGENCE SIMULATION RESULTS:\n")
                    f.write("-" * 50 + "\n")
                    summary = results['summary']
                    for key, value in summary.items():
                        f.write(f"• {key.replace('_', ' ').title()}: {value}\n")

                elif results['type'] == 'blackhole':
                    f.write("BLACK HOLE SIMULATION RESULTS:\n")
                    f.write("-" * 50 + "\n")
                    config = results['config']
                    simulator = results['simulator']
                    f.write(f"• Mass: {config['mass_solar_masses']} solar masses\n")
                    f.write(f"• Schwarzschild radius: {simulator.rs/1000:.2f} km\n")
                    f.write(f"• Hawking temperature: {simulator.compute_hawking_temperature():.2e} K\n")

                elif results['type'] == 'experiments':
                    f.write("EXPERIMENTAL PREDICTIONS:\n")
                    f.write("-" * 50 + "\n")
                    predictions = results['predictions']

                    qg_data = predictions['quantum_gravity']
                    f.write(f"• Quantum gravity entanglement rate: {qg_data['entanglement_rate']:.2e} Hz\n")

                    gw_data = predictions['gravitational_waves']
                    f.write(f"• Gravitational wave entanglement detectable: {gw_data['detectability']}\n")

            self.add_log_message(f"Report generated: {filename}")

    def load_configuration(self):
        """Load simulation configuration from file."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Configuration", "", "JSON files (*.json)"
        )

        if filename:
            try:
                with open(filename, 'r') as f:
                    config = json.load(f)

                # Apply configuration to UI
                if 'simulation_type' in config:
                    index = self.sim_type.findText(config['simulation_type'])
                    if index >= 0:
                        self.sim_type.setCurrentIndex(index)

                if 'parameters' in config:
                    params = config['parameters']

                    # Set spacetime parameters
                    if 'n_subsystems' in params:
                        self.n_subsystems.setValue(params['n_subsystems'])
                    if 'steps' in params:
                        self.evolution_steps.setValue(params['steps'])

                    # Set black hole parameters
                    if 'mass_solar_masses' in params:
                        self.mass_solar.setValue(params['mass_solar_masses'])
                    if 'spin' in params:
                        self.spin.setValue(params['spin'])

                self.add_log_message(f"Configuration loaded from {filename}")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load configuration:\n{str(e)}")

    def save_configuration(self):
        """Save current simulation configuration to file."""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Configuration", f"eg_qgem_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON files (*.json)"
        )

        if filename:
            config = {
                'simulation_type': self.sim_type.currentText(),
                'parameters': self.get_current_parameters(),
                'timestamp': datetime.now().isoformat()
            }

            with open(filename, 'w') as f:
                json.dump(config, f, indent=2)

            self.add_log_message(f"Configuration saved to {filename}")

    def toggle_log_panel(self):
        """Toggle visibility of the log panel."""
        log_group = self.findChild(QGroupBox, "log_group")
        if log_group:
            log_group.setVisible(not log_group.isVisible())

    def show_about(self):
        """Show about dialog."""
        about_text = """
        <h2>EG-QGEM Research Interface</h2>
        <p><b>Entangled Geometrodynamics & Quantum-Gravitational Entanglement Metric</b></p>
        <p>A comprehensive research framework for exploring the quantum origins of spacetime.</p>

        <h3>Features:</h3>
        <ul>
        <li>Spacetime emergence simulations</li>
        <li>Black hole physics with entanglement</li>
        <li>Experimental predictions</li>
        <li>Real-time visualization</li>
        <li>Data export and reporting</li>
        </ul>

        <p><i>Developed for advanced theoretical physics research.</i></p>
        """

        QMessageBox.about(self, "About EG-QGEM", about_text)


def main():
    """Main entry point for the GUI application."""
    app = QApplication(sys.argv)
    app.setApplicationName("EG-QGEM Research Interface")
    app.setApplicationVersion("1.0")

    # Set application icon (if available)
    try:
        app.setWindowIcon(QIcon("icon.png"))
    except:
        pass

    window = EGQGEMMainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
