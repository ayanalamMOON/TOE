# pyright: reportMissingImports=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportOptionalMemberAccess=false

"""
Advanced EG-QGEM Research GUI Interface
======================================

Enhanced PyQt5-based graphical user interface for EG-QGEM simulations
with advanced features including 3D visualization, real-time monitoring,
batch processing, and comprehensive data analysis tools.
"""

from PyQt5.QtWidgets import (QMainWindow, QApplication, QVBoxLayout, QHBoxLayout, QWidget,
                             QPushButton, QLabel, QProgressBar, QTextEdit, QMenuBar, QToolBar,
                             QStatusBar, QDockWidget, QSizePolicy, QFrame, QSplitter, QComboBox,
                             QSpinBox, QDoubleSpinBox, QFormLayout, QFileDialog, QMessageBox, QSlider,
                             QAction, QActionGroup, QGroupBox, QTabWidget, QCheckBox, QListWidget,
                             QListWidgetItem, QLineEdit)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt, QUrl
from PyQt5.QtGui import QPalette, QColor, QFont, QIcon, QDesktopServices

try:
    from PyQt5.QtWebEngineWidgets import QWebEngineView
    WEB_ENGINE_AVAILABLE = True
except ImportError:
    WEB_ENGINE_AVAILABLE = False
    QWebEngineView = None  # Already set

import sys
import psutil
import time
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import plotly.io as pio
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D # Moved import to top
import json
import traceback
from datetime import datetime
from pathlib import Path
import seaborn as sns

from research_interface import EGQGEMResearchInterface
from simulations.spacetime_emergence import SpacetimeEmergenceSimulator, run_emergence_simulation
from simulations.black_hole_simulator import BlackHoleSimulator
from experiments.predictions import generate_experimental_predictions
from visualization.plotting import SpacetimeVisualizer, BlackHoleVisualizer, ExperimentVisualizer


class AdvancedSimulationWorker(QThread):
    """Enhanced worker thread for running simulations with advanced monitoring."""

    progress_updated = pyqtSignal(int)
    stage_updated = pyqtSignal(str)
    log_updated = pyqtSignal(str)
    result_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    performance_updated = pyqtSignal(dict)
    intermediate_result = pyqtSignal(dict)

    def __init__(self, worker_type, parameters, parent=None):
        super().__init__(parent)
        self.worker_type = worker_type
        self.parameters = parameters if parameters is not None else {}
        self.is_paused = False
        self.is_stopped = False
        self.current_simulation_instance = None # For potential direct control

    def run(self):
        try:
            self.stage_updated.emit(f"Starting {self.worker_type} simulation...")
            if self.worker_type == 'spacetime':
                self._run_spacetime_simulation()
            elif self.worker_type == 'blackhole':
                self._run_blackhole_simulation()
            elif self.worker_type == 'experiments':
                self._run_experiment_simulation()
            elif self.worker_type == 'full':
                self._run_full_analysis()
            elif self.worker_type == 'batch':
                self._run_batch_simulation()
            elif self.worker_type == 'sweep':
                self._run_sweep_simulation()
            else:
                self.log_updated.emit(f"Error: Unknown worker type \'{self.worker_type}\'")
                self.error_occurred.emit(f"Unknown worker type: {self.worker_type}")
        except Exception as e:
            self.log_updated.emit(f"Critical error in simulation worker: {e}")
            self.error_occurred.emit(str(e))
            import traceback
            self.log_updated.emit(traceback.format_exc())
        finally:
            self.stage_updated.emit("Simulation worker finished.")

    def _run_spacetime_simulation(self):
        self.stage_updated.emit("Initializing Spacetime Simulation")
        n_subsystems = self.parameters.get('n_subsystems', 50)
        steps = self.parameters.get('steps', 100)
        dimension = self.parameters.get('dimension', 3)
        self.log_updated.emit(f"Spacetime params: Subsystems={n_subsystems}, Steps={steps}, Dimension={dimension}")

        # Placeholder for actual simulation
        # from simulations.spacetime_simulator import SpacetimeEmergenceSimulator
        # self.current_simulation_instance = SpacetimeEmergenceSimulator(n_subsystems=n_subsystems, steps=steps, dimension=dimension)
        # results_data = self.current_simulation_instance.run_simulation()

        total_sim_steps = steps
        for i in range(total_sim_steps):
            if self.is_stopped:
                self.log_updated.emit("Spacetime simulation stopped by user.")
                return
            while self.is_paused: self.msleep(100)

            self.msleep(20) # Simulate work
            self.progress_updated.emit(int(((i + 1) / total_sim_steps) * 100))
            if i % (total_sim_steps // 10 + 1) == 0:
                self.intermediate_result.emit({
                    'type': 'spacetime_step', 'step': i + 1, 'total_steps': total_sim_steps,
                    'current_complexity': np.random.rand() # Example intermediate data
                })

        self.stage_updated.emit("Finalizing Spacetime Simulation")
        # Dummy results
        results_data = {
            'type': 'spacetime',
            'parameters': self.parameters,
            'complexity_measures': np.random.rand(10).tolist(),
            'network_data': {'nodes': [], 'edges': []}, # Populate with some dummy nodes/edges
            'frames': [np.random.rand(n_subsystems, dimension) for _ in range(steps)] # For animation
        }
        # Example nodes and edges for 3D plot
        nodes = [{'id': i, 'size': np.random.uniform(0.5,2), 'color': np.random.rand(3).tolist()} for i in range(n_subsystems)]
        edges = [{'source': np.random.randint(0,n_subsystems), 'target': np.random.randint(0,n_subsystems)} for _ in range(n_subsystems*2)]
        results_data['network_data'] = {'nodes': nodes, 'edges': edges, 'positions': np.random.rand(n_subsystems, dimension).tolist()}


        self.result_ready.emit(results_data)
        self.log_updated.emit("Spacetime simulation completed.")

    def _run_blackhole_simulation(self):
        self.stage_updated.emit("Initializing Black Hole Simulation")
        mass = self.parameters.get('mass_solar_masses', 10.0)
        self.log_updated.emit(f"Simulating black hole with mass: {mass} M☉")

        # Placeholder for actual simulation
        # from simulations.black_hole_simulator import BlackHoleSimulator
        # self.current_simulation_instance = BlackHoleSimulator(mass_solar_masses=mass)
        # results_data = self.current_simulation_instance.calculate_properties()

        total_steps = 100 # Example steps
        for i in range(total_steps):
            if self.is_stopped:
                self.log_updated.emit("Black Hole simulation stopped by user.")
                return
            while self.is_paused: self.msleep(100)

            self.msleep(30) # Simulate work
            self.progress_updated.emit(int(((i + 1) / total_steps) * 100))
            if i % (total_steps // 10 + 1) == 0:
                rs_km = (2 * 6.674e-11 * (mass * 1.989e30) / (3e8)**2) / 1000
                self.intermediate_result.emit({
                    'type': 'blackhole_step', 'step': i + 1,
                    'current_property_calc': f"Calculating step {i+1}",
                    'schwarzschild_radius_km_so_far': rs_km * (i+1)/total_steps
                })

        self.stage_updated.emit("Finalizing Black Hole Simulation")
        rs_km = (2 * 6.674e-11 * (mass * 1.989e30) / (3e8)**2) / 1000
        temp_k = 6.17e-8 / mass
        results_data = {
            'type': 'blackhole',
            'parameters': self.parameters,
            'schwarzschild_radius_km': rs_km,
            'hawking_temperature_K': temp_k,
            'event_horizon_data': (np.random.rand(50, 2) * rs_km).tolist(), # Example data scaled by Rs
            'dummy_merger_waveform': np.sin(np.linspace(0, 10*np.pi, 200)).tolist()
        }
        self.result_ready.emit(results_data)
        self.log_updated.emit("Black Hole simulation completed.")

    def _run_experiment_simulation(self):
        self.stage_updated.emit("Initializing Experimental Prediction Simulation")
        experiment_type = self.parameters.get('experiment_type', 'LIGO') # Example parameter
        self.log_updated.emit(f"Simulating experiment: {experiment_type}")

        # Placeholder for actual simulation
        # from experiments.predictions import ExperimentalPredictor
        # self.current_simulation_instance = ExperimentalPredictor(params=self.parameters)
        # results_data = self.current_simulation_instance.run_prediction()

        total_steps = 70
        for i in range(total_steps):
            if self.is_stopped:
                self.log_updated.emit("Experiment simulation stopped by user.")
                return
            while self.is_paused: self.msleep(100)

            self.msleep(25) # Simulate work
            self.progress_updated.emit(int(((i + 1) / total_steps) * 100))
            if i % (total_steps // 5 + 1) == 0:
                 self.intermediate_result.emit({'type': 'experiment_step', 'step': i+1, 'status': 'processing data subset'})

        self.stage_updated.emit("Finalizing Experimental Predictions")
        results_data = {
            'type': 'experiments',
            'parameters': self.parameters,
            'predicted_event_rate': np.random.uniform(0.1, 10),
            'confidence_level': np.random.uniform(0.5, 0.99),
            'data_points': np.random.rand(100, 2).tolist() # Example X, Y data
        }
        self.result_ready.emit(results_data)
        self.log_updated.emit("Experimental prediction simulation completed.")

    def _run_full_analysis(self):
        self.stage_updated.emit("Initializing Full Comprehensive Analysis")
        # This would combine parameters and run multiple sub-simulations
        st_params = self.parameters.get('spacetime_params', {'n_subsystems': 30, 'steps': 50, 'dimension': 3})
        bh_params = self.parameters.get('blackhole_params', {'mass_solar_masses': 5.0})
        self.log_updated.emit(f"Full analysis with ST: {st_params}, BH: {bh_params}")

        # Placeholder: simulate running parts of other simulations
        total_stages = 3 # e.g., ST part, BH part, Cross-analysis part
        for stage_num in range(total_stages):
            self.stage_updated.emit(f"Full Analysis - Stage {stage_num+1}/{total_stages}")
            for i in range(50): # 50 steps per stage
                if self.is_stopped:
                    self.log_updated.emit("Full analysis stopped by user.")
                    return
                while self.is_paused: self.msleep(100)
                self.msleep(10) # Simulate work
                overall_progress = int(((stage_num * 50 + i + 1) / (total_stages * 50)) * 100)
                self.progress_updated.emit(overall_progress)
            self.intermediate_result.emit({'type': 'full_analysis_stage_complete', 'stage': stage_num+1})

        self.stage_updated.emit("Finalizing Full Analysis")
        results_data = {
            'type': 'full_analysis',
            'parameters': self.parameters,
            'spacetime_component': {'complexity': np.random.rand()},
            'blackhole_component': {'schwarzschild_radius_km': np.random.uniform(1,100)},
            'cross_correlation': np.random.rand()
        }
        self.result_ready.emit(results_data)
        self.log_updated.emit("Full comprehensive analysis completed.")

    def _run_batch_simulation(self):
        self.stage_updated.emit("Initializing Batch Processing")
        tasks = self.parameters.get("tasks", [])
        if not isinstance(tasks, list) or not tasks:
            self.log_updated.emit("Batch tasks not provided or in incorrect format.")
            self.error_occurred.emit("Batch processing error: No tasks.")
            return

        self.log_updated.emit(f"Starting batch processing for {len(tasks)} tasks.")
        all_results = []
        num_tasks = len(tasks)

        for i, task_params in enumerate(tasks):
            if self.is_stopped:
                self.log_updated.emit("Batch processing stopped by user.")
                return
            while self.is_paused: self.msleep(100)

            task_name = task_params.get('name', f'Task {i+1}')
            sim_type = task_params.get('type', 'spacetime') # Default to spacetime if not specified
            params = task_params.get('parameters', {})

            self.stage_updated.emit(f"Running batch task {i+1}/{num_tasks}: {task_name} ({sim_type})")
            self.log_updated.emit(f"Task {i+1} ({task_name}) params: {params}")

            # Simulate running this sub-task (in a real scenario, you might call another worker or sim function)
            self.msleep(1000) # Simulate work for this task
            sub_result = {'task_name': task_name, 'sim_type': sim_type, 'input_params': params, 'output_data': {'metric': np.random.rand()}}
            all_results.append(sub_result)

            self.progress_updated.emit(int(((i + 1) / num_tasks) * 100))
            self.intermediate_result.emit({'type': 'batch_task_complete', 'task_index': i, 'result': sub_result})

        self.stage_updated.emit("Finalizing Batch Processing")
        results_data = {'type': 'batch', 'parameters': self.parameters, 'batch_results': all_results}
        self.result_ready.emit(results_data)
        self.log_updated.emit("Batch processing completed.")

    def _run_sweep_simulation(self):
        self.stage_updated.emit("Initializing Parameter Sweep")
        base_sim_type = self.parameters.get('base_sim_type', 'spacetime')
        sweep_configs = self.parameters.get('sweep_configs', []) # Expect list of param dicts or range configs

        if not sweep_configs: # Check if sweep_configs is empty or not a list
            self.log_updated.emit("Parameter sweep configurations not provided or invalid.")
            self.error_occurred.emit("Sweep configuration error.")
            return

        self.log_updated.emit(f"Running parameter sweep for \'{base_sim_type}\' with {len(sweep_configs)} configurations.")
        all_results = []
        num_configs = len(sweep_configs)

        for i, config_params in enumerate(sweep_configs):
            if self.is_stopped:
                self.log_updated.emit("Parameter sweep stopped by user.")
                return
            while self.is_paused: self.msleep(100)

            self.stage_updated.emit(f"Running sweep configuration {i+1}/{num_configs}")
            self.log_updated.emit(f"Config {i+1}: {config_params}")

            # Simulate running a simulation with these config_params
            self.msleep(500) # Simulate work for this configuration
            # Dummy sub-result for this configuration
            sub_result = {'config_params': config_params, 'metric_A': np.random.rand(), 'metric_B': np.random.uniform(1,100)}
            all_results.append(sub_result)

            self.progress_updated.emit(int(((i + 1) / num_configs) * 100))
            self.intermediate_result.emit({'type': 'sweep_config_complete', 'config_index': i, 'result': sub_result})

        self.stage_updated.emit("Finalizing Parameter Sweep")
        results_data = {'type': 'sweep', 'base_sim_type': base_sim_type, 'sweep_results': all_results}
        self.result_ready.emit(results_data)
        self.log_updated.emit("Parameter sweep completed.")

    def pause(self):
        self.is_paused = True
        self.log_updated.emit("Simulation paused.")
        self.stage_updated.emit("Paused")

    def resume(self):
        self.is_paused = False
        self.log_updated.emit("Simulation resumed.")
        self.stage_updated.emit("Resuming...")

    def stop(self):
        """Stop the simulation gracefully."""
        self.is_stopped = True


class AdvancedPlotCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=7, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.axes = self.fig.add_subplot(111)
        self.axes_3d = None
        self.current_theme_is_dark = False
        self.animation = None
        self.animation_frames_data = [] # Initialize to avoid attribute errors

    def update_theme_sensititve_styles(self, is_dark):
        self.current_theme_is_dark = is_dark
        tick_color = 'white' if is_dark else 'black'
        label_color = 'white' if is_dark else 'black'
        face_color = '#2E3440' if is_dark else '#F0F0F0'

        self.fig.patch.set_facecolor(face_color)

        active_axes = self.axes_3d if self.axes_3d else self.axes

        if active_axes:
            active_axes.set_facecolor(face_color)
            active_axes.xaxis.label.set_color(label_color)
            active_axes.yaxis.label.set_color(label_color)
            active_axes.tick_params(axis='x', colors=tick_color)
            active_axes.tick_params(axis='y', colors=tick_color)

            if self.axes_3d and isinstance(active_axes, Axes3D): # Ensure it's 3D
                self.axes_3d.zaxis.label.set_color(label_color)
                self.axes_3d.tick_params(axis='z', colors=tick_color)
                pane_color = face_color if is_dark else '#D0D0D0'
                pane_edge_color = 'grey' if is_dark else 'silver'
                for pane_attr in ['xaxis', 'yaxis', 'zaxis']: # Check if panes exist
                    pane = getattr(self.axes_3d, pane_attr).pane
                    pane.set_facecolor(pane_color)
                    pane.set_edgecolor(pane_edge_color)
            elif not self.axes_3d: # 2D specific
                active_axes.spines['top'].set_color(tick_color)
                active_axes.spines['bottom'].set_color(tick_color)
                active_axes.spines['left'].set_color(tick_color)
                active_axes.spines['right'].set_color(tick_color)
        self.draw_idle()

    def plot_spacetime_3d(self, results_data):
        if not self.axes_3d or not isinstance(self.axes_3d, Axes3D):
            self.fig.clear()
            self.axes_3d = self.fig.add_subplot(111, projection='3d')
            self.axes = None
        else:
            self.axes_3d.clear()

        network_data = results_data.get('network_data', {})
        positions = np.array(network_data.get('positions', []))
        nodes = network_data.get('nodes', [])
        edges = network_data.get('edges', [])

        if positions.size == 0 or positions.shape[1] < 3:
            self.axes_3d.text(0.5, 0.5, 0.5, "No 3D data or insufficient dimensions",
                              transform=self.axes_3d.transAxes, ha='center', va='center',
                              color='red' if self.current_theme_is_dark else 'darkred')
            self.update_theme_sensititve_styles(self.current_theme_is_dark) # Apply theme to empty plot
            self.draw_idle()
            return

        node_colors_raw = [n.get('color', 'blue') for n in nodes]
        node_sizes_raw = [n.get('size', 50) for n in nodes]

        node_colors = node_colors_raw if len(node_colors_raw) == len(positions) else 'skyblue'
        # Ensure node_sizes is a scalar or an array of the same length as positions
        if len(node_sizes_raw) == len(positions):
            node_sizes = node_sizes_raw
        elif len(positions) > 0 : # If positions exist but sizes don't match, use a default scalar
            node_sizes = 50
        else: # No positions, no sizes needed or default to scalar if scatter is called
            node_sizes = 50

        if len(positions) > 0:
            # Ensure node_colors is compatible in length if it's a list
            if isinstance(node_colors, list) and len(node_colors) != len(positions):
                node_colors = 'skyblue' # Fallback color

            self.axes_3d.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                                      c=node_colors,
                                      s=node_sizes,
                                      depthshade=True)

        edge_color = 'white' if self.current_theme_is_dark else 'grey'
        for edge in edges:
            source_idx = edge.get('source')
            target_idx = edge.get('target')
            if source_idx is not None and target_idx is not None and \
               source_idx < len(positions) and target_idx < len(positions):
                p1 = positions[source_idx]
                p2 = positions[target_idx]
                self.axes_3d.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                                  color=edge_color, alpha=0.5, linewidth=0.8)

        self.axes_3d.set_xlabel("X Dimension")
        self.axes_3d.set_ylabel("Y Dimension")
        self.axes_3d.set_zlabel("Z Dimension")
        title_color = 'skyblue' if self.current_theme_is_dark else 'darkblue'
        self.axes_3d.set_title("Spacetime Network Emergence (3D)", color=title_color)

        self.update_theme_sensititve_styles(self.current_theme_is_dark)
        self.draw_idle()

    def plot_results(self, results_data):
        if self.axes_3d:
            self.fig.clear()
            self.axes = self.fig.add_subplot(111)
            self.axes_3d = None
        elif not self.axes:
             self.axes = self.fig.add_subplot(111)
        self.axes.clear()

        data_to_plot = results_data.get('complexity_measures', results_data.get('data_points'))
        if data_to_plot and isinstance(data_to_plot, list) and len(data_to_plot) > 0:
            if isinstance(data_to_plot[0], list) or (isinstance(data_to_plot[0], np.ndarray) and data_to_plot[0].ndim > 0) :
                 data_array = np.array(data_to_plot)
                 self.axes.plot(data_array[:,0], data_array[:,1], marker='o', linestyle='-', color='cyan')
                 self.axes.set_xlabel("X Value")
                 self.axes.set_ylabel("Y Value")
            else:
                self.axes.plot(data_to_plot, marker='o', linestyle='-', color='lightgreen')
                self.axes.set_xlabel("Index / Time Step")
                self.axes.set_ylabel("Value")
            self.axes.set_title(f"{results_data.get('type', 'Generic')} Results")
        else:
            self.axes.text(0.5, 0.5, "No plottable data.", transform=self.axes.transAxes, ha='center')

        self.update_theme_sensititve_styles(self.current_theme_is_dark)
        self.draw_idle()

    def update_animation_frame(self, frame_idx):
        if not self.animation_frames_data or frame_idx >= len(self.animation_frames_data):
            return [] # Return empty list of artists if no data

        frame_data = self.animation_frames_data[frame_idx]

        if not self.axes_3d or not isinstance(self.axes_3d, Axes3D):
            self.fig.clear()
            self.axes_3d = self.fig.add_subplot(111, projection='3d')
            self.axes = None
            # Important: Re-apply theme after recreating axes_3d for animation
            self.update_theme_sensititve_styles(self.current_theme_is_dark)
        else:
            self.axes_3d.clear() # Clear previous frame artists

        if not isinstance(frame_data, np.ndarray): frame_data = np.array(frame_data)

        artists = []
        if frame_data.ndim == 2 and frame_data.shape[1] == 3 and frame_data.size > 0:
            scatter_plot = self.axes_3d.scatter(frame_data[:, 0], frame_data[:, 1], frame_data[:, 2], c='cyan', s=30)
            artists.extend(self.axes_3d.collections) # Add scatter plot to artists

            self.axes_3d.set_xlim((np.min(frame_data[:,0])-1, np.max(frame_data[:,0])+1))
            self.axes_3d.set_ylim((np.min(frame_data[:,1])-1, np.max(frame_data[:,1])+1))
            self.axes_3d.set_zlim((np.min(frame_data[:,2])-1, np.max(frame_data[:,2])+1))
        else: # Fallback or empty frame
            self.axes_3d.text(0.5,0.5,0.5, "Invalid/Empty frame data for 3D scatter",
                              transform=self.axes_3d.transAxes, ha='center', va='center', color='red')
            self.axes_3d.set_xlim((-1,1)); self.axes_3d.set_ylim((-1,1)); self.axes_3d.set_zlim((-1,1))


        self.axes_3d.set_xlabel("X")
        self.axes_3d.set_ylabel("Y")
        self.axes_3d.set_zlabel("Z")
        self.axes_3d.set_title(f"Spacetime Evolution Animation (Frame {frame_idx})")
        # Theme styles are applied when axes_3d is created or cleared if needed.
        # If not clearing, ensure styles are consistent.
        # self.update_theme_sensititve_styles(self.current_theme_is_dark) # Re-applying might be too much here, but good for consistency if axes are not fully cleared.

        # For blit=False, returning artists is good practice but not strictly enforced like blit=True
        # The main thing is that the draw() call will render the changes.
        # artists.extend(self.axes_3d.lines) # if any lines are drawn per frame
        return artists # Return list of artists updated in this frame

    # Placeholder plotting methods to be implemented
    def plot_blackhole_results(self, results_data):
        self.plot_results(results_data) # Fallback to generic
        self.axes.set_title("Black Hole Simulation Results (Placeholder)")
        self.draw_idle()

    def plot_experimental_results(self, results_data):
        self.plot_results(results_data) # Fallback to generic
        self.axes.set_title("Experimental Prediction Results (Placeholder)")
        self.draw_idle()

    def plot_sweep_results(self, results_data):
        # This would typically be more complex, e.g., a heatmap or parallel coordinates
        self.plot_results({'type': 'sweep', 'data_points': [[i, r.get('metric_A',0)] for i,r in enumerate(results_data.get('sweep_results',[]))]})
        self.axes.set_title("Parameter Sweep Results (Placeholder Metric A)")
        self.draw_idle()

    def plot_batch_summary(self, results_data):
        # This could be a bar chart of metrics per task, or similar summary
        self.plot_results({'type': 'batch', 'data_points': [[i, r.get('output_data',{}).get('metric',0)] for i,r in enumerate(results_data.get('batch_results',[]))]})
        self.axes.set_title("Batch Processing Summary (Placeholder Metric)")
        self.draw_idle()


# Full definitions of custom widgets BEFORE AdvancedEGQGEMMainWindow
class SweepConfigurationWidget(QWidget):
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent)
        self.main_window = main_window
        layout = QVBoxLayout(self)

        # Base simulation type selection
        base_type_group = QGroupBox("Base Simulation Type")
        base_type_layout = QVBoxLayout(base_type_group)
        self.base_sim_combo = QComboBox()
        self.base_sim_combo.addItems(["spacetime", "blackhole", "experiments"])
        base_type_layout.addWidget(self.base_sim_combo)
        layout.addWidget(base_type_group)

        # Parameter sweep configuration
        sweep_params_group = QGroupBox("Sweep Parameters")
        sweep_params_layout = QFormLayout(sweep_params_group)

        # For spacetime parameters
        self.n_subsystems_min = QSpinBox()
        self.n_subsystems_min.setRange(10, 1000)
        self.n_subsystems_min.setValue(20)
        self.n_subsystems_max = QSpinBox()
        self.n_subsystems_max.setRange(10, 1000)
        self.n_subsystems_max.setValue(100)
        self.n_subsystems_step = QSpinBox()
        self.n_subsystems_step.setRange(1, 100)
        self.n_subsystems_step.setValue(20)

        sweep_params_layout.addRow("N Subsystems Min:", self.n_subsystems_min)
        sweep_params_layout.addRow("N Subsystems Max:", self.n_subsystems_max)
        sweep_params_layout.addRow("N Subsystems Step:", self.n_subsystems_step)

        # For blackhole parameters
        self.mass_min = QDoubleSpinBox()
        self.mass_min.setRange(0.1, 1000.0)
        self.mass_min.setValue(1.0)
        self.mass_max = QDoubleSpinBox()
        self.mass_max.setRange(0.1, 1000.0)
        self.mass_max.setValue(50.0)
        self.mass_step = QDoubleSpinBox()
        self.mass_step.setRange(0.1, 100.0)
        self.mass_step.setValue(5.0)

        sweep_params_layout.addRow("Mass Min (M☉):", self.mass_min)
        sweep_params_layout.addRow("Mass Max (M☉):", self.mass_max)
        sweep_params_layout.addRow("Mass Step (M☉):", self.mass_step)

        # Evolution steps for spacetime
        self.steps_min = QSpinBox()
        self.steps_min.setRange(10, 10000)
        self.steps_min.setValue(50)
        self.steps_max = QSpinBox()
        self.steps_max.setRange(10, 10000)
        self.steps_max.setValue(200)
        self.steps_step = QSpinBox()
        self.steps_step.setRange(1, 1000)
        self.steps_step.setValue(50)

        sweep_params_layout.addRow("Evolution Steps Min:", self.steps_min)
        sweep_params_layout.addRow("Evolution Steps Max:", self.steps_max)
        sweep_params_layout.addRow("Evolution Steps Step:", self.steps_step)

        layout.addWidget(sweep_params_group)

        # Sweep execution controls
        execution_group = QGroupBox("Sweep Execution")
        execution_layout = QVBoxLayout(execution_group)

        self.parallel_checkbox = QCheckBox("Enable Parallel Execution")
        self.parallel_checkbox.setChecked(False)
        execution_layout.addWidget(self.parallel_checkbox)

        self.max_workers_spin = QSpinBox()
        self.max_workers_spin.setRange(1, 16)
        self.max_workers_spin.setValue(4)
        execution_layout.addWidget(QLabel("Max Workers:"))
        execution_layout.addWidget(self.max_workers_spin)

        layout.addWidget(execution_group)

        self.setLayout(layout)
        if self.main_window:
            self.main_window.add_log_message("SweepConfigurationWidget initialized with full UI.", "Debug")

    def get_sweep_parameters(self):
        if self.main_window:
            self.main_window.add_log_message("Sweep widget generating parameters from UI configuration.", "Debug")

        base_sim_type = self.base_sim_combo.currentText()
        sweep_configs = []

        if base_sim_type == 'spacetime':
            # Generate sweep configurations for spacetime parameters
            n_min = self.n_subsystems_min.value()
            n_max = self.n_subsystems_max.value()
            n_step = self.n_subsystems_step.value()

            steps_min = self.steps_min.value()
            steps_max = self.steps_max.value()
            steps_step = self.steps_step.value()

            for n_subsystems in range(n_min, n_max + 1, n_step):
                for steps in range(steps_min, steps_max + 1, steps_step):
                    sweep_configs.append({
                        'n_subsystems': n_subsystems,
                        'steps': steps,
                        'dimension': 3
                    })

        elif base_sim_type == 'blackhole':
            # Generate sweep configurations for blackhole parameters
            mass_min = self.mass_min.value()
            mass_max = self.mass_max.value()
            mass_step = self.mass_step.value()

            mass = mass_min
            while mass <= mass_max:
                sweep_configs.append({
                    'mass_solar_masses': mass
                })
                mass += mass_step

        elif base_sim_type == 'experiments':
            # Generate sweep configurations for experimental parameters
            experiment_types = ['LIGO Noise', 'CMB Fluctuations', 'Particle Collider Event']
            for exp_type in experiment_types:
                sweep_configs.append({
                    'experiment_type': exp_type
                })

        return {
            'base_sim_type': base_sim_type,
            'sweep_configs': sweep_configs,
            'parallel_execution': self.parallel_checkbox.isChecked(),
            'max_workers': self.max_workers_spin.value() if self.parallel_checkbox.isChecked() else 1
        }

class BatchProcessingWidget(QWidget):
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent)
        self.main_window = main_window
        self.tasks = []  # Store task configurations

        layout = QVBoxLayout(self)

        # Task creation section
        task_creation_group = QGroupBox("Create New Task")
        task_creation_layout = QFormLayout(task_creation_group)

        self.task_name_edit = QLineEdit()
        self.task_name_edit.setPlaceholderText("Enter task name...")
        task_creation_layout.addRow("Task Name:", self.task_name_edit)

        self.task_type_combo = QComboBox()
        self.task_type_combo.addItems(["spacetime", "blackhole", "experiments"])
        task_creation_layout.addRow("Task Type:", self.task_type_combo)

        # Parameters for different task types
        self.param_n_subsystems = QSpinBox()
        self.param_n_subsystems.setRange(10, 1000)
        self.param_n_subsystems.setValue(50)
        task_creation_layout.addRow("N Subsystems:", self.param_n_subsystems)

        self.param_steps = QSpinBox()
        self.param_steps.setRange(10, 10000)
        self.param_steps.setValue(100)
        task_creation_layout.addRow("Evolution Steps:", self.param_steps)

        self.param_mass = QDoubleSpinBox()
        self.param_mass.setRange(0.1, 1000.0)
        self.param_mass.setValue(10.0)
        self.param_mass.setSuffix(" M☉")
        task_creation_layout.addRow("BH Mass:", self.param_mass)

        self.param_experiment_type = QComboBox()
        self.param_experiment_type.addItems(["LIGO Noise", "CMB Fluctuations", "Particle Collider Event"])
        task_creation_layout.addRow("Experiment Type:", self.param_experiment_type)

        # Add task button
        self.add_task_button = QPushButton("Add Task to Batch")
        self.add_task_button.clicked.connect(self.add_task)
        task_creation_layout.addRow(self.add_task_button)

        layout.addWidget(task_creation_group)

        # Task list section
        task_list_group = QGroupBox("Batch Task Queue")
        task_list_layout = QVBoxLayout(task_list_group)

        self.task_list = QListWidget()
        task_list_layout.addWidget(self.task_list)

        # Task management buttons
        task_buttons_layout = QHBoxLayout()
        self.remove_task_button = QPushButton("Remove Selected")
        self.remove_task_button.clicked.connect(self.remove_task)
        self.clear_tasks_button = QPushButton("Clear All")
        self.clear_tasks_button.clicked.connect(self.clear_tasks)
        self.move_up_button = QPushButton("Move Up")
        self.move_up_button.clicked.connect(self.move_task_up)
        self.move_down_button = QPushButton("Move Down")
        self.move_down_button.clicked.connect(self.move_task_down)

        task_buttons_layout.addWidget(self.remove_task_button)
        task_buttons_layout.addWidget(self.clear_tasks_button)
        task_buttons_layout.addWidget(self.move_up_button)
        task_buttons_layout.addWidget(self.move_down_button)

        task_list_layout.addLayout(task_buttons_layout)
        layout.addWidget(task_list_group)

        # Batch execution settings
        batch_settings_group = QGroupBox("Batch Execution Settings")
        batch_settings_layout = QFormLayout(batch_settings_group)

        self.parallel_batch_checkbox = QCheckBox("Enable Parallel Batch Processing")
        batch_settings_layout.addRow(self.parallel_batch_checkbox)

        self.max_concurrent_tasks = QSpinBox()
        self.max_concurrent_tasks.setRange(1, 8)
        self.max_concurrent_tasks.setValue(2)
        batch_settings_layout.addRow("Max Concurrent Tasks:", self.max_concurrent_tasks)

        self.save_intermediate_checkbox = QCheckBox("Save Intermediate Results")
        self.save_intermediate_checkbox.setChecked(True)
        batch_settings_layout.addRow(self.save_intermediate_checkbox)

        layout.addWidget(batch_settings_group)

        self.setLayout(layout)
        if self.main_window:
            self.main_window.add_log_message("BatchProcessingWidget initialized with full UI.", "Debug")

    def add_task(self):
        """Add a new task to the batch queue."""
        task_name = self.task_name_edit.text().strip()
        if not task_name:
            task_name = f"Task {len(self.tasks) + 1}"

        task_type = self.task_type_combo.currentText()

        # Build parameters based on task type
        parameters = {}
        if task_type == "spacetime":
            parameters = {
                'n_subsystems': self.param_n_subsystems.value(),
                'steps': self.param_steps.value(),
                'dimension': 3
            }
        elif task_type == "blackhole":
            parameters = {
                'mass_solar_masses': self.param_mass.value()
            }
        elif task_type == "experiments":
            parameters = {
                'experiment_type': self.param_experiment_type.currentText()
            }

        task = {
            'name': task_name,
            'type': task_type,
            'parameters': parameters
        }

        self.tasks.append(task)
        self.update_task_list()

        # Clear task name for next entry
        self.task_name_edit.clear()

        if self.main_window:
            self.main_window.add_log_message(f"Added task '{task_name}' ({task_type}) to batch queue.", "Info")

    def remove_task(self):
        """Remove selected task from the batch queue."""
        current_row = self.task_list.currentRow()
        if 0 <= current_row < len(self.tasks):
            removed_task = self.tasks.pop(current_row)
            self.update_task_list()
            if self.main_window:
                self.main_window.add_log_message(f"Removed task '{removed_task['name']}' from batch queue.", "Info")

    def clear_tasks(self):
        """Clear all tasks from the batch queue."""
        self.tasks.clear()
        self.update_task_list()
        if self.main_window:
            self.main_window.add_log_message("Cleared all tasks from batch queue.", "Info")

    def move_task_up(self):
        """Move selected task up in the queue."""
        current_row = self.task_list.currentRow()
        if current_row > 0:
            self.tasks[current_row], self.tasks[current_row - 1] = self.tasks[current_row - 1], self.tasks[current_row]
            self.update_task_list()
            self.task_list.setCurrentRow(current_row - 1)

    def move_task_down(self):
        """Move selected task down in the queue."""
        current_row = self.task_list.currentRow()
        if 0 <= current_row < len(self.tasks) - 1:
            self.tasks[current_row], self.tasks[current_row + 1] = self.tasks[current_row + 1], self.tasks[current_row]
            self.update_task_list()
            self.task_list.setCurrentRow(current_row + 1)

    def update_task_list(self):
        """Update the task list display."""
        self.task_list.clear()
        for i, task in enumerate(self.tasks):
            item_text = f"{i+1}. {task['name']} ({task['type']})"
            if task['type'] == 'spacetime':
                item_text += f" - N:{task['parameters'].get('n_subsystems', 'N/A')}, Steps:{task['parameters'].get('steps', 'N/A')}"
            elif task['type'] == 'blackhole':
                item_text += f" - Mass:{task['parameters'].get('mass_solar_masses', 'N/A')} M☉"
            elif task['type'] == 'experiments':
                item_text += f" - Type:{task['parameters'].get('experiment_type', 'N/A')}"

            self.task_list.addItem(item_text)

    def get_batch_parameters(self):
        if self.main_window:
            self.main_window.add_log_message(f"Batch widget returning {len(self.tasks)} configured tasks.", "Debug")

        return {
            'tasks': self.tasks.copy(),  # Return a copy to avoid modification
            'parallel_execution': self.parallel_batch_checkbox.isChecked(),
            'max_concurrent_tasks': self.max_concurrent_tasks.value(),
            'save_intermediate_results': self.save_intermediate_checkbox.isChecked()
        }

class AnalysisToolsWidget(QWidget):
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent)
        self.main_window = main_window
        layout = QVBoxLayout(self)
        self.label = QLabel("Data Analysis Tools")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)
        self.data_display = QTextEdit()
        self.data_display.setReadOnly(True)
        layout.addWidget(self.data_display)
        # Add more analysis controls here, e.g., buttons for specific analyses
        self.setLayout(layout)
        if self.main_window:
            self.main_window.add_log_message("AnalysisToolsWidget initialized.", "Debug")

    def set_data(self, data):
        self.current_data = data
        try:
            def convert_numpy_for_display(obj):
                if isinstance(obj, np.ndarray): return obj.tolist()
                if isinstance(obj, (np.generic, np.number)): return obj.item()
                if isinstance(obj, datetime): return obj.isoformat()
                # Add other custom types if necessary
                try: return str(obj)
                except: return f"<Object of type {type(obj).__name__} not serializable>"

            # For very large data (like frames), show a summary instead of full dump
            if isinstance(data, dict) and 'frames' in data and isinstance(data['frames'], list) and len(data['frames']) > 5:
                display_data = data.copy()
                display_data['frames'] = f"<Animation frames: {len(data['frames'])} frames, data omitted for brevity>"
            else:
                display_data = data

            self.data_display.setText(json.dumps(display_data, indent=2, default=convert_numpy_for_display))
        except Exception as e:
            error_message = f"Error displaying data in AnalysisToolsWidget: {e}\\nData was: {str(data)[:500]}..." # Show partial data
            self.data_display.setText(error_message)
            if self.main_window:
                 self.main_window.add_log_message(f"Analysis widget display error: {e}", "Error")


class AdvancedEGQGEMMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EG-QGEM Advanced Simulation Interface")
        self.setGeometry(100, 100, 1600, 900)
        self.current_theme = 'dark'
        self.themes = self.create_themes()
        self.worker = None
        self.current_results = None
        self.animation_frames_data = None

        self._init_core_ui_elements()
        self.init_ui()
        self._connect_actions()

        self.apply_theme(self.current_theme)
        self.add_log_message("Advanced EG-QGEM Interface Initialized.", "System")

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_time)
        self.timer.start(1000)

        self.monitor_timer = QTimer(self)
        self.monitor_timer.timeout.connect(self.update_performance_display_periodic)
        self.monitor_timer.start(2000)

    def _init_core_ui_elements(self):
        self.menubar = self.menuBar()
        self.toolbar = self.addToolBar("Main Toolbar")
        self.toolbar.setMovable(False) # Preference
        self.statusbar = self.statusBar()
        self.status_label = QLabel("Status: Ready")
        self.statusbar.addWidget(self.status_label)
        self.memory_status = QLabel("Mem: N/A")
        self.time_status = QLabel("Time: 00:00:00")
        self.statusbar.addPermanentWidget(QLabel(" | "))
        self.statusbar.addPermanentWidget(self.memory_status)
        self.statusbar.addPermanentWidget(QLabel(" | "))
        self.statusbar.addPermanentWidget(self.time_status)
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        self.h_splitter = QSplitter(Qt.Horizontal)
        self.plot_canvas = AdvancedPlotCanvas(self, width=7, height=6)
        self.h_splitter.addWidget(self.plot_canvas)
        main_layout.addWidget(self.h_splitter)

        # Initialize log text widget
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)

        # Web view: fallback to QLabel if QtWebEngine not available
        if WEB_ENGINE_AVAILABLE:
            self.web_view = QWebEngineView()
            self.web_view.setMinimumHeight(300)
        else:
            self.web_view = QLabel("PyQtWebEngine not installed. Install with 'pip install PyQtWebEngine'.")
            self.web_view.setAlignment(Qt.AlignCenter)
            self.web_view.setWordWrap(True)

        self.create_docks()
        self.create_enhanced_menu_bar()
        self.create_enhanced_toolbar()

    def create_docks(self):
        self.control_dock = QDockWidget("🎛️ Simulation Controls", self)
        self.control_panel_widget = self._create_control_panel_content()
        self.control_dock.setWidget(self.control_panel_widget)
        self.control_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.control_dock)

        self.log_dock = QDockWidget("📜 System Log", self)
        log_widget_container = QWidget()
        log_layout = QVBoxLayout(log_widget_container)
        log_layout.addWidget(self.log_text)
        self.clear_log_button = QPushButton("Clear Log")
        self.clear_log_button.setIcon(QIcon.fromTheme("edit-clear", QIcon("icons/edit-clear.png")))
        log_layout.addWidget(self.clear_log_button)
        self.log_dock.setWidget(log_widget_container)
        self.log_dock.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.log_dock)

        self.interactive_plot_dock = QDockWidget("🌐 Interactive Plot", self)
        self.interactive_plot_dock.setWidget(self.web_view)
        self.interactive_plot_dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.BottomDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, self.interactive_plot_dock)
        self.interactive_plot_dock.hide()

        self.monitor_dock = QDockWidget("📊 Real-time Monitor", self)
        self.monitor_widget = self._create_monitor_panel_content()
        self.monitor_dock.setWidget(self.monitor_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.monitor_dock)
        self.monitor_dock.hide()

        self.analysis_dock = QDockWidget("📈 Data Analysis Tools", self)
        self.analysis_tool_widget = AnalysisToolsWidget(self, main_window=self)
        self.analysis_dock.setWidget(self.analysis_tool_widget)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.analysis_dock) # Changed to Bottom for variety
        self.analysis_dock.hide()

        self.sweep_dock = QDockWidget("↔️ Parameter Sweep", self) # Added icon
        self.sweep_configuration_widget = SweepConfigurationWidget(self, main_window=self)
        self.sweep_dock.setWidget(self.sweep_configuration_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.sweep_dock) # Keep on left, below controls or tabbed
        self.sweep_dock.hide()

        self.batch_dock = QDockWidget("📊 Batch Processing", self) # Added icon
        self.batch_processing_widget = BatchProcessingWidget(self, main_window=self)
        self.batch_dock.setWidget(self.batch_processing_widget)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.batch_dock) # Can be tabbed with Log or Analysis
        self.batch_dock.hide()

    def _create_control_panel_content(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setAlignment(Qt.AlignTop)

        sim_type_group = QGroupBox("🎯 Simulation Type")
        sim_type_layout = QVBoxLayout(sim_type_group)
        self.sim_type = QComboBox()
        self.sim_type.addItems([
            "🌌 Spacetime Emergence", "🕳️ Black Hole Physics", "🔬 Experimental Predictions",
            "⚙️ Full Comprehensive Analysis", "↔️ Parameter Sweep Study", "📊 Batch Processing"
        ])
        sim_type_layout.addWidget(self.sim_type)
        layout.addWidget(sim_type_group)

        self.parameters_group = QGroupBox("Parameters")
        # Using a QVBoxLayout for parameters_group to hold multiple QWidget containers
        self.parameters_group_layout = QVBoxLayout(self.parameters_group)

        self.spacetime_params_widget = QWidget()
        spacetime_form_layout = QFormLayout(self.spacetime_params_widget)
        self.n_subsystems = QSpinBox()
        self.n_subsystems.setRange(10, 1000); self.n_subsystems.setValue(50)
        spacetime_form_layout.addRow("Number of Subsystems (N):", self.n_subsystems)
        self.n_subsystems_slider = QSlider(Qt.Horizontal)
        self.n_subsystems_slider.setRange(10,1000); self.n_subsystems_slider.setValue(50)
        spacetime_form_layout.addRow("N Slider:", self.n_subsystems_slider)
        self.evolution_steps = QSpinBox()
        self.evolution_steps.setRange(10, 10000); self.evolution_steps.setValue(100)
        spacetime_form_layout.addRow("Evolution Steps:", self.evolution_steps)
        self.parameters_group_layout.addWidget(self.spacetime_params_widget)

        self.blackhole_params_widget = QWidget()
        bh_form_layout = QFormLayout(self.blackhole_params_widget)
        self.mass_solar = QDoubleSpinBox()
        self.mass_solar.setRange(0.1, 1000.0); self.mass_solar.setValue(10.0); self.mass_solar.setSuffix(" M☉")
        bh_form_layout.addRow("Black Hole Mass:", self.mass_solar)
        self.parameters_group_layout.addWidget(self.blackhole_params_widget)

        # Placeholder for experimental params (can be a QWidget with its own QFormLayout)
        self.experimental_params_widget = QWidget()
        exp_form_layout = QFormLayout(self.experimental_params_widget)
        self.exp_type_combo = QComboBox()
        self.exp_type_combo.addItems(["LIGO Noise", "CMB Fluctuations", "Particle Collider Event"])
        exp_form_layout.addRow("Experiment Type:", self.exp_type_combo)
        self.parameters_group_layout.addWidget(self.experimental_params_widget)


        layout.addWidget(self.parameters_group)
        self.sim_type.currentTextChanged.connect(self.update_parameter_panel_visibility)
        self.update_parameter_panel_visibility(self.sim_type.currentText())

        exec_group = QGroupBox("⚡ Execution Controls")
        exec_layout = QVBoxLayout(exec_group)
        self.run_button = QPushButton("🚀 Run Simulation")
        self.run_button.setIcon(QIcon.fromTheme("media-playback-start", QIcon("icons/run.png")))
        exec_layout.addWidget(self.run_button)
        self.progress_bar = QProgressBar(); self.progress_bar.setVisible(False)
        exec_layout.addWidget(self.progress_bar)
        self.stage_label = QLabel("Stage: Idle"); self.stage_label.setAlignment(Qt.AlignCenter); self.stage_label.setVisible(False)
        exec_layout.addWidget(self.stage_label)

        control_buttons_layout = QHBoxLayout()
        self.pause_btn = QPushButton("⏸️ Pause"); self.pause_btn.setEnabled(False)
        self.pause_btn.setIcon(QIcon.fromTheme("media-playback-pause", QIcon("icons/pause.png")))
        control_buttons_layout.addWidget(self.pause_btn)
        self.stop_btn = QPushButton("⏹️ Stop"); self.stop_btn.setEnabled(False)
        self.stop_btn.setIcon(QIcon.fromTheme("media-playback-stop", QIcon("icons/stop.png")))
        control_buttons_layout.addWidget(self.stop_btn)
        exec_layout.addLayout(control_buttons_layout)
        layout.addWidget(exec_group)

        panel.setLayout(layout)
        return panel

    def update_parameter_panel_visibility(self, sim_type_text):
        is_spacetime = "Spacetime Emergence" in sim_type_text
        is_blackhole = "Black Hole Physics" in sim_type_text
        is_full_analysis = "Full Comprehensive Analysis" in sim_type_text
        is_sweep = "Parameter Sweep Study" in sim_type_text
        is_batch = "Batch Processing" in sim_type_text
        is_experimental = "Experimental Predictions" in sim_type_text

        self.spacetime_params_widget.setVisible(is_spacetime or is_full_analysis or (is_sweep and self.sweep_configuration_widget.get_sweep_parameters().get('base_sim_type') == 'spacetime') ) # Show if base for sweep
        self.blackhole_params_widget.setVisible(is_blackhole or is_full_analysis or (is_sweep and self.sweep_configuration_widget.get_sweep_parameters().get('base_sim_type') == 'blackhole')) # Show if base for sweep
        self.experimental_params_widget.setVisible(is_experimental or (is_sweep and self.sweep_configuration_widget.get_sweep_parameters().get('base_sim_type') == 'experiments'))


        if hasattr(self, 'sweep_dock'): self.sweep_dock.setVisible(is_sweep)
        if hasattr(self, 'batch_dock'): self.batch_dock.setVisible(is_batch)

        # Hide general parameter group if sweep/batch docks are visible, as they handle their own params
        self.parameters_group.setVisible(not (is_sweep or is_batch))


        if is_experimental: self.parameters_group.setTitle("Experimental Parameters")
        elif is_sweep: self.parameters_group.setTitle("Base Parameters (for non-swept values)") # This group might be hidden
        elif is_batch: self.parameters_group.setTitle("Global Batch Parameters (if any)") # This group might be hidden
        elif is_spacetime: self.parameters_group.setTitle("Spacetime Parameters")
        elif is_blackhole: self.parameters_group.setTitle("Blackhole Parameters")
        else: self.parameters_group.setTitle("Simulation Parameters")


    def _create_monitor_panel_content(self):
        monitor_panel = QWidget()
        layout = QVBoxLayout(monitor_panel)
        layout.setAlignment(Qt.AlignTop)

        perf_group = QGroupBox("Performance Metrics")
        perf_layout = QFormLayout(perf_group)
        self.cpu_label = QLabel("CPU: N/A")
        perf_layout.addRow(self.cpu_label)
        self.memory_label = QLabel("Memory: N/A")
        perf_layout.addRow(self.memory_label)
        layout.addWidget(perf_group)

        sim_status_group = QGroupBox("Simulation Status")
        sim_status_layout = QFormLayout(sim_status_group)
        self.sim_time_label = QLabel("Elapsed Sim Time: 0s")
        sim_status_layout.addRow(self.sim_time_label)
        self.current_stage_label_monitor = QLabel("Current Stage: Idle")
        sim_status_layout.addRow(self.current_stage_label_monitor)
        self.intermediate_results_display = QTextEdit() # For intermediate results
        self.intermediate_results_display.setReadOnly(True)
        self.intermediate_results_display.setFixedHeight(100) # Limit height
        sim_status_layout.addRow("Intermediate Updates:", self.intermediate_results_display)
        layout.addWidget(sim_status_group)

        layout.addStretch()
        return monitor_panel

    def create_enhanced_menu_bar(self):
        file_menu = self.menubar.addMenu('📁 &File') # Added icon
        self.new_action = QAction(QIcon.fromTheme("document-new", QIcon("icons/new.png")), '&New Project', self)
        file_menu.addAction(self.new_action)
        self.open_action = QAction(QIcon.fromTheme("document-open", QIcon("icons/open.png")), '&Open Project', self)
        file_menu.addAction(self.open_action)
        self.save_action = QAction(QIcon.fromTheme("document-save", QIcon("icons/save.png")), '&Save Project', self)
        file_menu.addAction(self.save_action)
        file_menu.addSeparator()
        export_menu = file_menu.addMenu(QIcon.fromTheme("document-export", QIcon("icons/export.png")),'📤 &Export')
        self.export_results_action = QAction(QIcon("icons/json.png"),'Export &Results as JSON...', self) # Custom icon example
        export_menu.addAction(self.export_results_action)
        self.export_plots_action = QAction(QIcon("icons/image.png"),'Export Current &Plot as PNG...', self) # Custom icon example
        export_menu.addAction(self.export_plots_action)
        file_menu.addSeparator()
        self.exit_action = QAction(QIcon.fromTheme("application-exit", QIcon("icons/exit.png")), '&Exit', self)
        file_menu.addAction(self.exit_action)

        view_menu = self.menubar.addMenu('👁️ &View') # Added icon
        theme_menu = view_menu.addMenu(QIcon.fromTheme("preferences-desktop-theme", QIcon("icons/theme.png")),'🎨 &Themes')
        self.theme_action_group = QActionGroup(self)

        self.light_theme_action = QAction('&Light Theme', self, checkable=True)
        theme_menu.addAction(self.light_theme_action)
        self.theme_action_group.addAction(self.light_theme_action)

        self.dark_theme_action = QAction('&Dark Theme', self, checkable=True)
        theme_menu.addAction(self.dark_theme_action)
        self.theme_action_group.addAction(self.dark_theme_action)
        self.dark_theme_action.setChecked(True)

        self.plasma_theme_action = QAction('&Plasma Theme (Plotly)', self, checkable=True) # Clarified
        theme_menu.addAction(self.plasma_theme_action)
        self.theme_action_group.addAction(self.plasma_theme_action)

        view_menu.addSeparator()
        # Add toggle actions for all docks
        if hasattr(self, 'control_dock'):
            action = self.control_dock.toggleViewAction()
            action.setIcon(QIcon.fromTheme("preferences-system")) # Example icon
            view_menu.addAction(action)
        if hasattr(self, 'log_dock'):
            action = self.log_dock.toggleViewAction()
            action.setIcon(QIcon.fromTheme("text-x-generic"))
            view_menu.addAction(action)
        if hasattr(self, 'monitor_dock'):
            action = self.monitor_dock.toggleViewAction()
            action.setIcon(QIcon.fromTheme("utilities-system-monitor"))
            view_menu.addAction(action)
        if hasattr(self, 'analysis_dock'):
            action = self.analysis_dock.toggleViewAction()
            action.setIcon(QIcon.fromTheme("office-chart-area"))
            view_menu.addAction(action)
        if hasattr(self, 'sweep_dock'):
            action = self.sweep_dock.toggleViewAction()
            action.setIcon(QIcon.fromTheme("view-sort-ascending")) # Example
            view_menu.addAction(action)
        if hasattr(self, 'batch_dock'):
            action = self.batch_dock.toggleViewAction()
            action.setIcon(QIcon.fromTheme("view-list-tree")) # Example
            view_menu.addAction(action)
        if hasattr(self, 'interactive_plot_dock'):
            self.interactive_plot_dock_action = self.interactive_plot_dock.toggleViewAction()
            self.interactive_plot_dock_action.setText('🌐 Interactive Plot') # Already set, but good practice
            self.interactive_plot_dock_action.setIcon(QIcon.fromTheme("applications-internet"))
            view_menu.addAction(self.interactive_plot_dock_action)

        tools_menu = self.menubar.addMenu('🔧 &Tools') # Added icon
        self.benchmark_action = QAction(QIcon.fromTheme("speedometer", QIcon("icons/benchmark.png")),'&Performance Benchmark', self)
        tools_menu.addAction(self.benchmark_action)
        self.optimizer_action = QAction(QIcon.fromTheme("system-run", QIcon("icons/optimizer.png")),'&Parameter Optimizer', self) # Example icon
        tools_menu.addAction(self.optimizer_action)
        tools_menu.addSeparator()
        self.research_mode_action = QAction(QIcon.fromTheme("user-status-pending", QIcon("icons/research.png")),'&Research Mode', self, checkable=True) # Example icon
        tools_menu.addAction(self.research_mode_action)

        help_menu = self.menubar.addMenu('❓ &Help') # Added icon
        self.tutorial_action = QAction(QIcon.fromTheme("help-contents", QIcon("icons/tutorial.png")),'&Tutorial', self)
        help_menu.addAction(self.tutorial_action)
        self.docs_action = QAction(QIcon.fromTheme("help-faq", QIcon("icons/docs.png")),'&Documentation', self)
        help_menu.addAction(self.docs_action)
        help_menu.addSeparator()
        self.about_action = QAction(QIcon.fromTheme("help-about", QIcon("icons/about.png")),'&About EG-QGEM', self)
        help_menu.addAction(self.about_action)


    def create_enhanced_toolbar(self):
        self.quick_spacetime_action = QAction(QIcon.fromTheme("applications-science", QIcon("icons/spacetime.png")), "Quick Spacetime", self)
        self.toolbar.addAction(self.quick_spacetime_action)
        self.quick_blackhole_action = QAction(QIcon.fromTheme("weather-clear-night", QIcon("icons/blackhole.png")), "Quick Blackhole", self)
        self.toolbar.addAction(self.quick_blackhole_action)
        self.toolbar.addSeparator()
        self.start_animation_action = QAction(QIcon.fromTheme("media-playback-start", QIcon("icons/play.png")), "Play Animation", self)
        self.start_animation_action.setEnabled(False) # Initially disabled
        self.toolbar.addAction(self.start_animation_action)
        self.fullscreen_action = QAction(QIcon.fromTheme("view-fullscreen", QIcon("icons/fullscreen.png")), "Fullscreen Plot", self) # Clarified
        self.fullscreen_action.setCheckable(True) # Make it a toggle
        self.toolbar.addAction(self.fullscreen_action)

    def _connect_actions(self):
        # File menu
        self.new_action.triggered.connect(self.new_project)
        self.open_action.triggered.connect(self.open_project)
        self.save_action.triggered.connect(self.save_project)
        self.export_results_action.triggered.connect(self.export_results)
        self.export_plots_action.triggered.connect(self.export_current_plot)
        self.exit_action.triggered.connect(self.close) # QMainWindow.close is a slot

        # View menu (Themes)
        self.light_theme_action.triggered.connect(lambda: self.apply_theme('light'))
        self.dark_theme_action.triggered.connect(lambda: self.apply_theme('dark'))
        self.plasma_theme_action.triggered.connect(lambda: self.apply_theme('plasma'))

        # Tools menu
        self.benchmark_action.triggered.connect(self.run_benchmark)
        self.optimizer_action.triggered.connect(self.open_optimizer)
        self.research_mode_action.triggered.connect(self.set_research_mode)

        # Help menu
        self.tutorial_action.triggered.connect(self.show_tutorial)
        self.docs_action.triggered.connect(self.show_documentation)
        self.about_action.triggered.connect(self.show_enhanced_about)

        # Toolbar actions
        self.quick_spacetime_action.triggered.connect(self.quick_spacetime_sim)
        self.quick_blackhole_action.triggered.connect(self.quick_blackhole_sim)
        self.start_animation_action.triggered.connect(self.start_animation_from_button)
        self.fullscreen_action.triggered.connect(self.toggle_fullscreen_plot) # Changed handler

        # Control panel buttons
        if hasattr(self, 'run_button'): self.run_button.clicked.connect(self.run_enhanced_simulation)
        if hasattr(self, 'pause_btn'): self.pause_btn.clicked.connect(self.pause_simulation)
        if hasattr(self, 'stop_btn'): self.stop_btn.clicked.connect(self.stop_simulation)
        if hasattr(self, 'clear_log_button'): self.clear_log_button.clicked.connect(self.clear_log)

        # Control panel links
        if hasattr(self, 'n_subsystems_slider') and hasattr(self, 'n_subsystems'):
            self.n_subsystems_slider.valueChanged.connect(self.n_subsystems.setValue)
            self.n_subsystems.valueChanged.connect(self.n_subsystems_slider.setValue)


    def apply_theme(self, theme_name):
        if theme_name in self.themes:
            self.setPalette(self.themes[theme_name])
            self.current_theme = theme_name
            is_dark = theme_name != 'light' # Plasma is also dark-ish background

            # Update plot canvas first
            if hasattr(self, 'plot_canvas'):
                self.plot_canvas.update_theme_sensititve_styles(is_dark)

            # Update log text style
            if hasattr(self, 'log_text'):
                log_palette = self.themes[theme_name]
                self.log_text.setStyleSheet(f"background-color: {log_palette.base().color().name()}; color: {log_palette.text().color().name()};")

            # Update other custom widgets if they have theme-sensitive elements
            # For example, AnalysisToolsWidget's QTextEdit
            if hasattr(self, 'analysis_tool_widget') and hasattr(self.analysis_tool_widget, 'data_display'):
                 analysis_palette = self.themes[theme_name]
                 self.analysis_tool_widget.data_display.setStyleSheet(
                     f"background-color: {analysis_palette.base().color().name()}; color: {analysis_palette.text().color().name()};"
                 )

            if hasattr(self, 'intermediate_results_display'):
                monitor_palette = self.themes[theme_name]
                self.intermediate_results_display.setStyleSheet(
                    f"background-color: {monitor_palette.base().color().name()}; color: {monitor_palette.text().color().name()};"
                )


            # Re-render interactive plot if visible and data exists
            if WEB_ENGINE_AVAILABLE and hasattr(self, 'web_view') and self.interactive_plot_dock and self.interactive_plot_dock.isVisible() and self.current_results:
                 self.create_interactive_plot(self.current_results)

        # Update theme action checks
        if hasattr(self, 'light_theme_action'): self.light_theme_action.setChecked(theme_name == 'light')
        if hasattr(self, 'dark_theme_action'): self.dark_theme_action.setChecked(theme_name == 'dark')
        if hasattr(self, 'plasma_theme_action'): self.plasma_theme_action.setChecked(theme_name == 'plasma')

        self.update() # Force repaint of the main window


    def run_enhanced_simulation(self):
        sim_type_text = self.sim_type.currentText()
        parameters = {}
        worker_type = ''

        if "Spacetime Emergence" in sim_type_text:
            parameters = {'n_subsystems': self.n_subsystems.value(), 'steps': self.evolution_steps.value(), 'dimension': 3}
            worker_type = 'spacetime'
        elif "Black Hole Physics" in sim_type_text:
            parameters = {'mass_solar_masses': self.mass_solar.value()}
            worker_type = 'blackhole'
        elif "Experimental Predictions" in sim_type_text:
            parameters = {'experiment_type': self.exp_type_combo.currentText()}
            worker_type = 'experiments'
        elif "Full Comprehensive Analysis" in sim_type_text:
            parameters = {
                'spacetime_params': {'n_subsystems': self.n_subsystems.value(), 'steps': self.evolution_steps.value(), 'dimension': 3},
                'blackhole_params': {'mass_solar_masses': self.mass_solar.value()}
                # Potentially add experimental params too
            }
            worker_type = 'full'
        elif "Parameter Sweep Study" in sim_type_text:
            worker_type = 'sweep'
            if hasattr(self.sweep_configuration_widget, 'get_sweep_parameters'):
                parameters = self.sweep_configuration_widget.get_sweep_parameters()
                # Optionally merge base params from main UI if sweep widget doesn't provide all
                base_params_for_sweep = {}
                if parameters.get('base_sim_type') == 'spacetime':
                    base_params_for_sweep = {'n_subsystems': self.n_subsystems.value(), 'steps': self.evolution_steps.value(), 'dimension': 3}
                elif parameters.get('base_sim_type') == 'blackhole':
                     base_params_for_sweep = {'mass_solar_masses': self.mass_solar.value()}
                # The sweep_configs in parameters should override these base_params_for_sweep as needed per run
                parameters['base_config'] = base_params_for_sweep # Store for reference or merging in worker
            else:
                self.add_log_message("Sweep widget not fully available. Using placeholder sweep.", "Warning")
                parameters = {'base_sim_type': 'spacetime', 'sweep_configs': [{'n_subsystems': v, 'steps':30} for v in range(20,50,10)]}
        elif "Batch Processing" in sim_type_text:
            worker_type = 'batch'
            if hasattr(self.batch_processing_widget, 'get_batch_parameters'):
                parameters = self.batch_processing_widget.get_batch_parameters()
            else:
                self.add_log_message("Batch widget not fully available. Using placeholder batch.", "Warning")
                parameters = {'tasks': [{'name': 'Default Task', 'type': 'spacetime', 'parameters': {'n_subsystems': 20}}]}
        else:
            self.add_log_message(f"Unknown sim type: {sim_type_text}", "Error"); return

        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Simulation Busy", "A simulation is already running. Please wait or stop it.")
            return

        self.worker = AdvancedSimulationWorker(worker_type, parameters)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.stage_updated.connect(self.update_stage)
        self.worker.log_updated.connect(self.add_log_message_from_worker)
        self.worker.result_ready.connect(self.handle_enhanced_results)
        self.worker.error_occurred.connect(self.handle_simulation_error)
        self.worker.performance_updated.connect(self.update_performance_display)
        self.worker.intermediate_result.connect(self.handle_intermediate_result)

        self.run_button.setEnabled(False); self.pause_btn.setEnabled(True); self.stop_btn.setEnabled(True)
        self.sim_type.setEnabled(False) # Disable changing sim type while running
        self.progress_bar.setValue(0) # Reset progress bar
        self.progress_bar.setVisible(True); self.stage_label.setVisible(True)
        self.start_animation_action.setEnabled(False) # Disable animation button during sim
        self.animation_frames_data = None # Clear previous animation data
        if self.plot_canvas.animation: # Stop any ongoing animation
            try: self.plot_canvas.animation.event_source.stop()
            except AttributeError: pass
            self.plot_canvas.animation = None

        self.worker.start()
        self.add_log_message(f"🚀 {sim_type_text} simulation started with params: {parameters}", "System")


    def handle_enhanced_results(self, results):
        self.current_results = results
        self.add_log_message(f"Sim \'{results.get('type','N/A')}\' finished. Results received.", "Info")
        result_type = results.get('type')
        try:
            # Clear previous animation data before plotting new results
            self.animation_frames_data = None
            if self.plot_canvas.animation:
                try: self.plot_canvas.animation.event_source.stop()
                except AttributeError: pass
                self.plot_canvas.animation = None

            self.start_animation_action.setEnabled(False) # Disable by default

            if result_type == 'spacetime':
                self.plot_canvas.plot_spacetime_3d(results)
                if 'frames' in results and results['frames']:
                    self.animation_frames_data = results['frames']
                    self.add_log_message(f"Animation data ({len(results['frames'])} frames) available. Use 'Play Animation' button.", "Info")
                    self.start_animation_action.setEnabled(True)
            elif result_type == 'blackhole':
                self.plot_canvas.plot_blackhole_results(results)
            elif result_type == 'experiments':
                self.plot_canvas.plot_experimental_results(results)
            elif result_type == 'full_analysis':
                # Could be a multi-panel plot or a summary
                self.plot_canvas.plot_results(results)
            elif result_type == 'sweep':
                self.plot_canvas.plot_sweep_results(results)
            elif result_type == 'batch':
                self.plot_canvas.plot_batch_summary(results)
            else:
                self.add_log_message(f"No specific plot handler for result type: {result_type}. Using generic plot.", "Warning")
                self.plot_canvas.plot_results(results)
        except Exception as e:
            self.add_log_message(f"Plotting error: {e}", "Error")
            import traceback
            self.add_log_message(traceback.format_exc(), "Debug")


        if hasattr(self.analysis_tool_widget, 'set_data'):
            self.analysis_tool_widget.set_data(results)
            if hasattr(self, 'analysis_dock') and not self.analysis_dock.isVisible():
                self.analysis_dock.show()
                self.analysis_dock.raise_() # Bring to front

        if WEB_ENGINE_AVAILABLE: self.create_interactive_plot(results)

        self.run_button.setEnabled(True); self.pause_btn.setEnabled(False); self.stop_btn.setEnabled(False)
        self.sim_type.setEnabled(True) # Re-enable sim type choice
        self.progress_bar.setVisible(False); self.stage_label.setVisible(False)
        self.worker = None
        self.add_log_message("✅ Simulation complete! Ready for next task.", "Success")


    def prepare_animation(self, frames_data):
        if not frames_data:
            self.add_log_message("No animation frames to play.", "Warning"); return

        self.plot_canvas.animation_frames_data = frames_data

        if self.plot_canvas.animation:
            try: self.plot_canvas.animation.event_source.stop()
            except AttributeError: pass # Already stopped or no event source
            self.plot_canvas.animation = None # Clear old animation object

        if not self.plot_canvas.axes_3d or not isinstance(self.plot_canvas.axes_3d, Axes3D):
            self.plot_canvas.fig.clear()
            self.plot_canvas.axes_3d = self.plot_canvas.fig.add_subplot(111, projection='3d')
            self.plot_canvas.axes = None
            self.plot_canvas.update_theme_sensititve_styles(self.plot_canvas.current_theme_is_dark)


        self.plot_canvas.animation = FuncAnimation(
            self.plot_canvas.fig,
            self.plot_canvas.update_animation_frame,
            frames=len(frames_data),
            interval=100, blit=False, repeat=True
        )
        self.plot_canvas.draw_idle()
        self.add_log_message(f"Animation started with {len(frames_data)} frames.", "Info")

    def start_animation_from_button(self):
        if self.plot_canvas.animation and self.plot_canvas.animation.event_source:
             self.add_log_message("Animation is already playing or prepared.", "Info")
             # Optionally, restart it:
             # self.plot_canvas.animation.frame_seq = self.plot_canvas.animation.new_frame_seq()
             # self.plot_canvas.draw_idle()
             return

        if self.animation_frames_data:
            self.prepare_animation(self.animation_frames_data)
        else:
            self.add_log_message("No animation data loaded from the last simulation. Run a simulation that generates frames first.", "Warning")
            QMessageBox.information(self, "No Animation Data", "Please run a simulation (e.g., Spacetime Emergence) that generates animation frames first.")

    def create_interactive_plot(self, results):
        if not WEB_ENGINE_AVAILABLE or not self.web_view or not isinstance(self.web_view, QWebEngineView):
            # self.add_log_message("Interactive plot requires PyQtWebEngine, which is not available or web_view not initialized.", "Warning")
            if hasattr(self, 'interactive_plot_dock'): self.interactive_plot_dock.hide()
            return

        try:
            fig = None
            result_type = results.get('type')

            # Determine colors based on current theme
            palette = self.palette() # Get current QPalette
            bg_color = palette.window().color().name() # General background
            text_color = palette.text().color().name()
            grid_color = palette.midlight().color().name() # A lighter color for grid lines

            plot_layout = go.Layout(
                paper_bgcolor=bg_color,
                plot_bgcolor=bg_color, # Plot area background
                font=dict(color=text_color),
                xaxis=dict(gridcolor=grid_color, linecolor=text_color, zerolinecolor=grid_color),
                yaxis=dict(gridcolor=grid_color, linecolor=text_color, zerolinecolor=grid_color),
                scene=dict( # For 3D plots
                    xaxis=dict(backgroundcolor=bg_color, gridcolor=grid_color, linecolor=text_color, zerolinecolor=grid_color, title_font_color=text_color, tickfont_color=text_color),
                    yaxis=dict(backgroundcolor=bg_color, gridcolor=grid_color, linecolor=text_color, zerolinecolor=grid_color, title_font_color=text_color, tickfont_color=text_color),
                    zaxis=dict(backgroundcolor=bg_color, gridcolor=grid_color, linecolor=text_color, zerolinecolor=grid_color, title_font_color=text_color, tickfont_color=text_color)
                )
            )


            if result_type == 'spacetime' and 'network_data' in results:
                network = results['network_data']
                pos = np.array(network.get('positions', []))
                if pos.ndim == 2 and pos.shape[1] == 3 and pos.shape[0] > 0:
                    nodes_trace = go.Scatter3d(
                        x=pos[:,0], y=pos[:,1], z=pos[:,2],
                        mode='markers',
                        marker=dict(
                            size=[n.get('size',5)*2 for n in network.get('nodes', [])] if network.get('nodes') else 5, # Scale size
                            color=[n.get('color', 'blue') for n in network.get('nodes', [])] if network.get('nodes') else 'blue',
                            opacity=0.8
                        ),
                        text=[f"Node {n.get('id','N/A')}" for n in network.get('nodes', [])] if network.get('nodes') else None,
                        hoverinfo='text'
                    )
                    # Edges
                    edge_x, edge_y, edge_z = [], [], []
                    for edge in network.get('edges', []):
                        try:
                            p1_idx, p2_idx = edge
                        except (ValueError, TypeError):
                            continue
                        if p1_idx < len(pos) and p2_idx < len(pos):
                            edge_x.extend([pos[p1_idx][0], pos[p2_idx][0], None])
                            edge_y.extend([pos[p1_idx][1], pos[p2_idx][1], None])
                            edge_z.extend([pos[p1_idx][2], pos[p2_idx][2], None])

                    edges_trace = go.Scatter3d(
                        x=edge_x, y=edge_y, z=edge_z,
                        mode='lines',
                        line=dict(color='grey', width=1),
                        hoverinfo='none'
                    )
                    fig = go.Figure(data=[nodes_trace, edges_trace], layout=plot_layout)
                    fig.update_layout(title_text="Interactive Spacetime Network (3D)", title_font_color=text_color)

            elif result_type == 'blackhole':
                data = []
                if 'event_horizon_data' in results: # Example: plot event horizon if 2D
                    eh_data = np.array(results['event_horizon_data'])
                    if eh_data.ndim == 2 and eh_data.shape[1] == 2:
                         data.append(go.Scatter(x=eh_data[:,0], y=eh_data[:,1], mode='lines', name='Event Horizon'))
                if 'dummy_merger_waveform' in results:
                    waveform = results['dummy_merger_waveform']
                    data.append(go.Scatter(y=waveform, mode='lines', name='Merger Waveform (Simulated)'))
                if data:
                    fig = go.Figure(data=data, layout=plot_layout)
                    fig.update_layout(title_text="Interactive Black Hole Properties", title_font_color=text_color)

            elif result_type == 'experiments' and 'data_points' in results:
                exp_data = np.array(results['data_points'])
                if exp_data.ndim == 2 and exp_data.shape[1] == 2:
                    fig = go.Figure(data=[go.Scatter(x=exp_data[:,0], y=exp_data[:,1], mode='markers', name='Experimental Data')], layout=plot_layout)
                    fig.update_layout(title_text="Interactive Experimental Data", title_font_color=text_color)

            elif result_type == 'sweep' and 'sweep_results' in results:
                sweep_res = results['sweep_results']
                # Example: plot one metric vs. a parameter if simple
                # For more complex sweeps, consider parallel coordinates or heatmaps
                param_key_guess = list(sweep_res[0]['config_params'].keys())[0] if sweep_res and sweep_res[0]['config_params'] else 'param'
                x_vals = [r['config_params'].get(param_key_guess, i) for i, r in enumerate(sweep_res)]
                y_vals_A = [r.get('metric_A', None) for r in sweep_res]
                y_vals_B = [r.get('metric_B', None) for r in sweep_res]

                data_traces = []
                if not all(y is None for y in y_vals_A):
                    data_traces.append(go.Scatter(x=x_vals, y=y_vals_A, mode='lines+markers', name='Metric A'))
                if not all(y is None for y in y_vals_B):
                    data_traces.append(go.Scatter(x=x_vals, y=y_vals_B, mode='lines+markers', name='Metric B'))

                if data_traces:
                    fig = go.Figure(data=data_traces, layout=plot_layout)
                    fig.update_layout(title_text=f"Interactive Sweep Results (vs. {param_key_guess})", title_font_color=text_color, xaxis_title=param_key_guess)


            if fig:
                # Use 'cdn' for include_plotlyjs as it's a valid string and often preferred.
                html_content = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
                self.web_view.setHtml(html_content)
                if hasattr(self, 'interactive_plot_dock') and not self.interactive_plot_dock.isVisible():
                     self.interactive_plot_dock.show()
                     self.interactive_plot_dock.raise_()
            else:
                 self.add_log_message(f"No specific interactive plot generated for type: {result_type}.", "Info")
                 self.web_view.setHtml(f"<p style='color:{text_color};'>No interactive plot available for type: {result_type}.</p>")
                 if hasattr(self, 'interactive_plot_dock') and not self.interactive_plot_dock.isVisible():
                     self.interactive_plot_dock.show()
                     self.interactive_plot_dock.raise_()

        except Exception as e:
            self.add_log_message(f"Interactive plot error: {e}", "Error")
            import traceback
            self.add_log_message(traceback.format_exc(), "Debug")
            if self.web_view and isinstance(self.web_view, QWebEngineView):
                self.web_view.setHtml(f"<p style='color:red;'>Plotly error: {e}</p>")


    def export_current_plot(self):
        if not hasattr(self.plot_canvas, 'fig') or not self.plot_canvas.fig.axes: # Check if figure has axes
            QMessageBox.warning(self, "Export Plot", "No plot to export. Please run a simulation first.")
            self.add_log_message("No plot to export.", "Warning")
            return

        default_filename = f"plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath, _ = QFileDialog.getSaveFileName(self, "Export Current Matplotlib Plot", default_filename,
                                                  "PNG files (*.png);;JPEG files (*.jpg);;PDF files (*.pdf);;SVG files (*.svg);;All Files (*)")
        if filepath:
            try:
                self.plot_canvas.fig.savefig(filepath, facecolor=self.plot_canvas.fig.get_facecolor()) # Save with current bg
                self.add_log_message(f"Plot saved to {filepath}", "Info")
                QMessageBox.information(self, "Plot Exported", f"Plot successfully saved to:\n{filepath}")
            except Exception as e:
                self.add_log_message(f"Error saving plot: {e}", "Error")
                QMessageBox.critical(self, "Export Error", f"Could not save plot: {e}")

    def export_results(self):
        if not self.current_results:
            QMessageBox.warning(self, "Export Results", "No results to export. Please run a simulation first.")
            self.add_log_message("No results to export.", "Warning")
            return

        default_filename = f"results_{self.current_results.get('type', 'data')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath, _ = QFileDialog.getSaveFileName(self, "Export Results as JSON", default_filename,
                                                  "JSON files (*.json);;All Files (*)")
        if filepath:
            try:
                def convert_numpy_for_json(obj):
                    if isinstance(obj, np.ndarray): return obj.tolist()
                    if isinstance(obj, (np.generic, np.number)): return obj.item() # Convert numpy numbers to Python native
                    if isinstance(obj, datetime): return obj.isoformat()
                    # Add other non-serializable types if necessary
                    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

                with open(filepath, 'w') as f:
                    json.dump(self.current_results, f, indent=2, default=convert_numpy_for_json)
                self.add_log_message(f"Results saved to {filepath}", "Info")
                QMessageBox.information(self, "Results Exported", f"Results successfully saved to:\n{filepath}")
            except Exception as e:
                self.add_log_message(f"Error saving results: {e}", "Error")
                QMessageBox.critical(self, "Export Error", f"Could not save results: {e}")

    def add_log_message(self, message, level="Info"):
        timestamp = datetime.now().strftime('%H:%M:%S')
        formatted_message = f"[{timestamp}][{level.upper()}] {message}"
        self.log_text.append(formatted_message)
        # Auto-scroll to the bottom
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
        if level.upper() == "ERROR":
            print(f"ERROR: {message}", file=sys.stderr) # Also print errors to console

        elif level.upper() == "WARNING":
            print(f"WARNING: {message}", file=sys.stderr)


    def add_log_message_from_worker(self, message): # Slot for worker's log_updated signal
        # Assuming worker message might not have level, add it or parse if worker includes it
        self.add_log_message(message, "Worker")


    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def update_stage(self, stage_text):
        self.stage_label.setText(f"Stage: {stage_text}")
        self.current_stage_label_monitor.setText(f"Current Stage: {stage_text}") # Update monitor too
        self.add_log_message(f"Simulation stage: {stage_text}", "Debug")


    def handle_simulation_error(self, error_message):
        self.add_log_message(f"Simulation Error: {error_message}", "Error")
        QMessageBox.critical(self, "Simulation Error", str(error_message))
        self.run_button.setEnabled(True); self.pause_btn.setEnabled(False); self.stop_btn.setEnabled(False)
        self.sim_type.setEnabled(True) # Re-enable
        self.progress_bar.setVisible(False); self.stage_label.setVisible(False)
        self.worker = None # Clear worker

    def handle_intermediate_result(self, data):
        # Display in monitor panel's text edit
        if hasattr(self, 'intermediate_results_display') and self.intermediate_results_display is not None:
            self.intermediate_results_display.append(json.dumps(data, indent=1, default=str))  # type: ignore[attr-defined]
            scrollbar = self.intermediate_results_display.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())  # type: ignore[attr-defined]

        # Potentially update plot if data type is suitable for live plotting (e.g. a growing line)
        # For now, just log it
        self.add_log_message(f"Intermediate result: {data.get('type', 'Unknown type')}", "Debug")


    def update_time(self):
        current_time = datetime.now().strftime('%H:%M:%S')
        self.time_status.setText(f"Time: {current_time}")

    def update_performance_display(self, perf_data): # For direct updates from worker
        if 'cpu' in perf_data: self.cpu_label.setText(f"CPU: {perf_data['cpu']:.1f}%")
        if 'memory_percent' in perf_data: self.memory_label.setText(f"Mem (Proc): {perf_data['memory_percent']:.1f}%")
        if 'elapsed_time' in perf_data: self.sim_time_label.setText(f"Elapsed Sim Time: {perf_data['elapsed_time']:.2f}s")

    def update_performance_display_periodic(self): # For general system status
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory()
        # self.cpu_label.setText(f"CPU (Sys): {cpu:.1f}%") # Can distinguish if needed
        self.memory_status.setText(f"Mem (Sys): {mem.percent:.1f}% ({mem.used/1024**3:.1f}/{mem.total/1024**3:.1f}GB)")
        # If worker is running, it might update its own specific CPU/Mem via performance_updated signal

    def create_themes(self):
        themes = {}
        # Light Theme (Default Qt Fusion often looks like this)
        light_palette = QPalette() # Start with default
        themes['light'] = light_palette

        # Dark Theme (Nord-inspired)
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(46, 52, 64)) # nord0 - Background
        dark_palette.setColor(QPalette.WindowText, QColor(216, 222, 233)) # nord4 - Foreground
        dark_palette.setColor(QPalette.Base, QColor(59, 66, 82)) # nord1 - Slightly Lighter Background (e.g. text edits)
        dark_palette.setColor(QPalette.AlternateBase, QColor(76, 86, 106)) # nord3 - Selection Background
        dark_palette.setColor(QPalette.ToolTipBase, QColor(46, 52, 64))
        dark_palette.setColor(QPalette.ToolTipText, QColor(216, 222, 233))
        dark_palette.setColor(QPalette.Text, QColor(229, 233, 240)) # nord4 (brighter variant for text)
        dark_palette.setColor(QPalette.Button, QColor(59, 66, 82)) # nord1
        dark_palette.setColor(QPalette.ButtonText, QColor(216, 222, 233)) # nord4
        dark_palette.setColor(QPalette.BrightText, QColor(236, 239, 244)) # nord6 - Bright text (e.g. highlighted)
        dark_palette.setColor(QPalette.Link, QColor(136, 192, 208)) # nord9 - Blue
        dark_palette.setColor(QPalette.Highlight, QColor(136, 192, 208)) # nord9 - For selections
        dark_palette.setColor(QPalette.HighlightedText, QColor(46, 52, 64)) # nord0 - Text on selection
        # Disabled states
        dark_palette.setColor(QPalette.Disabled, QPalette.Text, QColor(128, 128, 128))
        dark_palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(128, 128, 128))
        dark_palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(128,128,128))
        themes['dark'] = dark_palette

        # Plasma Theme (for Plotly primarily, but can set a QPalette too)
        # This QPalette is more for the app frame if 'plasma' is chosen
        plasma_palette = QPalette()
        plasma_palette.setColor(QPalette.Window, QColor(13, 17, 23)) # Very dark blue/black
        plasma_palette.setColor(QPalette.WindowText, QColor(240, 240, 240))
        plasma_palette.setColor(QPalette.Base, QColor(20, 30, 40))
        plasma_palette.setColor(QPalette.AlternateBase, QColor(30, 40, 50))
        plasma_palette.setColor(QPalette.ToolTipBase, QColor(13, 17, 23))
        plasma_palette.setColor(QPalette.ToolTipText, QColor(240, 240, 240))
        plasma_palette.setColor(QPalette.Text, QColor(220, 220, 220))
        plasma_palette.setColor(QPalette.Button, QColor(30, 40, 50))
        plasma_palette.setColor(QPalette.ButtonText, QColor(220, 220, 220))
        plasma_palette.setColor(QPalette.BrightText, QColor(255, 0, 0)) # Bright red
        plasma_palette.setColor(QPalette.Link, QColor(0, 122, 204))
        plasma_palette.setColor(QPalette.Highlight, QColor(0, 122, 204)) # Blue highlight
        plasma_palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255)) # White
        themes['plasma'] = plasma_palette

        return themes

    def quick_spacetime_sim(self):
        self.sim_type.setCurrentText("🌌 Spacetime Emergence")
        # Optionally set some default parameters for quick sim
        self.n_subsystems.setValue(30)
        self.evolution_steps.setValue(50)
        self.run_enhanced_simulation()
        self.add_log_message("Quick Spacetime Simulation triggered.", "Action")


    def quick_blackhole_sim(self):
        self.sim_type.setCurrentText("🕳️ Black Hole Physics")
        self.mass_solar.setValue(np.random.uniform(1, 20)) # Random mass for fun
        self.run_enhanced_simulation()
        self.add_log_message("Quick Black Hole Simulation triggered.", "Action")

    def pause_simulation(self):
        if self.worker and self.worker.isRunning():
            if not self.worker.is_paused:
                self.worker.pause()
                self.pause_btn.setText("▶️ Resume")
                self.pause_btn.setIcon(QIcon.fromTheme("media-playback-start", QIcon("icons/resume.png"))) # Or play icon
                self.add_log_message("Simulation pause requested.", "Action")
            else:
                self.worker.resume()
                self.pause_btn.setText("⏸️ Pause")
                self.pause_btn.setIcon(QIcon.fromTheme("media-playback-pause", QIcon("icons/pause.png")))
                self.add_log_message("Simulation resume requested.", "Action")
        else:
            self.add_log_message("No simulation running to pause/resume.", "Warning")


    def stop_simulation(self):
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(self, 'Stop Simulation',
                                       "Are you sure you want to stop the current simulation?",
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.worker.stop()
                self.add_log_message("Simulation stop requested by user.", "Action")
                # UI updates (enabled buttons etc.) will be handled in handle_enhanced_results or error handler
                # after worker thread actually finishes.
                # For now, just re-enable run button to allow new sim
                self.run_button.setEnabled(True) # Prematurely enable run, worker will confirm
                self.pause_btn.setEnabled(False); self.pause_btn.setText("⏸️ Pause")
                self.stop_btn.setEnabled(False)
                self.sim_type.setEnabled(True)
            else:
                self.add_log_message("Simulation stop cancelled by user.", "Info")
        else:
            self.add_log_message("No simulation running to stop.", "Warning")


    def clear_log(self):
        self.log_text.clear()
        self.intermediate_results_display.clear() # Also clear intermediate results
        self.add_log_message("Log cleared.", "System")

    def new_project(self): self.add_log_message("Action: New Project (Not Implemented)", "Placeholder")
    def open_project(self): self.add_log_message("Action: Open Project (Not Implemented)", "Placeholder")
    def save_project(self): self.add_log_message("Action: Save Project (Not Implemented)", "Placeholder")

    def run_benchmark(self): self.add_log_message("Action: Run Benchmark (Not Implemented)", "Placeholder")
    def open_optimizer(self): self.add_log_message("Action: Open Optimizer (Not Implemented)", "Placeholder")
    def set_research_mode(self, checked): self.add_log_message(f"Action: Research Mode set to {checked} (Not Implemented)", "Placeholder")

    def show_tutorial(self):
        self.add_log_message("Action: Show Tutorial (Not Implemented)", "Placeholder")
        QDesktopServices.openUrl(QUrl("https://example.com/egqgem-tutorial")) # Example URL

    def show_documentation(self):
        self.add_log_message("Action: Show Documentation (Not Implemented)", "Placeholder")
        QDesktopServices.openUrl(QUrl("https://example.com/egqgem-docs")) # Example URL

    def show_enhanced_about(self):
        about_text = """
        <b>EG-QGEM Advanced Simulation Interface</b><br><br>
        Version: 0.2.0 (Enhanced GUI)<br>
        Copyright © 2024 Your Name/Organization<br><br>
        This application provides an advanced interface for exploring
        Emergent Gravity and Quantum Geometry Entanglement Mechanics (EG-QGEM)
        simulations. Features include various simulation types, 3D visualizations,
        parameter sweeping, batch processing, and data analysis tools.<br><br>
        Built with Python, PyQt5, Matplotlib, and Plotly.
        """
        QMessageBox.about(self, "About EG-QGEM Interface", about_text)
        self.add_log_message("Displayed About dialog.", "Info")

    def toggle_fullscreen_plot(self, checked):
        if checked:
            # Store current dock states and hide them
            self._original_dock_states = {}
            for dock_name in ['control_dock', 'log_dock', 'monitor_dock', 'analysis_dock', 'sweep_dock', 'batch_dock', 'interactive_plot_dock']:
                if hasattr(self, dock_name):
                    dock = getattr(self, dock_name)
                    self._original_dock_states[dock_name] = not dock.isHidden()
                    dock.hide()

            # Store central widget and replace it with plot_canvas
            self._original_central_widget = self.centralWidget()
            if self._original_central_widget: # Detach from main window
                 self._original_central_widget.setParent(None)

            self.setCentralWidget(self.plot_canvas)
            self.plot_canvas.setFocus() # Give focus to plot
            self.add_log_message("Plot fullscreen enabled.", "Interface")
        else:
            # Restore central widget
            if hasattr(self, '_original_central_widget') and self._original_central_widget:
                current_central = self.centralWidget()
                if current_central is self.plot_canvas:  # type: ignore[union-attr]
                    # Detach plot_canvas safely
                    self.plot_canvas.setParent(None)  # type: ignore[attr-defined]
                # Restore original central widget
                self.setCentralWidget(self._original_central_widget)
                # Re-insert plot_canvas into splitter
                try:
                    self.h_splitter.insertWidget(0, self.plot_canvas)
                    if hasattr(self, '_original_splitter_sizes'):
                        self.h_splitter.setSizes(self._original_splitter_sizes)
                except Exception:
                    pass
            # Restore docks
            if hasattr(self, '_original_dock_states'):
                for dock_name, was_visible in self._original_dock_states.items():
                    if hasattr(self, dock_name):
                        dock = getattr(self, dock_name)
                        if was_visible: dock.show()
                        else: dock.hide()
                del self._original_dock_states # Clean up

            self.add_log_message("Plot fullscreen disabled.", "Interface")
        self.fullscreen_action.setChecked(checked) # Sync check state


    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(self, 'Confirm Exit',
                                       "A simulation is currently running. Are you sure you want to exit? The simulation will be stopped.",
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.worker.stop() # Request stop
                # self.worker.wait(5000) # Wait a bit for it to finish, optional
                event.accept()
                self.add_log_message("Exiting application during simulation.", "System")
            else:
                event.ignore()
                self.add_log_message("Exit cancelled by user.", "Info")
        else:
            self.add_log_message("Exiting application.", "System")
            event.accept()
