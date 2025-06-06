"""
Visualization Tools for EG-QGEM Research
=======================================

This module provides comprehensive visualization capabilities for
theoretical predictions, simulation results, and experimental data.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import seaborn as sns
from scipy.interpolate import griddata
import h5py

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SpacetimeVisualizer:
    """
    Visualizes emergent spacetime from entanglement networks.
    """

    def __init__(self):
        self.fig = None
        self.axes = None

    def plot_entanglement_network(self, positions, entanglement_matrix, curvature=None):
        """
        Create 3D visualization of entanglement network.

        Args:
            positions (ndarray): Node positions in 3D space
            entanglement_matrix (ndarray): Entanglement strengths between nodes
            curvature (ndarray): Local curvature values at each node
        """
        # Create networkx graph
        G = nx.Graph()
        n_nodes = len(positions)

        for i in range(n_nodes):
            G.add_node(i, pos=positions[i])

        # Add edges based on entanglement
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                if entanglement_matrix[i, j] > 0.1:  # Threshold for visualization
                    G.add_edge(i, j, weight=entanglement_matrix[i, j])

        # 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Node colors based on curvature
        if curvature is not None:
            node_colors = curvature
            cmap = 'coolwarm'
        else:
            node_colors = 'blue'
            cmap = None

        # Draw nodes
        scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                           c=node_colors, cmap=cmap, s=100, alpha=0.8)

        # Draw edges
        for edge in G.edges(data=True):
            i, j = edge[0], edge[1]
            weight = edge[2]['weight']

            # Edge thickness based on entanglement strength
            linewidth = max(0.5, 3 * weight)
            alpha = min(1.0, 2 * weight)

            ax.plot([positions[i, 0], positions[j, 0]],
                   [positions[i, 1], positions[j, 1]],
                   [positions[i, 2], positions[j, 2]],
                   'gray', linewidth=linewidth, alpha=alpha)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Entanglement Network in Emergent Spacetime')

        if curvature is not None:
            plt.colorbar(scatter, ax=ax, label='Local Curvature')

        plt.tight_layout()
        return fig

    def plot_metric_evolution(self, times, metric_components):
        """
        Plot evolution of metric tensor components.

        Args:
            times (array): Time points
            metric_components (dict): Dictionary of metric component arrays
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        components = ['g_tt', 'g_rr', 'g_θθ', 'g_φφ']
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

        for comp, pos in zip(components, positions):
            if comp in metric_components:
                ax = axes[pos]
                ax.plot(times, metric_components[comp])
                ax.set_xlabel('Time')
                ax.set_ylabel(comp)
                ax.set_title(f'Metric Component {comp}')
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def animate_spacetime_emergence(self, evolution_data, save_path=None):
        """
        Create animation of spacetime emergence.

        Args:
            evolution_data (list): Time series of spacetime states
            save_path (str): Path to save animation
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        def animate(frame):
            axes[0].clear()
            axes[1].clear()

            data = evolution_data[frame]

            # Plot entanglement network
            positions = data['positions']
            entanglement = data['entanglement_matrix']

            # Create network plot
            G = nx.Graph()
            for i in range(len(positions)):
                G.add_node(i, pos=positions[i][:2])  # 2D projection

            pos_dict = dict(enumerate(positions[:, :2]))

            # Add significant edges
            for i in range(len(positions)):
                for j in range(i+1, len(positions)):
                    if entanglement[i, j] > 0.1:
                        G.add_edge(i, j, weight=entanglement[i, j])

            nx.draw(G, pos_dict, ax=axes[0], node_size=50,
                   edge_color='gray', node_color='blue', alpha=0.7)
            axes[0].set_title(f'Entanglement Network (t={frame})')

            # Plot metrics
            axes[1].plot(range(frame+1), [d['total_entanglement'] for d in evolution_data[:frame+1]])
            axes[1].set_xlabel('Time Step')
            axes[1].set_ylabel('Total Entanglement')
            axes[1].set_title('Entanglement Evolution')
            axes[1].grid(True, alpha=0.3)

        anim = animation.FuncAnimation(fig, animate, frames=len(evolution_data),
                                     interval=200, blit=False)

        if save_path:
            anim.save(save_path, writer='pillow', fps=5)

        return anim

class BlackHoleVisualizer:
    """
    Visualizes black hole structure and dynamics in EG-QGEM theory.
    """

    def plot_modified_spacetime(self, r_grid, metric_components, rs):
        """
        Plot black hole spacetime with entanglement modifications.

        Args:
            r_grid (array): Radial coordinate grid
            metric_components (dict): Metric tensor components
            rs (float): Schwarzschild radius
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Metric components
        axes[0, 0].semilogx(r_grid/rs, -metric_components['g_tt'],
                           label='g_tt (EG-QGEM)', linewidth=2)
        axes[0, 0].semilogx(r_grid/rs, np.maximum(1 - rs/r_grid, 0.01),
                           '--', label='g_tt (Classical)', alpha=0.7)
        axes[0, 0].axvline(1.0, color='red', linestyle=':', alpha=0.7, label='Horizon')
        axes[0, 0].set_xlabel('r/rs')
        axes[0, 0].set_ylabel('Metric components')
        axes[0, 0].set_title('Modified Metric Near Black Hole')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Entanglement density
        entanglement_density = np.exp(-r_grid/1e-35) / r_grid**2  # Simplified
        axes[0, 1].loglog(r_grid/rs, entanglement_density)
        axes[0, 1].axvline(1.0, color='red', linestyle=':', alpha=0.7)
        axes[0, 1].set_xlabel('r/rs')
        axes[0, 1].set_ylabel('Entanglement density')
        axes[0, 1].set_title('Entanglement Profile')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Potential well
        potential = -1/r_grid + entanglement_density * 1e35  # Simplified
        axes[1, 0].plot(r_grid/rs, potential/np.max(np.abs(potential)))
        axes[1, 0].axvline(1.0, color='red', linestyle=':', alpha=0.7)
        axes[1, 0].set_xlabel('r/rs')
        axes[1, 0].set_ylabel('Effective potential (normalized)')
        axes[1, 0].set_title('Gravitational + Entanglement Potential')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Curvature scalars
        ricci_scalar = 1/r_grid**6 * np.exp(-r_grid/1e-35)  # Simplified
        axes[1, 1].loglog(r_grid/rs, ricci_scalar)
        axes[1, 1].axvline(1.0, color='red', linestyle=':', alpha=0.7)
        axes[1, 1].set_xlabel('r/rs')
        axes[1, 1].set_ylabel('Ricci scalar')
        axes[1, 1].set_title('Spacetime Curvature')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_hawking_radiation(self, evolution_data):
        """
        Plot Hawking radiation with entanglement echoes.

        Args:
            evolution_data (dict): Radiation evolution data
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        times = evolution_data['times']

        # 1. Mass evolution
        axes[0, 0].plot(times/(365.25*24*3600), evolution_data['mass_evolution']/evolution_data['mass_evolution'][0])
        axes[0, 0].set_xlabel('Time (years)')
        axes[0, 0].set_ylabel('M(t)/M₀')
        axes[0, 0].set_title('Black Hole Mass Evolution')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Temperature evolution
        axes[0, 1].semilogy(times/(365.25*24*3600), evolution_data['temperature_evolution'])
        axes[0, 1].set_xlabel('Time (years)')
        axes[0, 1].set_ylabel('Temperature (K)')
        axes[0, 1].set_title('Hawking Temperature')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Entropy evolution
        axes[1, 0].plot(times/(365.25*24*3600), evolution_data['entropy_evolution'])
        axes[1, 0].set_xlabel('Time (years)')
        axes[1, 0].set_ylabel('Entropy')
        axes[1, 0].set_title('Black Hole Entropy')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Entanglement flux
        axes[1, 1].semilogy(times/(365.25*24*3600), evolution_data['entanglement_flux'])
        axes[1, 1].set_xlabel('Time (years)')
        axes[1, 1].set_ylabel('Entanglement flux')
        axes[1, 1].set_title('Information in Hawking Radiation')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

class ExperimentVisualizer:
    """
    Visualizes experimental predictions and comparisons.
    """

    def plot_quantum_gravity_experiment(self, prediction_data):
        """
        Plot quantum gravity experiment predictions.

        Args:
            prediction_data (dict): Experimental prediction data
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        times = prediction_data['times']

        # 1. Phase evolution
        axes[0, 0].plot(times*1000, prediction_data['phase_shifts'])
        axes[0, 0].set_xlabel('Time (ms)')
        axes[0, 0].set_ylabel('Phase shift (rad)')
        axes[0, 0].set_title('Gravitational Phase Shift')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Visibility evolution
        axes[0, 1].plot(times*1000, prediction_data['visibility_evolution'])
        axes[0, 1].set_xlabel('Time (ms)')
        axes[0, 1].set_ylabel('Interference visibility')
        axes[0, 1].set_title('Quantum Decoherence')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Entanglement rate
        rate = prediction_data['entanglement_rate']
        axes[1, 0].bar(['Entanglement Rate'], [rate])
        axes[1, 0].set_ylabel('Rate (Hz)')
        axes[1, 0].set_title('Gravity-Mediated Entanglement')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Parameter space
        masses = np.logspace(-15, -12, 50)
        rates = [rate * (m/1e-14)**2 for m in masses]
        axes[1, 1].loglog(masses*1e15, rates)
        axes[1, 1].set_xlabel('Mass (fg)')
        axes[1, 1].set_ylabel('Entanglement rate (Hz)')
        axes[1, 1].set_title('Mass Dependence')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_cosmological_predictions(self, cmb_data, rotation_data):
        """
        Plot cosmological predictions.

        Args:
            cmb_data (dict): CMB power spectrum data
            rotation_data (dict): Galaxy rotation curve data
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # 1. CMB power spectrum
        l_values = cmb_data['l_values']
        axes[0].loglog(l_values, l_values*(l_values+1)*cmb_data['C_l_standard']/(2*np.pi),
                      label='Standard ΛCDM', linewidth=2)
        axes[0].loglog(l_values, l_values*(l_values+1)*cmb_data['C_l_modified']/(2*np.pi),
                      label='EG-QGEM modified', linewidth=2)
        axes[0].set_xlabel('Multipole l')
        axes[0].set_ylabel('l(l+1)Cₗ/2π (μK²)')
        axes[0].set_title('CMB Temperature Power Spectrum')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 2. Galaxy rotation curves
        radii = rotation_data['radii']
        axes[1].plot(radii, rotation_data['v_newtonian']/1000,
                    '--', label='Newtonian', linewidth=2)
        axes[1].plot(radii, rotation_data['v_total']/1000,
                    label='With entanglement', linewidth=2)
        axes[1].set_xlabel('Radius (kpc)')
        axes[1].set_ylabel('Velocity (km/s)')
        axes[1].set_title('Galaxy Rotation Curve')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

class InteractiveVisualizer:
    """
    Creates interactive visualizations using Plotly.
    """

    def create_3d_spacetime(self, positions, entanglement_matrix, curvature):
        """
        Create interactive 3D spacetime visualization.

        Args:
            positions (ndarray): Node positions
            entanglement_matrix (ndarray): Entanglement connections
            curvature (ndarray): Local curvature values
        """
        # Nodes
        scatter = go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='markers',
            marker=dict(
                size=8,
                color=curvature,
                colorscale='Viridis',
                colorbar=dict(title="Local Curvature"),
                opacity=0.8
            ),
            text=[f'Node {i}<br>Curvature: {c:.3f}' for i, c in enumerate(curvature)],
            hovertemplate='<b>%{text}</b><br>Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'
        )

        # Edges
        edge_traces = []
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                if entanglement_matrix[i, j] > 0.1:
                    edge_trace = go.Scatter3d(
                        x=[positions[i, 0], positions[j, 0], None],
                        y=[positions[i, 1], positions[j, 1], None],
                        z=[positions[i, 2], positions[j, 2], None],
                        mode='lines',
                        line=dict(
                            color='rgba(125, 125, 125, 0.5)',
                            width=max(1, 5*entanglement_matrix[i, j])
                        ),
                        hoverinfo='none'
                    )
                    edge_traces.append(edge_trace)

        # Layout
        layout = go.Layout(
            title='Interactive Emergent Spacetime',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                bgcolor='rgba(0,0,0,0)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            showlegend=False
        )

        fig = go.Figure(data=[scatter] + edge_traces, layout=layout)
        return fig

    def create_parameter_explorer(self, parameter_ranges, compute_func):
        """
        Create interactive parameter exploration dashboard.

        Args:
            parameter_ranges (dict): Parameter names and ranges
            compute_func (callable): Function to compute results
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Entanglement Evolution', 'Curvature Distribution',
                          'Network Connectivity', 'Phase Diagram'),
            specs=[[{'type': 'scatter'}, {'type': 'histogram'}],
                   [{'type': 'scatter'}, {'type': 'heatmap'}]]
        )

        # Add traces (this would be expanded with actual interactive widgets)
        # For now, show example with default parameters
        results = compute_func(**{k: v[0] for k, v in parameter_ranges.items()})

        # Entanglement evolution
        fig.add_trace(
            go.Scatter(x=list(range(len(results['entanglement']))),
                      y=results['entanglement'],
                      name='Entanglement'),
            row=1, col=1
        )

        # Curvature histogram
        fig.add_trace(
            go.Histogram(x=results['curvature'], name='Curvature'),
            row=1, col=2
        )

        fig.update_layout(title='EG-QGEM Parameter Explorer')
        return fig

def create_research_dashboard(simulation_data, experimental_data):
    """
    Create comprehensive research dashboard.

    Args:
        simulation_data (dict): Results from simulations
        experimental_data (dict): Experimental predictions
    """
    # Initialize visualizers
    spacetime_viz = SpacetimeVisualizer()
    blackhole_viz = BlackHoleVisualizer()
    experiment_viz = ExperimentVisualizer()
    interactive_viz = InteractiveVisualizer()

    # Create figure grid
    fig = plt.figure(figsize=(20, 16))

    # Main spacetime emergence plot
    ax1 = plt.subplot2grid((4, 4), (0, 0), colspan=2, rowspan=2)
    if 'spacetime_emergence' in simulation_data:
        # Plot spacetime network
        pass  # Implementation would go here

    # Black hole structure
    ax2 = plt.subplot2grid((4, 4), (0, 2), colspan=2, rowspan=2)
    if 'black_hole' in simulation_data:
        # Plot black hole metrics
        pass  # Implementation would go here

    # Experimental predictions
    ax3 = plt.subplot2grid((4, 4), (2, 0), colspan=4, rowspan=2)
    if 'quantum_gravity' in experimental_data:
        # Plot experimental predictions
        pass  # Implementation would go here

    plt.tight_layout()
    return fig

def save_visualization_data(data, filename):
    """
    Save visualization data for later use.

    Args:
        data (dict): Visualization data
        filename (str): Output filename
    """
    with h5py.File(filename, 'w') as f:
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                f.create_dataset(key, data=value)
            elif isinstance(value, dict):
                group = f.create_group(key)
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, np.ndarray):
                        group.create_dataset(subkey, data=subvalue)

if __name__ == "__main__":
    # Example usage
    print("EG-QGEM Visualization Tools")
    print("Creating example visualizations...")

    # Generate sample data
    n_nodes = 50
    positions = np.random.randn(n_nodes, 3)
    entanglement_matrix = np.random.exponential(0.1, (n_nodes, n_nodes))
    entanglement_matrix = (entanglement_matrix + entanglement_matrix.T) / 2
    np.fill_diagonal(entanglement_matrix, 0)
    curvature = np.random.normal(0, 1, n_nodes)

    # Create visualizations
    spacetime_viz = SpacetimeVisualizer()
    fig1 = spacetime_viz.plot_entanglement_network(positions, entanglement_matrix, curvature)
    plt.show()

    # Interactive visualization
    interactive_viz = InteractiveVisualizer()
    fig2 = interactive_viz.create_3d_spacetime(positions, entanglement_matrix, curvature)
    fig2.show()

    print("Visualization examples complete!")
