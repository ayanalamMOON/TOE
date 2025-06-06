"""
Entanglement Tensor Implementation
=================================

This module implements the core entanglement tensor E_μν that describes
the local density and flux of quantum correlations in spacetime.
"""

import numpy as np
from scipy import linalg
import sympy as sp
from .constants import CONSTANTS

class EntanglementTensor:
    """
    The entanglement tensor E_μν encodes quantum correlations that source spacetime curvature.

    In EG-QGEM theory, this tensor represents:
    - Local entanglement density
    - Directional flow of quantum correlations
    - Source term for modified Einstein equations
    """

    def __init__(self, dimensions=4):
        """
        Initialize entanglement tensor.

        Args:
            dimensions (int): Spacetime dimensions (default: 4)
        """
        self.dim = dimensions
        self.tensor = np.zeros((dimensions, dimensions))
        self.coordinates = ['t', 'x', 'y', 'z'][:dimensions]

    def set_from_entanglement_entropy(self, entropy_density, flow_vector=None):
        """
        Construct E_μν from entanglement entropy density and flow.

        Args:
            entropy_density (float): Local entanglement entropy density
            flow_vector (array): Direction of entanglement flow
        """
        if flow_vector is None:
            flow_vector = np.zeros(self.dim)
            flow_vector[0] = 1.0  # Default timelike flow

        # Normalize flow vector
        flow_norm = np.linalg.norm(flow_vector)
        if flow_norm > 0:
            flow_vector = flow_vector / flow_norm

        # Construct tensor: E_μν = ρ_E * u_μ * u_ν + P_E * h_μν
        # where u_μ is flow vector and h_μν is spatial projection
        self.tensor = entropy_density * np.outer(flow_vector, flow_vector)

        # Add pressure term (simplified isotropic case)
        pressure = entropy_density / 3.0  # Radiation-like equation of state
        for i in range(1, self.dim):  # Spatial components only
            self.tensor[i, i] += pressure

    def set_from_quantum_state(self, rho_matrix, subsystem_partition):
        """
        Calculate E_μν from quantum density matrix.

        Args:
            rho_matrix (ndarray): Quantum density matrix
            subsystem_partition (tuple): How to partition system for entanglement
        """
        # Calculate reduced density matrix for subsystem A
        dim_A, dim_B = subsystem_partition
        rho_A = self._partial_trace(rho_matrix, dim_A, dim_B)

        # Compute von Neumann entropy
        eigenvals = linalg.eigvals(rho_A)
        eigenvals = eigenvals[eigenvals > 1e-12]  # Remove near-zero eigenvals
        entropy = -np.sum(eigenvals * np.log(eigenvals))

        # Map entropy to spacetime tensor (simplified mapping)
        self.set_from_entanglement_entropy(entropy * CONSTANTS.rho_E_crit)

    def _partial_trace(self, rho, dim_A, dim_B):
        """Compute partial trace over subsystem B."""
        rho_reshaped = rho.reshape(dim_A, dim_B, dim_A, dim_B)
        return np.trace(rho_reshaped, axis1=1, axis2=3)

    def trace(self):
        """Return trace of entanglement tensor."""
        return np.trace(self.tensor)

    def stress_energy_tensor(self):
        """
        Compute the entanglement stress-energy tensor T^(E)_μν.

        Returns:
            ndarray: Stress-energy contribution from entanglement
        """
        trace = self.trace()
        T_E = CONSTANTS.kappa_E * (self.tensor - 0.5 * trace * np.eye(self.dim))
        return T_E

    def divergence(self, metric, christoffel_symbols):
        """
        Compute covariant divergence ∇_μ E^μν.

        Args:
            metric (ndarray): Spacetime metric tensor g_μν
            christoffel_symbols (ndarray): Connection coefficients Γ^α_μν

        Returns:
            ndarray: Divergence vector
        """
        # Raise indices: E^μν = g^μα g^νβ E_αβ
        g_inv = linalg.inv(metric)
        E_raised = np.einsum('ma,nb,ab->mn', g_inv, g_inv, self.tensor)

        # Compute divergence (simplified - needs proper coordinate system)
        div = np.zeros(self.dim)
        for nu in range(self.dim):
            for mu in range(self.dim):
                for alpha in range(self.dim):
                    div[nu] += christoffel_symbols[alpha, mu, alpha] * E_raised[mu, nu]
                    div[nu] += christoffel_symbols[mu, alpha, nu] * E_raised[mu, alpha]

        return div

    def evolve_dynamics(self, dt, source_term=None):
        """
        Evolve entanglement tensor according to field equation:
        □E_μν = S_μν(matter)

        Args:
            dt (float): Time step
            source_term (ndarray): Matter source S_μν
        """
        if source_term is None:
            source_term = np.zeros_like(self.tensor)

        # Simplified evolution (proper implementation needs spacetime coordinates)
        # This is a placeholder for the wave equation solution
        self.tensor += dt * source_term

    def visualization_data(self):
        """Return data for visualization."""
        return {
            'tensor': self.tensor.copy(),
            'trace': self.trace(),
            'eigenvalues': linalg.eigvals(self.tensor),
            'determinant': linalg.det(self.tensor)
        }

    def __str__(self):
        """String representation of tensor."""
        return f"EntanglementTensor({self.dim}D):\n{self.tensor}"

    def __repr__(self):
        return f"EntanglementTensor(dimensions={self.dim})"

class EntanglementField:
    """
    Spacetime field of entanglement tensors E_μν(x).
    """

    def __init__(self, spacetime_grid):
        """
        Initialize field on spacetime grid.

        Args:
            spacetime_grid (tuple): Grid dimensions (nt, nx, ny, nz)
        """
        self.grid_shape = spacetime_grid
        self.field = np.zeros(spacetime_grid + (4, 4))  # E_μν at each point

    def set_initial_conditions(self, initial_func):
        """
        Set initial entanglement distribution.

        Args:
            initial_func (callable): Function(t,x,y,z) -> EntanglementTensor
        """
        nt, nx, ny, nz = self.grid_shape
        for it in range(nt):
            for ix in range(nx):
                for iy in range(ny):
                    for iz in range(nz):
                        coords = (it, ix, iy, iz)
                        tensor = initial_func(*coords)
                        self.field[it, ix, iy, iz] = tensor.tensor

    def get_tensor_at(self, coordinates):
        """Get entanglement tensor at specific coordinates."""
        t, x, y, z = coordinates
        tensor = EntanglementTensor()
        tensor.tensor = self.field[t, x, y, z].copy()
        return tensor

    def total_entanglement(self):
        """Compute total entanglement in spacetime volume."""
        return np.sum(np.trace(self.field, axis1=-2, axis2=-1))

    def gradient(self, direction):
        """Compute spatial gradient of entanglement field."""
        return np.gradient(self.field, axis=direction+1)  # +1 for time axis
