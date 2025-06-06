"""
Numerical Solvers for EG-QGEM Field Equations
============================================

Advanced numerical methods for solving the modified Einstein field equations
with entanglement contributions.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, eigsh
from scipy.integrate import solve_ivp, quad
from scipy.optimize import minimize, root
import warnings

class EGQGEMFieldSolver:
    """
    Advanced numerical solver for EG-QGEM field equations.
    """

    def __init__(self, grid_size=64, domain_size=10.0):
        """
        Initialize the field solver.

        Parameters:
        -----------
        grid_size : int
            Number of grid points in each spatial dimension
        domain_size : float
            Physical size of the computational domain
        """
        self.grid_size = grid_size
        self.domain_size = domain_size
        self.dx = domain_size / grid_size
        self.dt = 0.001  # Default time step

        # Create coordinate grids
        self.x = np.linspace(-domain_size/2, domain_size/2, grid_size)
        self.y = np.linspace(-domain_size/2, domain_size/2, grid_size)
        self.z = np.linspace(-domain_size/2, domain_size/2, grid_size)

        # Initialize fields
        self.metric = np.zeros((grid_size, grid_size, grid_size, 4, 4))
        self.entanglement_density = np.zeros((grid_size, grid_size, grid_size))

        # Set up differential operators
        self._setup_operators()

    def _setup_operators(self):
        """Set up finite difference operators."""
        n = self.grid_size

        # Second derivative operator (1D)
        self.d2_op = sp.diags(
            [1, -2, 1],
            [-1, 0, 1],
            shape=(n, n),
            format='csr'
        ) / (self.dx**2)

        # First derivative operator (1D)
        self.d1_op = sp.diags(
            [-1, 0, 1],
            [-1, 0, 1],
            shape=(n, n),
            format='csr'
        ) / (2 * self.dx)

        # 3D Laplacian operator
        I = sp.eye(n, format='csr')
        self.laplacian_3d = (
            sp.kron(sp.kron(I, I), self.d2_op) +
            sp.kron(sp.kron(I, self.d2_op), I) +
            sp.kron(sp.kron(self.d2_op, I), I)
        )

    def initialize_fields(self, initial_conditions):
        """
        Initialize metric and entanglement fields.

        Parameters:
        -----------
        initial_conditions : dict
            Dictionary containing initial field configurations
        """
        if 'metric' in initial_conditions:
            self.metric = initial_conditions['metric'].copy()
        else:
            # Default to Minkowski metric
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    for k in range(self.grid_size):
                        self.metric[i, j, k] = np.diag([-1, 1, 1, 1])

        if 'entanglement_density' in initial_conditions:
            self.entanglement_density = initial_conditions['entanglement_density'].copy()
        else:
            # Default entanglement density (small random fluctuations)
            self.entanglement_density = 1e-10 * np.random.randn(
                self.grid_size, self.grid_size, self.grid_size
            )

    def calculate_christoffel_symbols(self, metric):
        """
        Calculate Christoffel symbols from metric tensor.

        Parameters:
        -----------
        metric : ndarray
            4x4 metric tensor

        Returns:
        --------
        christoffel : ndarray
            Christoffel symbols Γ^μ_νρ
        """
        # Compute metric inverse
        try:
            metric_inv = np.linalg.inv(metric)
        except np.linalg.LinAlgError:
            # Handle singular matrices
            metric_inv = np.linalg.pinv(metric)

        christoffel = np.zeros((4, 4, 4))

        # Calculate derivatives of metric (finite differences)
        # This is a simplified version - full implementation would use
        # proper finite difference stencils on the grid
        for mu in range(4):
            for nu in range(4):
                for rho in range(4):
                    for sigma in range(4):
                        # Γ^μ_νρ = (1/2) g^μσ (∂_ν g_σρ + ∂_ρ g_σν - ∂_σ g_νρ)
                        # This is a placeholder - actual implementation needs
                        # proper spatial derivatives
                        christoffel[mu, nu, rho] += 0.5 * metric_inv[mu, sigma] * (
                            0  # Would be actual partial derivatives
                        )

        return christoffel

    def calculate_ricci_tensor(self, christoffel):
        """
        Calculate Ricci tensor from Christoffel symbols.

        Parameters:
        -----------
        christoffel : ndarray
            Christoffel symbols

        Returns:
        --------
        ricci : ndarray
            Ricci tensor R_μν
        """
        ricci = np.zeros((4, 4))

        for mu in range(4):
            for nu in range(4):
                for alpha in range(4):
                    # R_μν = ∂_α Γ^α_μν - ∂_ν Γ^α_μα + Γ^α_αβ Γ^β_μν - Γ^α_νβ Γ^β_μα
                    # Simplified calculation - full version needs proper derivatives
                    ricci[mu, nu] += 0  # Placeholder

        return ricci

    def entanglement_stress_energy(self, position):
        """
        Calculate entanglement contribution to stress-energy tensor.

        Parameters:
        -----------
        position : array-like
            Spatial position (i, j, k) indices

        Returns:
        --------
        T_E : ndarray
            Entanglement stress-energy tensor
        """
        i, j, k = position

        # Get local entanglement density
        rho_E = self.entanglement_density[i, j, k]

        # Calculate entanglement pressure and energy density
        # Based on EG-QGEM theory equations
        kappa_E = 1e-42  # Entanglement coupling constant

        energy_density = kappa_E * rho_E**2
        pressure = kappa_E * rho_E**2 / 3  # Equation of state

        # Construct stress-energy tensor
        T_E = np.zeros((4, 4))
        T_E[0, 0] = -energy_density  # Energy density (with - for signature)
        T_E[1, 1] = pressure         # Pressure
        T_E[2, 2] = pressure
        T_E[3, 3] = pressure

        return T_E

    def field_equations_residual(self, metric_flat, position):
        """
        Calculate residual of modified Einstein field equations.

        Parameters:
        -----------
        metric_flat : ndarray
            Flattened metric tensor components
        position : tuple
            Grid position (i, j, k)

        Returns:
        --------
        residual : ndarray
            Field equation residual
        """
        # Reshape metric
        metric = metric_flat.reshape(4, 4)

        # Calculate geometric quantities
        christoffel = self.calculate_christoffel_symbols(metric)
        ricci = self.calculate_ricci_tensor(christoffel)
        ricci_scalar = np.trace(np.linalg.solve(metric, ricci))

        # Einstein tensor
        einstein = ricci - 0.5 * ricci_scalar * metric

        # Entanglement stress-energy
        T_E = self.entanglement_stress_energy(position)

        # Modified field equations: G_μν + Λ g_μν = 8πG T_E_μν
        G = 6.67430e-11  # Gravitational constant
        Lambda = 1e-52   # Cosmological constant

        lhs = einstein + Lambda * metric
        rhs = 8 * np.pi * G * T_E

        residual = lhs - rhs
        return residual.flatten()

    def solve_field_equations_point(self, position, initial_metric=None):
        """
        Solve field equations at a single grid point.

        Parameters:
        -----------
        position : tuple
            Grid position (i, j, k)
        initial_metric : ndarray, optional
            Initial guess for metric

        Returns:
        --------
        metric : ndarray
            Solution metric tensor
        success : bool
            Whether solution converged
        """
        if initial_metric is None:
            initial_metric = np.diag([-1, 1, 1, 1])

        # Flatten for optimization
        x0 = initial_metric.flatten()

        # Solve nonlinear system
        try:
            sol = root(
                self.field_equations_residual,
                x0,
                args=(position,),
                method='hybr',
                options={'xtol': 1e-10}
            )

            if sol.success:
                metric = sol.x.reshape(4, 4)
                # Ensure metric is symmetric
                metric = 0.5 * (metric + metric.T)
                return metric, True
            else:
                return initial_metric, False

        except Exception as e:
            warnings.warn(f"Field equation solver failed at {position}: {e}")
            return initial_metric, False

    def solve_full_field_equations(self, max_iterations=100, tolerance=1e-8):
        """
        Solve field equations on the entire grid using iterative method.

        Parameters:
        -----------
        max_iterations : int
            Maximum number of iterations
        tolerance : float
            Convergence tolerance

        Returns:
        --------
        converged : bool
            Whether the solution converged
        iterations : int
            Number of iterations performed
        """
        print("Solving EG-QGEM field equations...")

        for iteration in range(max_iterations):
            old_metric = self.metric.copy()
            total_change = 0.0
            successful_points = 0

            # Solve at each grid point
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    for k in range(self.grid_size):
                        new_metric, success = self.solve_field_equations_point(
                            (i, j, k),
                            self.metric[i, j, k]
                        )

                        if success:
                            self.metric[i, j, k] = new_metric
                            successful_points += 1

                        # Calculate change
                        change = np.linalg.norm(
                            self.metric[i, j, k] - old_metric[i, j, k]
                        )
                        total_change += change

            # Check convergence
            avg_change = total_change / (self.grid_size**3)
            success_rate = successful_points / (self.grid_size**3)

            print(f"Iteration {iteration + 1}: avg_change = {avg_change:.2e}, "
                  f"success_rate = {success_rate:.1%}")

            if avg_change < tolerance and success_rate > 0.95:
                print(f"Converged after {iteration + 1} iterations")
                return True, iteration + 1

        print(f"Did not converge after {max_iterations} iterations")
        return False, max_iterations

    def evolve_entanglement_density(self, dt=None):
        """
        Evolve entanglement density field according to EG-QGEM dynamics.

        Parameters:
        -----------
        dt : float, optional
            Time step (uses self.dt if not provided)
        """
        if dt is None:
            dt = self.dt

        # Entanglement evolution equation (simplified)
        # ∂ρ_E/∂t = -∇²ρ_E + α ρ_E (1 - ρ_E/ρ_max) - β R ρ_E

        # Parameters
        alpha = 1e-6  # Growth rate
        beta = 1e-8   # Curvature coupling
        rho_max = 1e-6  # Maximum entanglement density

        # Calculate Laplacian of entanglement density
        rho_flat = self.entanglement_density.flatten()
        laplacian_rho = self.laplacian_3d.dot(rho_flat)
        laplacian_rho = laplacian_rho.reshape(self.entanglement_density.shape)

        # Calculate local curvature (simplified)
        curvature = np.zeros_like(self.entanglement_density)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                for k in range(self.grid_size):
                    metric = self.metric[i, j, k]
                    # Simplified curvature estimate
                    curvature[i, j, k] = np.trace(metric) - (-1 + 3)  # Deviation from flat

        # Evolution equation
        drho_dt = (
            -laplacian_rho +
            alpha * self.entanglement_density * (1 - self.entanglement_density / rho_max) -
            beta * curvature * self.entanglement_density
        )

        # Update entanglement density
        self.entanglement_density += dt * drho_dt

        # Ensure positivity
        self.entanglement_density = np.maximum(self.entanglement_density, 0)

    def get_solution_summary(self):
        """
        Get summary statistics of the current solution.

        Returns:
        --------
        summary : dict
            Summary statistics
        """
        # Calculate various diagnostics
        metric_determinants = []
        curvature_scalars = []
        entanglement_stats = {}

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                for k in range(self.grid_size):
                    metric = self.metric[i, j, k]

                    # Metric determinant
                    det = np.linalg.det(metric)
                    metric_determinants.append(det)

                    # Approximate curvature scalar
                    R = np.trace(metric) - (-1 + 3)  # Simplified
                    curvature_scalars.append(R)

        # Entanglement statistics
        entanglement_stats = {
            'mean': np.mean(self.entanglement_density),
            'std': np.std(self.entanglement_density),
            'min': np.min(self.entanglement_density),
            'max': np.max(self.entanglement_density),
            'total': np.sum(self.entanglement_density) * (self.dx**3)
        }

        summary = {
            'metric_determinants': {
                'mean': np.mean(metric_determinants),
                'std': np.std(metric_determinants),
                'min': np.min(metric_determinants),
                'max': np.max(metric_determinants)
            },
            'curvature_scalars': {
                'mean': np.mean(curvature_scalars),
                'std': np.std(curvature_scalars),
                'min': np.min(curvature_scalars),
                'max': np.max(curvature_scalars)
            },
            'entanglement_density': entanglement_stats,
            'grid_info': {
                'size': self.grid_size,
                'domain_size': self.domain_size,
                'resolution': self.dx
            }
        }

        return summary


def solve_egqgem_cosmology(initial_conditions, time_span, method='RK45'):
    """
    Solve EG-QGEM cosmological evolution equations.

    Parameters:
    -----------
    initial_conditions : dict
        Initial values for cosmological parameters
    time_span : tuple
        (t_start, t_end) time integration range
    method : str
        Integration method for solve_ivp

    Returns:
    --------
    solution : dict
        Time evolution of cosmological parameters
    """
    def cosmology_equations(t, y):
        """
        EG-QGEM cosmological evolution equations.

        y = [a, H, rho_m, rho_E] where:
        a = scale factor
        H = Hubble parameter
        rho_m = matter density
        rho_E = entanglement density
        """
        a, H, rho_m, rho_E = y

        # Constants
        G = 6.67430e-11
        c = 299792458
        kappa_E = 1e-42  # Entanglement coupling

        # Friedmann equations with entanglement
        # H² = (8πG/3)(ρ_m + ρ_E) + Λ/3
        # ä/a = -(4πG/3)(ρ_m + ρ_E + 3p_E/c²) + Λ/3

        # Entanglement equation of state
        p_E = kappa_E * rho_E**2 / 3  # Entanglement pressure

        # Evolution equations
        da_dt = a * H

        # Acceleration equation
        Lambda = 1e-52
        a_ddot_over_a = -(4 * np.pi * G / 3) * (rho_m + rho_E + 3 * p_E / c**2) + Lambda / 3
        dH_dt = a_ddot_over_a - H**2

        # Matter conservation
        drho_m_dt = -3 * H * rho_m

        # Entanglement evolution (including creation/destruction)
        gamma_E = 1e-10  # Entanglement production rate
        drho_E_dt = -3 * H * (rho_E + p_E / c**2) + gamma_E * rho_m * rho_E

        return [da_dt, dH_dt, drho_m_dt, drho_E_dt]

    # Extract initial conditions
    y0 = [
        initial_conditions['scale_factor'],
        initial_conditions['hubble_parameter'],
        initial_conditions['matter_density'],
        initial_conditions['entanglement_density']
    ]

    # Solve ODEs
    sol = solve_ivp(
        cosmology_equations,
        time_span,
        y0,
        method=method,
        dense_output=True,
        rtol=1e-8,
        atol=1e-12
    )

    if not sol.success:
        raise RuntimeError(f"Cosmology integration failed: {sol.message}")

    # Package results
    solution = {
        'time': sol.t,
        'scale_factor': sol.y[0],
        'hubble_parameter': sol.y[1],
        'matter_density': sol.y[2],
        'entanglement_density': sol.y[3],
        'solution_object': sol
    }

    return solution
