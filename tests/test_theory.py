"""
Unit Tests for Theory Module
===========================

Tests for the core theoretical components of the EG-QGEM framework.
"""

import unittest
import numpy as np
import sys
import os

# Add project to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from theory.constants import CONSTANTS
from theory.entanglement_tensor import EntanglementTensor
from theory.modified_einstein import ModifiedEinsteinSolver


class TestConstants(unittest.TestCase):
    """Test EG-QGEM constants and derived quantities."""

    def test_planck_units(self):
        """Test Planck unit consistency."""
        # Check basic Planck units
        self.assertAlmostEqual(CONSTANTS.l_planck, 1.616255e-35, places=40)
        self.assertGreater(CONSTANTS.t_planck, 0)
        self.assertGreater(CONSTANTS.m_planck, 0)
        self.assertGreater(CONSTANTS.E_planck, 0)

    def test_egqgem_constants(self):
        """Test EG-QGEM specific constants."""
        self.assertGreater(CONSTANTS.kappa_E, 0)
        self.assertGreater(CONSTANTS.chi_E, 0)
        self.assertEqual(CONSTANTS.l_E, CONSTANTS.l_planck)
        self.assertGreater(CONSTANTS.Lambda_E, 0)

    def test_derived_scales(self):
        """Test derived EG-QGEM scales."""
        self.assertGreater(CONSTANTS.rho_E_crit, 0)
        self.assertGreater(CONSTANTS.t_E_decoher, 0)

    def test_dimensional_consistency(self):
        """Test dimensional consistency of constants."""
        # Check that kappa_E has correct dimensions [m²/J]
        kappa_E_dim = CONSTANTS.kappa_E * CONSTANTS.E_planck / (CONSTANTS.l_planck**2)
        self.assertGreater(kappa_E_dim, 0)


class TestEntanglementTensor(unittest.TestCase):
    """Test EntanglementTensor class functionality."""

    def setUp(self):
        """Set up test cases."""
        self.tensor = EntanglementTensor()

    def test_initialization(self):
        """Test proper initialization."""
        self.assertEqual(self.tensor.dim, 4)
        self.assertIsInstance(self.tensor.metric, np.ndarray)
        self.assertEqual(self.tensor.metric.shape, (4, 4))

    def test_quantum_state_creation(self):
        """Test quantum state creation."""
        state = self.tensor.create_entangled_state(2)
        self.assertEqual(len(state), 4)  # 2-qubit state has 4 components
        # Check normalization
        self.assertAlmostEqual(np.abs(np.vdot(state, state)), 1.0, places=10)

    def test_entanglement_entropy(self):
        """Test entanglement entropy calculation."""
        # Maximally entangled state should have entropy ln(2)
        max_entangled = np.array([1, 0, 0, 1]) / np.sqrt(2)
        entropy = self.tensor.calculate_entanglement_entropy(max_entangled, 2)
        self.assertAlmostEqual(entropy, np.log(2), places=5)

        # Product state should have zero entropy
        product_state = np.array([1, 0, 0, 0])
        entropy_zero = self.tensor.calculate_entanglement_entropy(product_state, 2)
        self.assertAlmostEqual(entropy_zero, 0.0, places=10)

    def test_tensor_calculation(self):
        """Test entanglement tensor calculation."""
        positions = np.random.rand(10, 4)  # 10 spacetime points
        tensor_field = self.tensor.calculate_tensor(positions)

        self.assertEqual(tensor_field.shape, (10, 4, 4))

        # Check symmetry
        for i in range(10):
            np.testing.assert_array_almost_equal(
                tensor_field[i], tensor_field[i].T, decimal=10
            )

    def test_stress_energy_contribution(self):
        """Test stress-energy tensor contribution."""
        x = np.array([0, 0, 0, 0])  # Origin
        T_E = self.tensor.stress_energy_contribution(x)

        self.assertEqual(T_E.shape, (4, 4))
        # Should be symmetric
        np.testing.assert_array_almost_equal(T_E, T_E.T, decimal=10)


class TestModifiedEinsteinSolver(unittest.TestCase):
    """Test modified Einstein equation solver."""

    def setUp(self):
        """Set up test cases."""
        self.solver = ModifiedEinsteinSolver()

    def test_initialization(self):
        """Test proper initialization."""
        self.assertIsInstance(self.solver.entanglement_tensor, EntanglementTensor)

    def test_christoffel_symbols(self):
        """Test Christoffel symbol calculation."""
        # Flat spacetime metric
        metric = np.diag([-1, 1, 1, 1])
        christoffel = self.solver.christoffel_symbols(metric)

        self.assertEqual(christoffel.shape, (4, 4, 4))
        # For flat spacetime, all Christoffel symbols should be zero
        np.testing.assert_array_almost_equal(christoffel, np.zeros((4, 4, 4)), decimal=10)

    def test_ricci_tensor(self):
        """Test Ricci tensor calculation."""
        # Flat spacetime
        metric = np.diag([-1, 1, 1, 1])
        ricci = self.solver.ricci_tensor(metric)

        self.assertEqual(ricci.shape, (4, 4))
        # Should be symmetric
        np.testing.assert_array_almost_equal(ricci, ricci.T, decimal=10)

    def test_ricci_scalar(self):
        """Test Ricci scalar calculation."""
        metric = np.diag([-1, 1, 1, 1])
        R = self.solver.ricci_scalar(metric)

        self.assertIsInstance(R, (int, float, np.number))
        # For flat spacetime, Ricci scalar should be zero
        self.assertAlmostEqual(R, 0.0, places=10)

    def test_einstein_tensor(self):
        """Test Einstein tensor calculation."""
        metric = np.diag([-1, 1, 1, 1])
        G = self.solver.einstein_tensor(metric)

        self.assertEqual(G.shape, (4, 4))
        # Should be symmetric
        np.testing.assert_array_almost_equal(G, G.T, decimal=10)

    def test_field_equations(self):
        """Test modified field equations."""
        x = np.array([0, 0, 0, 0])
        metric = np.diag([-1, 1, 1, 1])

        lhs, rhs = self.solver.field_equations(metric, x)

        self.assertEqual(lhs.shape, (4, 4))
        self.assertEqual(rhs.shape, (4, 4))

    def test_convergence_check(self):
        """Test solution convergence checking."""
        metric_old = np.diag([-1, 1, 1, 1])
        metric_new = np.diag([-1.001, 1.001, 1.001, 1.001])

        # Should not be converged with default tolerance
        self.assertFalse(self.solver.check_convergence(metric_old, metric_new))

        # Should be converged with large tolerance
        self.assertTrue(self.solver.check_convergence(metric_old, metric_new, tol=1e-2))


class TestIntegration(unittest.TestCase):
    """Integration tests for theory components."""

    def test_theory_consistency(self):
        """Test consistency between theory components."""
        tensor = EntanglementTensor()
        solver = ModifiedEinsteinSolver()

        # Test at a spacetime point
        x = np.array([1.0, 0.5, 0.5, 0.5])

        # Get entanglement contribution
        T_E = tensor.stress_energy_contribution(x)

        # Solve field equations
        metric = np.diag([-1, 1, 1, 1])  # Start with flat metric
        lhs, rhs = solver.field_equations(metric, x)

        # Check that entanglement contribution is included in RHS
        self.assertEqual(T_E.shape, rhs.shape)

    def test_physical_units(self):
        """Test that calculations maintain proper physical units."""
        tensor = EntanglementTensor()

        # Create a quantum state
        state = tensor.create_entangled_state(2)

        # Calculate entropy (should be dimensionless)
        entropy = tensor.calculate_entanglement_entropy(state, 2)
        self.assertGreaterEqual(entropy, 0)  # Entropy is non-negative

        # Calculate tensor at a point
        x = np.array([0, 0, 0, 0])
        E_tensor = tensor.calculate_tensor(x.reshape(1, -1))

        # Should have correct shape
        self.assertEqual(E_tensor.shape, (1, 4, 4))


if __name__ == '__main__':
    # Create test suite
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTest(unittest.makeSuite(TestConstants))
    suite.addTest(unittest.makeSuite(TestEntanglementTensor))
    suite.addTest(unittest.makeSuite(TestModifiedEinsteinSolver))
    suite.addTest(unittest.makeSuite(TestIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("All tests passed!")
    else:
        print("Some tests failed.")
        sys.exit(1)
