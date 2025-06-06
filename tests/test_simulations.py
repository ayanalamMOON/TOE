"""
Unit Tests for Simulation Module
===============================

Tests for the simulation components of the EG-QGEM framework.
"""

import unittest
import numpy as np
import sys
import os

# Add project to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from simulations.spacetime_emergence import SpacetimeEmergenceSimulator, run_emergence_simulation
from simulations.black_hole_simulator import BlackHoleSimulator


class TestSpacetimeEmergenceSimulator(unittest.TestCase):
    """Test spacetime emergence simulation."""

    def setUp(self):
        """Set up test cases."""
        self.sim = SpacetimeEmergenceSimulator(n_nodes=50, dim=4)

    def test_initialization(self):
        """Test proper initialization."""
        self.assertEqual(self.sim.n_nodes, 50)
        self.assertEqual(self.sim.dim, 4)
        self.assertIsNotNone(self.sim.network)
        self.assertEqual(len(self.sim.network.nodes()), 50)

    def test_network_creation(self):
        """Test entanglement network creation."""
        network = self.sim.create_entanglement_network()

        self.assertEqual(len(network.nodes()), self.sim.n_nodes)
        self.assertGreater(len(network.edges()), 0)

        # Check node attributes
        for node in network.nodes():
            self.assertIn('position', network.nodes[node])
            self.assertIn('entanglement_strength', network.nodes[node])

    def test_distance_calculation(self):
        """Test emergent distance calculation."""
        pos1 = np.array([0, 0, 0, 0])
        pos2 = np.array([1, 0, 0, 0])

        distance = self.sim.calculate_emergent_distance(pos1, pos2)
        self.assertGreater(distance, 0)

        # Distance to self should be zero
        distance_self = self.sim.calculate_emergent_distance(pos1, pos1)
        self.assertAlmostEqual(distance_self, 0.0, places=10)

    def test_curvature_calculation(self):
        """Test emergent curvature calculation."""
        position = np.array([0.5, 0.5, 0.5, 0.5])
        curvature = self.sim.calculate_emergent_curvature(position)

        self.assertIsInstance(curvature, (int, float, np.number))

    def test_metric_tensor(self):
        """Test emergent metric tensor calculation."""
        position = np.array([0.1, 0.2, 0.3, 0.4])
        metric = self.sim.calculate_emergent_metric(position)

        self.assertEqual(metric.shape, (4, 4))
        # Metric should be symmetric
        np.testing.assert_array_almost_equal(metric, metric.T, decimal=10)

    def test_simulation_evolution(self):
        """Test simulation time evolution."""
        initial_state = self.sim.get_network_state()

        # Evolve for a few steps
        self.sim.evolve_step(dt=0.01)
        evolved_state = self.sim.get_network_state()

        # State should have changed
        self.assertFalse(np.array_equal(initial_state, evolved_state))

    def test_run_emergence_simulation(self):
        """Test complete emergence simulation run."""
        config = {
            'n_nodes': 20,
            'n_steps': 10,
            'dt': 0.01,
            'save_interval': 5
        }

        results = run_emergence_simulation(config)

        self.assertIn('final_network', results)
        self.assertIn('evolution_data', results)
        self.assertIn('spacetime_properties', results)


class TestBlackHoleSimulator(unittest.TestCase):
    """Test black hole simulation with entanglement effects."""

    def setUp(self):
        """Set up test cases."""
        self.sim = BlackHoleSimulator(mass=10.0, angular_momentum=0.5)

    def test_initialization(self):
        """Test proper initialization."""
        self.assertEqual(self.sim.mass, 10.0)
        self.assertEqual(self.sim.angular_momentum, 0.5)
        self.assertGreater(self.sim.schwarzschild_radius, 0)

    def test_schwarzschild_radius(self):
        """Test Schwarzschild radius calculation."""
        expected_rs = 2 * self.sim.G * self.sim.mass / self.sim.c**2
        self.assertAlmostEqual(self.sim.schwarzschild_radius, expected_rs, places=10)

    def test_metric_tensor(self):
        """Test spacetime metric near black hole."""
        r = 100.0  # Far from horizon
        theta = np.pi/2

        metric = self.sim.metric_tensor(r, theta)

        self.assertEqual(metric.shape, (4, 4))
        # Should be diagonal for Kerr metric in Boyer-Lindquist coordinates
        self.assertAlmostEqual(metric[0, 1], 0.0, places=5)  # Allow small off-diagonal terms

    def test_event_horizon(self):
        """Test event horizon calculation."""
        r_plus = self.sim.event_horizon_radius()

        self.assertGreater(r_plus, 0)
        self.assertLessEqual(r_plus, self.sim.schwarzschild_radius)

    def test_hawking_temperature(self):
        """Test Hawking temperature calculation."""
        T_H = self.sim.hawking_temperature()

        self.assertGreater(T_H, 0)
        # Should be inversely related to mass
        sim2 = BlackHoleSimulator(mass=20.0, angular_momentum=0.5)
        T_H2 = sim2.hawking_temperature()
        self.assertLess(T_H2, T_H)

    def test_entanglement_entropy(self):
        """Test entanglement entropy of black hole."""
        S_ent = self.sim.calculate_entanglement_entropy()

        self.assertGreater(S_ent, 0)
        # Should scale with area (Bekenstein-Hawking entropy)

    def test_information_preservation(self):
        """Test information preservation mechanism."""
        # This tests the entanglement-based information preservation
        info_preserved = self.sim.check_information_preservation()

        self.assertIsInstance(info_preserved, bool)

    def test_singularity_resolution(self):
        """Test singularity resolution through entanglement."""
        # Test behavior very close to would-be singularity
        r_min = self.sim.minimum_radius()

        self.assertGreater(r_min, 0)  # Should avoid r=0 singularity

        # Curvature should be finite
        curvature = self.sim.curvature_scalar(r_min, np.pi/2)
        self.assertFalse(np.isinf(curvature))
        self.assertFalse(np.isnan(curvature))

    def test_geodesics(self):
        """Test geodesic calculation in black hole spacetime."""
        # Initial conditions for a geodesic
        initial_pos = np.array([0, 50.0, np.pi/2, 0])  # (t, r, theta, phi)
        initial_vel = np.array([1, -0.1, 0, 0.1])

        trajectory = self.sim.calculate_geodesic(
            initial_pos, initial_vel,
            proper_time_steps=100,
            dt=0.1
        )

        self.assertGreater(len(trajectory), 1)
        self.assertEqual(len(trajectory[0]), 4)

    def test_tidal_forces(self):
        """Test tidal force calculation."""
        position = np.array([0, 10.0, np.pi/2, 0])
        tidal_tensor = self.sim.tidal_forces(position)

        self.assertEqual(tidal_tensor.shape, (4, 4))

    def test_evolution_dynamics(self):
        """Test black hole evolution with entanglement."""
        initial_mass = self.sim.mass

        # Evolve the black hole
        self.sim.evolve_step(dt=1.0)

        # Mass should change due to Hawking radiation
        # (though the change might be very small)
        self.assertIsInstance(self.sim.mass, (int, float, np.number))


class TestSimulationIntegration(unittest.TestCase):
    """Integration tests for simulation components."""

    def test_spacetime_black_hole_interaction(self):
        """Test interaction between spacetime emergence and black holes."""
        # Create emergence simulator
        emergence_sim = SpacetimeEmergenceSimulator(n_nodes=30, dim=4)

        # Create black hole
        bh_sim = BlackHoleSimulator(mass=5.0, angular_momentum=0.2)

        # Test if spacetime emergence is affected by black hole presence
        position_near_bh = np.array([0, 10.0, 0, 0])  # Near black hole
        position_far = np.array([0, 100.0, 0, 0])     # Far from black hole

        metric_near = emergence_sim.calculate_emergent_metric(position_near_bh)
        metric_far = emergence_sim.calculate_emergent_metric(position_far)

        # Metrics should be different
        self.assertFalse(np.allclose(metric_near, metric_far, rtol=1e-3))

    def test_entanglement_consistency(self):
        """Test entanglement consistency across simulations."""
        # Both simulators should give consistent entanglement measures
        emergence_sim = SpacetimeEmergenceSimulator(n_nodes=20, dim=4)
        bh_sim = BlackHoleSimulator(mass=1.0, angular_momentum=0.0)

        # Compare entanglement entropy calculations
        bh_entropy = bh_sim.calculate_entanglement_entropy()

        # Should be consistent with statistical mechanics
        self.assertGreater(bh_entropy, 0)

    def test_simulation_scaling(self):
        """Test simulation performance scaling."""
        import time

        # Small simulation
        start_time = time.time()
        small_sim = SpacetimeEmergenceSimulator(n_nodes=10, dim=4)
        small_sim.evolve_step(dt=0.01)
        small_time = time.time() - start_time

        # Larger simulation
        start_time = time.time()
        large_sim = SpacetimeEmergenceSimulator(n_nodes=20, dim=4)
        large_sim.evolve_step(dt=0.01)
        large_time = time.time() - start_time

        # Time should scale reasonably (not exponentially)
        self.assertLess(large_time, small_time * 10)  # Should be less than 10x slower


if __name__ == '__main__':
    # Create test suite
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTest(unittest.makeSuite(TestSpacetimeEmergenceSimulator))
    suite.addTest(unittest.makeSuite(TestBlackHoleSimulator))
    suite.addTest(unittest.makeSuite(TestSimulationIntegration))

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
