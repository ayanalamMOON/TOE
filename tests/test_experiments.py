"""
Unit Tests for Experimental Predictions
======================================

Tests for the experimental prediction components of the EG-QGEM framework.
"""

import unittest
import numpy as np
import sys
import os

# Add project to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from experiments.predictions import generate_experimental_predictions, visualize_predictions


class TestExperimentalPredictions(unittest.TestCase):
    """Test experimental prediction generation."""

    def setUp(self):
        """Set up test cases."""
        self.config = {
            'mass_range': [1e-6, 1e6],  # kg
            'distance_range': [1e-3, 1e3],  # m
            'energy_range': [1e-20, 1e-10],  # J
            'time_range': [1e-15, 1e-3]  # s
        }

    def test_quantum_gravity_predictions(self):
        """Test quantum gravity experiment predictions."""
        predictions = generate_experimental_predictions(self.config)

        self.assertIn('quantum_gravity', predictions)
        qg_pred = predictions['quantum_gravity']

        # Check required fields
        self.assertIn('decoherence_rate', qg_pred)
        self.assertIn('entanglement_signature', qg_pred)
        self.assertIn('gravitational_dephasing', qg_pred)

        # Check data types and ranges
        self.assertIsInstance(qg_pred['decoherence_rate'], (list, np.ndarray))
        self.assertGreater(len(qg_pred['decoherence_rate']), 0)

    def test_gravitational_wave_predictions(self):
        """Test gravitational wave predictions."""
        predictions = generate_experimental_predictions(self.config)

        self.assertIn('gravitational_waves', predictions)
        gw_pred = predictions['gravitational_waves']

        # Check required fields
        self.assertIn('strain_amplitude', gw_pred)
        self.assertIn('frequency_spectrum', gw_pred)
        self.assertIn('entanglement_modulation', gw_pred)

        # Strain should be dimensionless and very small
        strain = np.array(gw_pred['strain_amplitude'])
        self.assertTrue(np.all(strain < 1e-15))  # Typical GW strain scales
        self.assertTrue(np.all(strain > 0))

    def test_cosmological_predictions(self):
        """Test cosmological predictions."""
        predictions = generate_experimental_predictions(self.config)

        self.assertIn('cosmology', predictions)
        cosmo_pred = predictions['cosmology']

        # Check required fields
        self.assertIn('cmb_features', cosmo_pred)
        self.assertIn('dark_matter_distribution', cosmo_pred)
        self.assertIn('hubble_tension_resolution', cosmo_pred)

        # CMB features should have temperature and polarization
        cmb = cosmo_pred['cmb_features']
        self.assertIn('temperature_spectrum', cmb)
        self.assertIn('polarization_spectrum', cmb)

    def test_decoherence_predictions(self):
        """Test fundamental decoherence predictions."""
        predictions = generate_experimental_predictions(self.config)

        self.assertIn('decoherence', predictions)
        decoher_pred = predictions['decoherence']

        # Check required fields
        self.assertIn('decoherence_timescale', decoher_pred)
        self.assertIn('mass_dependence', decoher_pred)
        self.assertIn('spatial_correlation', decoher_pred)

        # Decoherence timescale should be positive
        timescale = np.array(decoher_pred['decoherence_timescale'])
        self.assertTrue(np.all(timescale > 0))

    def test_prediction_consistency(self):
        """Test consistency between different predictions."""
        predictions = generate_experimental_predictions(self.config)

        # All prediction categories should be present
        expected_categories = [
            'quantum_gravity',
            'gravitational_waves',
            'cosmology',
            'decoherence'
        ]

        for category in expected_categories:
            self.assertIn(category, predictions)

    def test_parameter_scaling(self):
        """Test how predictions scale with parameters."""
        # Test with different mass ranges
        config1 = self.config.copy()
        config1['mass_range'] = [1e-9, 1e-6]  # Smaller masses

        config2 = self.config.copy()
        config2['mass_range'] = [1e-3, 1]     # Larger masses

        pred1 = generate_experimental_predictions(config1)
        pred2 = generate_experimental_predictions(config2)

        # Decoherence rates should be different for different mass ranges
        rate1 = np.mean(pred1['quantum_gravity']['decoherence_rate'])
        rate2 = np.mean(pred2['quantum_gravity']['decoherence_rate'])

        self.assertNotAlmostEqual(rate1, rate2, places=5)

    def test_visualization_compatibility(self):
        """Test that predictions can be visualized."""
        predictions = generate_experimental_predictions(self.config)

        # This should not raise an exception
        try:
            fig = visualize_predictions(predictions)
            self.assertIsNotNone(fig)
        except Exception as e:
            self.fail(f"Visualization failed: {e}")

    def test_physical_units(self):
        """Test that predictions have correct physical units."""
        predictions = generate_experimental_predictions(self.config)

        # Gravitational wave strain should be dimensionless
        strain = predictions['gravitational_waves']['strain_amplitude']
        self.assertIsInstance(strain, (list, np.ndarray))

        # Decoherence timescale should be in seconds
        timescale = predictions['decoherence']['decoherence_timescale']
        self.assertIsInstance(timescale, (list, np.ndarray))
        self.assertTrue(np.all(np.array(timescale) > 0))

    def test_experimental_feasibility(self):
        """Test that predictions are within experimentally accessible ranges."""
        predictions = generate_experimental_predictions(self.config)

        # Gravitational wave frequencies should be in detectable range
        freq = predictions['gravitational_waves']['frequency_spectrum']
        freq_array = np.array(freq)

        # Should have some frequencies in LIGO band (10 Hz - 1000 Hz)
        ligo_band = (freq_array >= 10) & (freq_array <= 1000)
        self.assertTrue(np.any(ligo_band))

        # CMB temperature variations should be reasonable
        cmb_temp = predictions['cosmology']['cmb_features']['temperature_spectrum']
        temp_array = np.array(cmb_temp)

        # Should be small fractional variations
        self.assertTrue(np.all(np.abs(temp_array) < 1e-3))  # < 0.1% variations

    def test_entanglement_signatures(self):
        """Test entanglement-specific signatures in predictions."""
        predictions = generate_experimental_predictions(self.config)

        # Quantum gravity predictions should have entanglement signatures
        qg_pred = predictions['quantum_gravity']
        self.assertIn('entanglement_signature', qg_pred)

        ent_sig = qg_pred['entanglement_signature']
        self.assertIsInstance(ent_sig, (list, np.ndarray, dict))

        # Gravitational waves should have entanglement modulation
        gw_pred = predictions['gravitational_waves']
        self.assertIn('entanglement_modulation', gw_pred)

        ent_mod = gw_pred['entanglement_modulation']
        self.assertIsInstance(ent_mod, (list, np.ndarray))


class TestPredictionAccuracy(unittest.TestCase):
    """Test accuracy and reliability of predictions."""

    def test_reproducibility(self):
        """Test that predictions are reproducible."""
        config = {
            'mass_range': [1e-6, 1e-3],
            'distance_range': [1e-2, 1e2],
            'energy_range': [1e-18, 1e-15],
            'time_range': [1e-12, 1e-9]
        }

        # Generate predictions twice
        pred1 = generate_experimental_predictions(config)
        pred2 = generate_experimental_predictions(config)

        # Should be identical (if no randomness)
        for category in pred1.keys():
            if isinstance(pred1[category], dict):
                for key in pred1[category].keys():
                    if isinstance(pred1[category][key], (list, np.ndarray)):
                        np.testing.assert_array_almost_equal(
                            pred1[category][key],
                            pred2[category][key],
                            decimal=10
                        )

    def test_limit_behavior(self):
        """Test behavior in physical limits."""
        # Test weak field limit
        weak_config = {
            'mass_range': [1e-15, 1e-12],  # Very small masses
            'distance_range': [1, 10],
            'energy_range': [1e-25, 1e-20],
            'time_range': [1e-10, 1e-8]
        }

        weak_pred = generate_experimental_predictions(weak_config)

        # In weak field limit, corrections should be small
        decoher_rate = np.array(weak_pred['quantum_gravity']['decoherence_rate'])
        self.assertTrue(np.all(decoher_rate < 1e6))  # Not too large

    def test_dimensional_analysis(self):
        """Test dimensional consistency of predictions."""
        config = {
            'mass_range': [1e-6, 1e-3],
            'distance_range': [1e-3, 1],
            'energy_range': [1e-20, 1e-15],
            'time_range': [1e-15, 1e-10]
        }

        predictions = generate_experimental_predictions(config)

        # Check that all quantities have reasonable magnitudes
        # (This is a sanity check for dimensional errors)

        # Decoherence rates should be in reasonable range (Hz)
        rate = np.array(predictions['quantum_gravity']['decoherence_rate'])
        self.assertTrue(np.all(rate > 1e-20))  # Not ridiculously small
        self.assertTrue(np.all(rate < 1e20))   # Not ridiculously large

        # GW strain should be dimensionless and small
        strain = np.array(predictions['gravitational_waves']['strain_amplitude'])
        self.assertTrue(np.all(strain > 1e-30))  # Not ridiculously small
        self.assertTrue(np.all(strain < 1))      # Should be << 1


if __name__ == '__main__':
    # Create test suite
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTest(unittest.makeSuite(TestExperimentalPredictions))
    suite.addTest(unittest.makeSuite(TestPredictionAccuracy))

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
