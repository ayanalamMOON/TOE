"""
Physical and Mathematical Constants for EG-QGEM Theory
====================================================

This module defines all the fundamental constants used in the
Entangled Geometrodynamics framework.
"""

import numpy as np

# Universal Physical Constants
c = 299792458.0  # Speed of light (m/s)
G = 6.67430e-11  # Gravitational constant (m³/kg⋅s²)
hbar = 1.054571817e-34  # Reduced Planck constant (J⋅s)
k_B = 1.380649e-23  # Boltzmann constant (J/K)
e = 1.602176634e-19  # Elementary charge (C)

# Planck Units
l_planck = np.sqrt(hbar * G / c**3)  # Planck length (m)
t_planck = l_planck / c  # Planck time (s)
m_planck = np.sqrt(hbar * c / G)  # Planck mass (kg)
E_planck = m_planck * c**2  # Planck energy (J)

# EG-QGEM Specific Constants
kappa_E = 1.0e-42  # Entanglement-curvature coupling constant (m²/J)
chi_E = 1.0e-20  # Spin-entanglement coupling constant (dimensionless)
l_E = l_planck  # Entanglement saturation length scale (m)
Lambda_E = 1.0e-52  # Entanglement cosmological term (m⁻²)

# Derived EG-QGEM Scales
rho_E_crit = c**4 / (8 * np.pi * G * kappa_E)  # Critical entanglement density (kg/m³)
t_E_decoher = hbar / (chi_E * c**2)  # Gravitational decoherence timescale (s)

# Cosmological Constants
H_0 = 67.4  # Hubble constant (km/s/Mpc)
Omega_m = 0.315  # Matter density parameter
Omega_Lambda = 0.685  # Dark energy density parameter
Omega_b = 0.049  # Baryon density parameter

# Conversion Factors
eV_to_J = 1.602176634e-19  # electron volts to joules
Mpc_to_m = 3.0857e22  # megaparsecs to meters
yr_to_s = 365.25 * 24 * 3600  # years to seconds

# Mathematical Constants
pi = np.pi
sqrt_2 = np.sqrt(2)
sqrt_pi = np.sqrt(np.pi)
euler_gamma = 0.5772156649015329  # Euler-Mascheroni constant

class EGQGEMConstants:
    """Container class for EG-QGEM theoretical constants."""

    def __init__(self):
        # Store all constants as class attributes
        for name, value in globals().items():
            if isinstance(value, (int, float, complex, np.number)):
                setattr(self, name, value)

    def planck_units(self):
        """Return dictionary of Planck units."""
        return {
            'length': l_planck,
            'time': t_planck,
            'mass': m_planck,
            'energy': E_planck
        }

    def entanglement_scales(self):
        """Return dictionary of EG-QGEM characteristic scales."""
        return {
            'coupling': kappa_E,
            'spin_coupling': chi_E,
            'length_scale': l_E,
            'critical_density': rho_E_crit,
            'decoherence_time': t_E_decoher
        }

    def cosmological_params(self):
        """Return dictionary of cosmological parameters."""
        return {
            'hubble': H_0,
            'matter_density': Omega_m,
            'dark_energy_density': Omega_Lambda,
            'baryon_density': Omega_b
        }

# Create global constants instance
CONSTANTS = EGQGEMConstants()
