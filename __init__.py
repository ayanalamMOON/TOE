"""
Entangled Geometrodynamics (EG-QGEM) Research Framework
=====================================================

Core theoretical framework for studying emergent spacetime from quantum entanglement.

This module provides the fundamental classes and functions for:
- Entanglement tensor calculations
- Field equation solving
- Geometric emergence modeling
- Quantum-gravitational coupling
"""

__version__ = "1.0.0"
__author__ = "EG-QGEM Research Team"
__email__ = "research@eg-qgem.org"

# Core theory modules
from .theory import (
    EntanglementTensor,
    ModifiedEinstein,
    GeometryEmergence,
    QuantumGravityCoupling
)

# Simulation modules
from .simulations import (
    SpacetimeSimulator,
    BlackHoleSimulator,
    CosmologySimulator,
    GravitationalWaveSimulator
)

# Experimental prediction modules
from .experiments import (
    QuantumGravityExperiment,
    DecoherencePredictor,
    DarkMatterAnalyzer,
    CMBSignatureCalculator
)

# Utility modules
from .utils import (
    Constants,
    MathUtils,
    DataProcessor,
    Visualizer
)

# Make key constants available at package level
from .utils.constants import *

__all__ = [
    'EntanglementTensor',
    'ModifiedEinstein',
    'GeometryEmergence',
    'QuantumGravityCoupling',
    'SpacetimeSimulator',
    'BlackHoleSimulator',
    'CosmologySimulator',
    'GravitationalWaveSimulator',
    'QuantumGravityExperiment',
    'DecoherencePredictor',
    'DarkMatterAnalyzer',
    'CMBSignatureCalculator',
    'Constants',
    'MathUtils',
    'DataProcessor',
    'Visualizer'
]
