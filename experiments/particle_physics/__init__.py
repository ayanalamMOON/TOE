"""
Particle Physics Experiments for EG-QGEM Theory
===============================================

This module provides comprehensive tools for predicting and analyzing
particle physics phenomena within the EG-QGEM theoretical framework,
including collision analysis, detector simulation, accelerator physics,
and experimental design optimization.

Components:
-----------
- EGQGEMCollisionAnalyzer: Comprehensive collision analysis framework
- EGQGEMDetectorSimulation: Multi-layer detector simulation
- EGQGEMAcceleratorPhysics: Beam dynamics and accelerator physics
- EGQGEMExperimentalDesign: Experimental optimization and design
"""

from .collision_analysis import EGQGEMCollisionAnalyzer
from .detector_simulation import EGQGEMDetectorSimulation, DetectorGeometry, ParticleHit, DetectorLayer
from .accelerator_physics import EGQGEMAcceleratorPhysics, BeamParameters, AcceleratorLattice
from .experimental_design import (EGQGEMExperimentalDesign, ExperimentalParameters,
                                PhysicsSignature)

__all__ = [
    'EGQGEMCollisionAnalyzer',
    'EGQGEMDetectorSimulation',
    'DetectorGeometry',
    'ParticleHit',
    'DetectorLayer',
    'EGQGEMAcceleratorPhysics',
    'BeamParameters',
    'AcceleratorLattice',
    'EGQGEMExperimentalDesign',
    'ExperimentalParameters',
    'PhysicsSignature'
]

# Version information
__version__ = "2.0.0"
