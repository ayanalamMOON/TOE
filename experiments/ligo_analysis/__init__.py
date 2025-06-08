"""
LIGO Analysis Module for EG-QGEM Theory
=======================================

This module provides comprehensive tools for analyzing gravitational wave
signatures of entanglement-modified spacetime dynamics in LIGO/Virgo data.

Components:
-----------
- EGQGEMWaveformAnalyzer: Generate EG-QGEM modified gravitational waveforms
- LIGODataProcessor: Process and analyze LIGO strain data
- EntanglementSignalDetector: Detect entanglement signatures in GW data
- EGQGEMParameterEstimator: Bayesian parameter estimation for EG-QGEM
- AdvancedNoiseAnalyzer: Sophisticated noise characterization and glitch detection
- EntanglementMatchedFilter: Matched filtering optimized for EG-QGEM signals
"""

from .waveform_analysis import EGQGEMWaveformAnalyzer
from .data_analysis import LIGODataProcessor
from .signal_detection import EntanglementSignalDetector
from .parameter_estimation import EGQGEMParameterEstimator
from .noise_analysis import AdvancedNoiseAnalyzer
from .matched_filter import EntanglementMatchedFilter

__all__ = [
    'EGQGEMWaveformAnalyzer',
    'LIGODataProcessor',
    'EntanglementSignalDetector',
    'EGQGEMParameterEstimator',
    'AdvancedNoiseAnalyzer',
    'EntanglementMatchedFilter'
]

# Version information
__version__ = "1.0.0"
