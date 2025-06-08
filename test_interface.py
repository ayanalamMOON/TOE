#!/usr/bin/env python3
"""
Test script for debugging the research interface
"""

import sys
import traceback

def test_imports():
    """Test all imports step by step"""
    print("=" * 50)
    print("TESTING RESEARCH INTERFACE IMPORTS")
    print("=" * 50)

    try:
        print("1. Testing basic Python imports...")
        import numpy as np
        import matplotlib.pyplot as plt
        print("✓ Basic imports successful")
    except Exception as e:
        print(f"✗ Basic imports failed: {e}")
        return False

    try:
        print("2. Testing theory imports...")
        from theory.constants import CONSTANTS
        from theory.entanglement_tensor import EntanglementTensor
        print("✓ Theory imports successful")
    except Exception as e:
        print(f"✗ Theory imports failed: {e}")
        traceback.print_exc()
        return False

    try:
        print("3. Testing LIGO analysis imports...")
        from experiments.ligo_analysis import EGQGEMWaveformAnalyzer
        print("✓ Waveform analyzer imported")

        from experiments.ligo_analysis import LIGODataProcessor
        print("✓ Data processor imported")

        from experiments.ligo_analysis import EntanglementSignalDetector
        print("✓ Signal detector imported")

        from experiments.ligo_analysis import EGQGEMParameterEstimator
        print("✓ Parameter estimator imported")

    except Exception as e:
        print(f"✗ LIGO analysis imports failed: {e}")
        traceback.print_exc()
        return False

    try:
        print("4. Testing particle physics imports...")
        from experiments.particle_physics import EGQGEMCollisionAnalyzer
        print("✓ Collision analyzer imported")

        from experiments.particle_physics import EGQGEMDetectorSimulation
        print("✓ Detector simulation imported")

        from experiments.particle_physics import EGQGEMAcceleratorPhysics
        print("✓ Accelerator physics imported")

        from experiments.particle_physics import EGQGEMExperimentalDesign
        print("✓ Experimental design imported")

    except Exception as e:
        print(f"✗ Particle physics imports failed: {e}")
        traceback.print_exc()
        return False

    try:
        print("5. Testing research interface...")
        from research_interface import EGQGEMResearchInterface
        interface = EGQGEMResearchInterface()
        print("✓ Research interface created successfully")
        return True

    except Exception as e:
        print(f"✗ Research interface failed: {e}")
        traceback.print_exc()
        return False

def test_ligo_analysis():
    """Test LIGO analysis functionality"""
    print("\n" + "=" * 50)
    print("TESTING LIGO ANALYSIS FUNCTIONALITY")
    print("=" * 50)

    try:
        from research_interface import EGQGEMResearchInterface
        interface = EGQGEMResearchInterface()

        print("Running LIGO analysis...")
        interface.run_ligo_analysis()
        print("✓ LIGO analysis completed successfully")
        return True

    except Exception as e:
        print(f"✗ LIGO analysis failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imports()
    if success:
        test_ligo_analysis()
    else:
        print("\nImport tests failed. Cannot proceed with functionality tests.")
