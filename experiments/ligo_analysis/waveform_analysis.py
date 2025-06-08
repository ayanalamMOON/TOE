"""
EG-QGEM Waveform Analysis Module
===============================

Generates and analyzes gravitational waveforms with entanglement modifications
for comparison with LIGO/Virgo observations.
"""

import numpy as np
import scipy.signal as signal
from scipy.interpolate import interp1d
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import h5py
import json
from datetime import datetime

# Project imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from theory.constants import CONSTANTS
from theory.entanglement_tensor import EntanglementTensor


class EGQGEMWaveformAnalyzer:
    """
    Advanced waveform analysis for EG-QGEM gravitational wave signatures.
    """

    def __init__(self, sample_rate: float = 4096.0):
        """
        Initialize the waveform analyzer.

        Args:
            sample_rate: Sampling rate in Hz
        """
        self.sample_rate = sample_rate
        self.entanglement_tensor = EntanglementTensor()

        # EG-QGEM modification parameters
        self.entanglement_coupling_range = (1e-6, 1e-2)
        self.frequency_dependent_coupling = True
        self.nonlinear_effects = True

    def generate_inspiral_waveform(self,
                                 mass1: float,
                                 mass2: float,
                                 distance: float,
                                 duration: float = 4.0,
                                 entanglement_params: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        """
        Generate inspiral waveform with EG-QGEM modifications.

        Args:
            mass1, mass2: Component masses in solar masses
            distance: Luminosity distance in Mpc
            duration: Waveform duration in seconds
            entanglement_params: Dictionary of entanglement modification parameters

        Returns:
            Dictionary containing waveform data and metadata
        """
        if entanglement_params is None:
            entanglement_params = {
                'coupling_strength': 1e-4,
                'frequency_cutoff': 100.0,  # Hz
                'phase_modification': True,
                'amplitude_modification': True,
                'polarization_mixing': False
            }

        # Time array
        t = np.arange(0, duration, 1/self.sample_rate)

        # Basic orbital parameters
        M_total = mass1 + mass2
        M_chirp = (mass1 * mass2)**(3/5) / M_total**(1/5)
        eta = mass1 * mass2 / M_total**2

        # Convert to SI units
        M_total_kg = M_total * CONSTANTS.M_sun
        M_chirp_kg = M_chirp * CONSTANTS.M_sun
        distance_m = distance * 3.086e22  # Mpc to meters

        # Frequency evolution (post-Newtonian approximation)
        f_start = 20.0  # Hz
        f_end = 1000.0  # Hz

        # Chirp time
        tau = 5 * CONSTANTS.c**5 / (256 * np.pi**(8/3) * CONSTANTS.G**(5/3) * M_chirp_kg**(5/3))
        t_coalesce = duration * 0.9

        # Frequency evolution
        t_from_merger = t_coalesce - t
        f_t = np.zeros_like(t)

        valid_mask = t_from_merger > 0
        f_t[valid_mask] = (tau / (8*np.pi) / t_from_merger[valid_mask])**(3/8)
        f_t = np.maximum(f_t, f_start)
        f_t = np.minimum(f_t, f_end)

        # Phase evolution
        phi_t = np.zeros_like(t)
        for i in range(1, len(t)):
            phi_t[i] = phi_t[i-1] + 2*np.pi*f_t[i] / self.sample_rate

        # Standard GR amplitude
        amplitude_GR = (CONSTANTS.G**(5/6) * M_chirp_kg**(5/6) /
                       (CONSTANTS.c**(3/2) * distance_m) *
                       (np.pi * f_t * CONSTANTS.G * M_total_kg / CONSTANTS.c**3)**(-7/6))

        # EG-QGEM modifications
        coupling = entanglement_params['coupling_strength']
        f_cutoff = entanglement_params['frequency_cutoff']

        # Frequency-dependent entanglement coupling
        coupling_frequency = coupling * np.exp(-f_t / f_cutoff)

        # Phase modifications from entanglement
        if entanglement_params['phase_modification']:
            # Additional phase evolution from entanglement-geometry coupling
            phi_entanglement = coupling_frequency * np.sin(2*np.pi*f_t*t) * np.cumsum(f_t) / self.sample_rate
            phi_total = phi_t + phi_entanglement
        else:
            phi_total = phi_t

        # Amplitude modifications
        if entanglement_params['amplitude_modification']:
            # Entanglement-induced amplitude modulation
            amplitude_modulation = 1 + coupling_frequency * np.sin(np.pi*f_t*t / f_cutoff)
            amplitude_total = amplitude_GR * amplitude_modulation
        else:
            amplitude_total = amplitude_GR

        # Generate polarizations
        h_plus = amplitude_total * np.cos(phi_total)
        h_cross = amplitude_total * np.sin(phi_total)

        # Polarization mixing from entanglement (if enabled)
        if entanglement_params['polarization_mixing']:
            mixing_angle = coupling_frequency * np.pi/4
            h_plus_mixed = h_plus * np.cos(mixing_angle) - h_cross * np.sin(mixing_angle)
            h_cross_mixed = h_plus * np.sin(mixing_angle) + h_cross * np.cos(mixing_angle)
            h_plus, h_cross = h_plus_mixed, h_cross_mixed

        return {
            'time': t,
            'frequency': f_t,
            'h_plus': h_plus,
            'h_cross': h_cross,
            'amplitude': amplitude_total,
            'phase': phi_total,
            'entanglement_signature': coupling_frequency,
            'metadata': {
                'mass1': mass1,
                'mass2': mass2,
                'distance': distance,
                'chirp_mass': M_chirp,
                'total_mass': M_total,
                'entanglement_params': entanglement_params,
                'generation_time': datetime.now().isoformat()
            }
        }

    def generate_merger_ringdown_waveform(self,
                                        mass1: float,
                                        mass2: float,
                                        spin1: float = 0.0,
                                        spin2: float = 0.0,
                                        entanglement_params: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        """
        Generate merger and ringdown waveform with EG-QGEM modifications.

        Args:
            mass1, mass2: Component masses in solar masses
            spin1, spin2: Dimensionless spin parameters
            entanglement_params: Entanglement modification parameters

        Returns:
            Dictionary containing merger-ringdown waveform data
        """
        if entanglement_params is None:
            entanglement_params = {'coupling_strength': 1e-4}

        # Final black hole properties
        M_final = mass1 + mass2  # Simplified
        a_final = (spin1 + spin2) / 2  # Simplified

        # Quasi-normal mode frequencies for Kerr black holes
        # Using approximate formulas for fundamental l=2, m=2 mode
        M_final_kg = M_final * CONSTANTS.M_sun
        r_g = CONSTANTS.G * M_final_kg / CONSTANTS.c**2

        # QNM frequency (simplified formula)
        f_qnm_base = CONSTANTS.c / (2*np.pi*r_g) * (1 - 0.63*(1-a_final)**(0.3))

        # EG-QGEM modification to QNM frequencies
        coupling = entanglement_params['coupling_strength']
        f_qnm = f_qnm_base * (1 + coupling * np.log(1 + 10*coupling))

        # Damping time
        tau_damp = r_g / CONSTANTS.c / (0.09 + 0.09*a_final)

        # Time array for ringdown (starting at merger)
        t_ringdown = np.arange(0, 10*tau_damp, 1/self.sample_rate)

        # Ringdown amplitude decay
        amplitude_decay = np.exp(-t_ringdown / tau_damp)

        # Initial amplitude (matched to inspiral)
        A_0 = 1e-21  # Typical strain amplitude

        # Generate ringdown signal
        h_plus_ringdown = A_0 * amplitude_decay * np.cos(2*np.pi*f_qnm*t_ringdown)
        h_cross_ringdown = A_0 * amplitude_decay * np.sin(2*np.pi*f_qnm*t_ringdown)

        # Entanglement-induced modulations
        entanglement_modulation = 1 + coupling * np.sin(2*np.pi*f_qnm*t_ringdown/10)
        h_plus_ringdown *= entanglement_modulation
        h_cross_ringdown *= entanglement_modulation

        return {
            'time': t_ringdown,
            'h_plus': h_plus_ringdown,
            'h_cross': h_cross_ringdown,
            'frequency': np.full_like(t_ringdown, f_qnm),
            'qnm_frequency': f_qnm,
            'damping_time': tau_damp,
            'final_mass': M_final,
            'final_spin': a_final,
            'entanglement_coupling': coupling,
            'metadata': {
                'waveform_type': 'merger_ringdown',
                'mass1': mass1,
                'mass2': mass2,
                'spin1': spin1,
                'spin2': spin2,
                'entanglement_params': entanglement_params
            }
        }

    def compute_entanglement_signature_strength(self,
                                              waveform_data: Dict,
                                              analysis_method: str = 'frequency_domain') -> Dict[str, float]:
        """
        Compute the strength of entanglement signatures in a waveform.

        Args:
            waveform_data: Waveform data dictionary
            analysis_method: 'frequency_domain' or 'time_domain'

        Returns:
            Dictionary of signature strength metrics
        """
        h_plus = waveform_data['h_plus']
        h_cross = waveform_data['h_cross']
        time = waveform_data['time']

        if analysis_method == 'frequency_domain':
            # FFT analysis
            freqs, h_plus_fft = signal.welch(h_plus, fs=self.sample_rate, nperseg=2048)
            freqs, h_cross_fft = signal.welch(h_cross, fs=self.sample_rate, nperseg=2048)

            # Look for entanglement-specific spectral features
            coupling = waveform_data['metadata']['entanglement_params']['coupling_strength']

            # Signature strength based on spectral deviations
            spectral_deviation = np.std(h_plus_fft) / np.mean(h_plus_fft)
            signature_strength = coupling * spectral_deviation

            return {
                'signature_strength': signature_strength,
                'spectral_deviation': spectral_deviation,
                'frequency_range': (freqs[0], freqs[-1]),
                'peak_frequency': freqs[np.argmax(h_plus_fft)],
                'coupling_strength': coupling
            }

        elif analysis_method == 'time_domain':
            # Time-domain analysis
            # Look for phase deviations and amplitude modulations

            if 'entanglement_signature' in waveform_data:
                entanglement_signal = waveform_data['entanglement_signature']

                # RMS of entanglement signature
                rms_signature = np.sqrt(np.mean(entanglement_signal**2))

                # Relative strength compared to main signal
                rms_main = np.sqrt(np.mean(h_plus**2))
                relative_strength = rms_signature / rms_main if rms_main > 0 else 0

                return {
                    'signature_strength': relative_strength,
                    'rms_signature': rms_signature,
                    'rms_main_signal': rms_main,
                    'peak_signature': np.max(np.abs(entanglement_signal)),
                    'signal_duration': time[-1] - time[0]
                }

        return {'signature_strength': 0.0, 'error': 'Unknown analysis method'}

    def compare_with_gr_prediction(self,
                                 egqgem_waveform: Dict,
                                 gr_waveform: Dict) -> Dict[str, float]:
        """
        Compare EG-QGEM waveform with pure GR prediction.

        Args:
            egqgem_waveform: EG-QGEM waveform data
            gr_waveform: Pure GR waveform data

        Returns:
            Dictionary of comparison metrics
        """
        # Ensure same time sampling
        t_min = max(egqgem_waveform['time'][0], gr_waveform['time'][0])
        t_max = min(egqgem_waveform['time'][-1], gr_waveform['time'][-1])

        t_common = np.arange(t_min, t_max, 1/self.sample_rate)

        # Interpolate to common time grid
        h_plus_egqgem = np.interp(t_common, egqgem_waveform['time'], egqgem_waveform['h_plus'])
        h_plus_gr = np.interp(t_common, gr_waveform['time'], gr_waveform['h_plus'])

        h_cross_egqgem = np.interp(t_common, egqgem_waveform['time'], egqgem_waveform['h_cross'])
        h_cross_gr = np.interp(t_common, gr_waveform['time'], gr_waveform['h_cross'])

        # Compute differences
        delta_h_plus = h_plus_egqgem - h_plus_gr
        delta_h_cross = h_cross_egqgem - h_cross_gr

        # Metrics
        mismatch_plus = np.sqrt(np.mean(delta_h_plus**2)) / np.sqrt(np.mean(h_plus_gr**2))
        mismatch_cross = np.sqrt(np.mean(delta_h_cross**2)) / np.sqrt(np.mean(h_cross_gr**2))

        # Phase difference
        phase_egqgem = np.angle(h_plus_egqgem + 1j*h_cross_egqgem)
        phase_gr = np.angle(h_plus_gr + 1j*h_cross_gr)
        phase_difference = np.unwrap(phase_egqgem - phase_gr)

        # Frequency difference
        freq_egqgem = np.gradient(np.unwrap(phase_egqgem)) * self.sample_rate / (2*np.pi)
        freq_gr = np.gradient(np.unwrap(phase_gr)) * self.sample_rate / (2*np.pi)
        freq_difference = freq_egqgem - freq_gr

        return {
            'mismatch_h_plus': mismatch_plus,
            'mismatch_h_cross': mismatch_cross,
            'total_mismatch': np.sqrt(mismatch_plus**2 + mismatch_cross**2),
            'max_phase_difference': np.max(np.abs(phase_difference)),
            'rms_phase_difference': np.sqrt(np.mean(phase_difference**2)),
            'max_frequency_difference': np.max(np.abs(freq_difference)),
            'rms_frequency_difference': np.sqrt(np.mean(freq_difference**2)),
            'comparison_duration': t_max - t_min,
            'n_samples': len(t_common)
        }

    def save_waveform_analysis(self,
                             waveform_data: Dict,
                             filename: str,
                             format: str = 'hdf5') -> None:
        """
        Save waveform analysis results to file.

        Args:
            waveform_data: Waveform data dictionary
            filename: Output filename
            format: 'hdf5' or 'json'
        """
        if format == 'hdf5':
            with h5py.File(filename, 'w') as f:
                # Save arrays
                f.create_dataset('time', data=waveform_data['time'])
                f.create_dataset('h_plus', data=waveform_data['h_plus'])
                f.create_dataset('h_cross', data=waveform_data['h_cross'])
                f.create_dataset('frequency', data=waveform_data['frequency'])

                if 'amplitude' in waveform_data:
                    f.create_dataset('amplitude', data=waveform_data['amplitude'])
                if 'phase' in waveform_data:
                    f.create_dataset('phase', data=waveform_data['phase'])
                if 'entanglement_signature' in waveform_data:
                    f.create_dataset('entanglement_signature', data=waveform_data['entanglement_signature'])

                # Save metadata as attributes
                metadata = waveform_data['metadata']
                for key, value in metadata.items():
                    if isinstance(value, dict):
                        grp = f.create_group(key)
                        for subkey, subvalue in value.items():
                            grp.attrs[subkey] = subvalue
                    else:
                        f.attrs[key] = value

        elif format == 'json':
            # Convert numpy arrays to lists for JSON serialization
            json_data = {}
            for key, value in waveform_data.items():
                if isinstance(value, np.ndarray):
                    json_data[key] = value.tolist()
                else:
                    json_data[key] = value

            with open(filename, 'w') as f:
                json.dump(json_data, f, indent=2, default=str)

    def generate_template_bank(self,
                             mass_range: Tuple[float, float] = (5.0, 50.0),
                             entanglement_range: Tuple[float, float] = (1e-6, 1e-3),
                             n_templates: int = 1000) -> List[Dict]:
        """
        Generate a bank of EG-QGEM waveform templates for matched filtering.

        Args:
            mass_range: Range of component masses (solar masses)
            entanglement_range: Range of entanglement coupling strengths
            n_templates: Number of templates to generate

        Returns:
            List of waveform template dictionaries
        """
        templates = []

        # Random parameter sampling
        np.random.seed(42)  # For reproducibility

        for i in range(n_templates):
            # Sample masses
            mass1 = np.random.uniform(mass_range[0], mass_range[1])
            mass2 = np.random.uniform(mass_range[0], mass_range[1])

            # Ensure mass1 >= mass2
            if mass1 < mass2:
                mass1, mass2 = mass2, mass1

            # Sample entanglement parameters
            coupling = np.random.uniform(entanglement_range[0], entanglement_range[1])
            freq_cutoff = np.random.uniform(50.0, 500.0)

            entanglement_params = {
                'coupling_strength': coupling,
                'frequency_cutoff': freq_cutoff,
                'phase_modification': np.random.choice([True, False]),
                'amplitude_modification': np.random.choice([True, False])
            }

            # Generate template
            template = self.generate_inspiral_waveform(
                mass1=mass1,
                mass2=mass2,
                distance=100.0,  # Fixed distance for templates
                duration=2.0,     # Shorter duration for efficiency
                entanglement_params=entanglement_params
            )

            template['template_id'] = i
            templates.append(template)

        return templates


# Additional analysis functions
def compute_overlap(h1: np.ndarray, h2: np.ndarray, psd: Optional[np.ndarray] = None) -> float:
    """
    Compute the overlap between two waveforms.

    Args:
        h1, h2: Complex waveforms
        psd: Power spectral density (optional)

    Returns:
        Overlap value (0 to 1)
    """
    if psd is None:
        # Simple L2 norm without noise weighting
        inner_prod = np.real(np.sum(np.conj(h1) * h2))
        norm1 = np.sqrt(np.real(np.sum(np.conj(h1) * h1)))
        norm2 = np.sqrt(np.real(np.sum(np.conj(h2) * h2)))
    else:
        # Noise-weighted inner product
        # This would require proper frequency-domain implementation
        # Simplified version for now
        inner_prod = np.real(np.sum(np.conj(h1) * h2 / psd))
        norm1 = np.sqrt(np.real(np.sum(np.conj(h1) * h1 / psd)))
        norm2 = np.sqrt(np.real(np.sum(np.conj(h2) * h2 / psd)))

    if norm1 > 0 and norm2 > 0:
        return inner_prod / (norm1 * norm2)
    else:
        return 0.0


def compute_fitting_factor(template: np.ndarray, signal: np.ndarray,
                         psd: Optional[np.ndarray] = None) -> float:
    """
    Compute the fitting factor between a template and signal.

    Args:
        template: Template waveform
        signal: Signal waveform
        psd: Power spectral density

    Returns:
        Fitting factor (maximum overlap over phase shifts)
    """
    # This is a simplified implementation
    # Full implementation would include optimization over phase and time shifts
    return compute_overlap(template, signal, psd)
