"""
LIGO Data Analysis Module for EG-QGEM Theory
===========================================

Advanced data analysis tools for processing LIGO/Virgo gravitational wave data
and searching for entanglement-modified signatures.
"""

import numpy as np
import scipy.signal as signal
from scipy.stats import chi2
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Union
import h5py
import json
from datetime import datetime, timedelta
import warnings

# Project imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from theory.constants import CONSTANTS


class LIGODataProcessor:
    """
    Comprehensive LIGO/Virgo data processing for EG-QGEM analysis.
    """

    def __init__(self,
                 detector: str = 'H1',
                 sample_rate: float = 4096.0):
        """
        Initialize LIGO data processor.

        Args:
            detector: Detector name ('H1', 'L1', 'V1')
            sample_rate: Data sampling rate in Hz
        """
        self.detector = detector
        self.sample_rate = sample_rate

        # Detector-specific parameters
        self.detector_params = self._get_detector_parameters(detector)

        # Analysis parameters
        self.frequency_range = (20.0, 2048.0)  # Hz
        self.segment_duration = 4.0  # seconds
        self.overlap_duration = 0.5  # seconds

    def _get_detector_parameters(self, detector: str) -> Dict[str, Any]:
        """Get detector-specific parameters."""
        params = {
            'H1': {
                'name': 'LIGO Hanford',
                'location': (46.4547, -119.4088),  # lat, lon
                'arm_length': 4000.0,  # meters
                'strain_sensitivity': 1e-23,  # typical
                'frequency_response': 'advanced_ligo'
            },
            'L1': {
                'name': 'LIGO Livingston',
                'location': (30.5629, -90.7742),
                'arm_length': 4000.0,
                'strain_sensitivity': 1e-23,
                'frequency_response': 'advanced_ligo'
            },
            'V1': {
                'name': 'Virgo',
                'location': (43.6314, 10.5045),
                'arm_length': 3000.0,
                'strain_sensitivity': 2e-23,
                'frequency_response': 'advanced_virgo'
            }
        }
        return params.get(detector, params['H1'])

    def load_strain_data(self,
                        data_file: str,
                        gps_start: Optional[float] = None,
                        duration: Optional[float] = None) -> Dict[str, np.ndarray]:
        """
        Load strain data from LIGO data files.

        Args:
            data_file: Path to strain data file (HDF5 format)
            gps_start: GPS start time (optional)
            duration: Data duration in seconds (optional)

        Returns:
            Dictionary containing strain data and metadata
        """
        try:
            with h5py.File(data_file, 'r') as f:
                # Standard LIGO data format
                strain = f['strain']['Strain'][:]
                time_offset = f['strain']['Strain'].attrs['Xstart']
                sample_rate_file = f['strain']['Strain'].attrs['Xspacing']

                # Create time array
                n_samples = len(strain)
                time = time_offset + np.arange(n_samples) * sample_rate_file

                # Apply time selection if specified
                if gps_start is not None and duration is not None:
                    start_idx = int((gps_start - time_offset) * self.sample_rate)
                    end_idx = start_idx + int(duration * self.sample_rate)

                    start_idx = max(0, start_idx)
                    end_idx = min(len(strain), end_idx)

                    strain = strain[start_idx:end_idx]
                    time = time[start_idx:end_idx]

                return {
                    'strain': strain,
                    'time': time,
                    'gps_start': time[0] if len(time) > 0 else time_offset,
                    'duration': time[-1] - time[0] if len(time) > 1 else 0,
                    'sample_rate': 1.0 / sample_rate_file,
                    'detector': self.detector,
                    'n_samples': len(strain)
                }

        except Exception as e:
            # Generate synthetic LIGO-like data for demonstration
            warnings.warn(f"Could not load {data_file}: {e}. Generating synthetic data.")
            return self._generate_synthetic_strain_data(gps_start, duration)

    def _generate_synthetic_strain_data(self,
                                      gps_start: Optional[float] = None,
                                      duration: Optional[float] = None) -> Dict[str, np.ndarray]:
        """Generate synthetic LIGO-like strain data for testing."""
        if gps_start is None:
            gps_start = 1126259462.0  # GW150914 GPS time
        if duration is None:
            duration = 32.0

        n_samples = int(duration * self.sample_rate)
        time = gps_start + np.arange(n_samples) / self.sample_rate

        # Generate realistic noise
        strain = self._generate_detector_noise(n_samples)

        # Add a simulated GW signal (optional)
        if np.random.random() < 0.1:  # 10% chance of signal
            strain += self._inject_test_signal(time - gps_start)

        return {
            'strain': strain,
            'time': time,
            'gps_start': gps_start,
            'duration': duration,
            'sample_rate': self.sample_rate,
            'detector': self.detector,
            'n_samples': n_samples,
            'synthetic': True
        }

    def _generate_detector_noise(self, n_samples: int) -> np.ndarray:
        """Generate realistic detector noise."""
        # Simplified LIGO noise model
        freqs = np.fft.fftfreq(n_samples, 1/self.sample_rate)

        # LIGO-like noise PSD (simplified)
        psd = np.zeros_like(freqs)
        f_pos = freqs[freqs > 0]

        # Seismic noise (low frequency)
        seismic = 1e-40 * (f_pos / 10)**(-4)

        # Thermal noise (mid frequency)
        thermal = 1e-47 * np.ones_like(f_pos)

        # Shot noise (high frequency)
        shot = 1e-47 * (f_pos / 100)**2

        # Combined noise
        psd[freqs > 0] = seismic + thermal + shot
        psd[freqs < 0] = psd[freqs > 0][::-1]

        # Generate colored noise
        noise_fft = np.sqrt(psd * self.sample_rate / 2) * (
            np.random.normal(size=n_samples) + 1j * np.random.normal(size=n_samples)
        )
        noise_fft[0] = 0  # DC component

        noise = np.real(np.fft.ifft(noise_fft))
        return noise

    def _inject_test_signal(self, time: np.ndarray) -> np.ndarray:
        """Inject a test gravitational wave signal."""
        # Simple chirp signal
        f0 = 50.0  # Hz
        f1 = 300.0  # Hz
        t_duration = 0.2  # seconds

        mask = (time >= 15.0) & (time <= 15.0 + t_duration)
        signal = np.zeros_like(time)

        if np.any(mask):
            t_signal = time[mask] - 15.0
            # Chirp frequency
            f_t = f0 + (f1 - f0) * (t_signal / t_duration)**3
            # Amplitude envelope
            amplitude = 1e-21 * np.exp(-((t_signal - t_duration/2) / (t_duration/4))**2)
            signal[mask] = amplitude * np.cos(2*np.pi * np.cumsum(f_t) / self.sample_rate)

        return signal

    def preprocess_data(self,
                       strain_data: Dict[str, np.ndarray],
                       apply_bandpass: bool = True,
                       remove_glitches: bool = True,
                       normalize: bool = True) -> Dict[str, np.ndarray]:
        """
        Preprocess strain data for analysis.

        Args:
            strain_data: Raw strain data dictionary
            apply_bandpass: Apply bandpass filtering
            remove_glitches: Apply glitch removal
            normalize: Normalize the data

        Returns:
            Preprocessed strain data dictionary
        """
        strain = strain_data['strain'].copy()
        time = strain_data['time']

        # Apply bandpass filter
        if apply_bandpass:
            nyquist = self.sample_rate / 2
            low = self.frequency_range[0] / nyquist
            high = min(self.frequency_range[1], nyquist - 1) / nyquist

            b, a = signal.butter(8, [low, high], btype='band')
            strain = signal.filtfilt(b, a, strain)

        # Remove obvious glitches (simplified)
        if remove_glitches:
            strain = self._remove_glitches(strain)

        # Normalize data
        if normalize:
            strain = strain / np.std(strain)

        # Update data dictionary
        processed_data = strain_data.copy()
        processed_data['strain'] = strain
        processed_data['preprocessing'] = {
            'bandpass_applied': apply_bandpass,
            'frequency_range': self.frequency_range,
            'glitch_removal': remove_glitches,
            'normalized': normalize,
            'processing_time': datetime.now().isoformat()
        }

        return processed_data

    def _remove_glitches(self, strain: np.ndarray, threshold: float = 5.0) -> np.ndarray:
        """Simple glitch removal using outlier detection."""
        # Calculate rolling standard deviation
        window_size = int(0.1 * self.sample_rate)  # 0.1 second window

        strain_filtered = strain.copy()

        for i in range(window_size, len(strain) - window_size):
            window = strain[i-window_size:i+window_size]
            local_std = np.std(window)
            local_mean = np.mean(window)

            if np.abs(strain[i] - local_mean) > threshold * local_std:
                # Replace outlier with interpolated value
                strain_filtered[i] = local_mean

        return strain_filtered

    def compute_power_spectral_density(self,
                                     strain_data: Dict[str, np.ndarray],
                                     method: str = 'welch',
                                     nperseg: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Compute power spectral density of strain data.

        Args:
            strain_data: Strain data dictionary
            method: PSD estimation method ('welch', 'periodogram')
            nperseg: Length of each segment for Welch method

        Returns:
            Dictionary containing frequency array and PSD
        """
        strain = strain_data['strain']

        if nperseg is None:
            nperseg = int(4 * self.sample_rate)  # 4 second segments

        if method == 'welch':
            freqs, psd = signal.welch(
                strain,
                fs=self.sample_rate,
                nperseg=nperseg,
                noverlap=nperseg//2,
                window='hann'
            )
        elif method == 'periodogram':
            freqs, psd = signal.periodogram(
                strain,
                fs=self.sample_rate,
                window='hann'
            )
        else:
            raise ValueError(f"Unknown PSD method: {method}")

        return {
            'frequencies': freqs,
            'psd': psd,
            'method': method,
            'nperseg': nperseg,
            'detector': self.detector,
            'frequency_resolution': freqs[1] - freqs[0]
        }

    def search_for_transients(self,
                            strain_data: Dict[str, np.ndarray],
                            q_range: Tuple[float, float] = (4.0, 100.0),
                            frequency_range: Optional[Tuple[float, float]] = None,
                            snr_threshold: float = 8.0) -> List[Dict[str, Any]]:
        """
        Search for transient signals using Q-transform analysis.

        Args:
            strain_data: Strain data dictionary
            q_range: Range of Q values for Q-transform
            frequency_range: Frequency range for search
            snr_threshold: SNR threshold for detection

        Returns:
            List of detected transient events
        """
        strain = strain_data['strain']
        time = strain_data['time']

        if frequency_range is None:
            frequency_range = self.frequency_range

        # Simplified Q-transform implementation
        # In practice, would use specialized packages like gwpy

        detections = []

        # Sliding window analysis
        window_duration = 1.0  # seconds
        window_samples = int(window_duration * self.sample_rate)
        hop_samples = window_samples // 4

        for i in range(0, len(strain) - window_samples, hop_samples):
            window_strain = strain[i:i+window_samples]
            window_time = time[i:i+window_samples]

            # Compute spectrogram
            f, t_spec, Sxx = signal.spectrogram(
                window_strain,
                fs=self.sample_rate,
                nperseg=256,
                noverlap=128
            )

            # Apply frequency range filter
            freq_mask = (f >= frequency_range[0]) & (f <= frequency_range[1])
            f_filtered = f[freq_mask]
            Sxx_filtered = Sxx[freq_mask, :]

            # Look for excess power
            background_power = np.median(Sxx_filtered, axis=1, keepdims=True)
            snr_map = Sxx_filtered / background_power

            # Find peaks above threshold
            peak_indices = np.where(snr_map > snr_threshold)

            if len(peak_indices[0]) > 0:
                for j in range(len(peak_indices[0])):
                    freq_idx = peak_indices[0][j]
                    time_idx = peak_indices[1][j]

                    detection = {
                        'time': window_time[0] + t_spec[time_idx],
                        'frequency': f_filtered[freq_idx],
                        'snr': snr_map[freq_idx, time_idx],
                        'duration': 0.1,  # Estimated
                        'bandwidth': 10.0,  # Estimated
                        'detector': self.detector
                    }
                    detections.append(detection)

        # Remove duplicates and sort by SNR
        detections = sorted(detections, key=lambda x: x['snr'], reverse=True)

        return detections

    def cross_correlate_detectors(self,
                                strain_data1: Dict[str, np.ndarray],
                                strain_data2: Dict[str, np.ndarray],
                                max_delay: float = 0.01) -> Dict[str, np.ndarray]:
        """
        Cross-correlate data from two detectors.

        Args:
            strain_data1: First detector strain data
            strain_data2: Second detector strain data
            max_delay: Maximum time delay to consider (seconds)

        Returns:
            Cross-correlation results
        """
        strain1 = strain_data1['strain']
        strain2 = strain_data2['strain']

        # Ensure same length
        min_length = min(len(strain1), len(strain2))
        strain1 = strain1[:min_length]
        strain2 = strain2[:min_length]

        # Compute cross-correlation
        correlation = signal.correlate(strain1, strain2, mode='full')

        # Time delays
        max_delay_samples = int(max_delay * self.sample_rate)
        center = len(correlation) // 2

        start_idx = center - max_delay_samples
        end_idx = center + max_delay_samples + 1

        correlation_windowed = correlation[start_idx:end_idx]
        delays = np.arange(-max_delay_samples, max_delay_samples + 1) / self.sample_rate

        # Find peak correlation
        peak_idx = np.argmax(np.abs(correlation_windowed))
        peak_delay = delays[peak_idx]
        peak_correlation = correlation_windowed[peak_idx]

        return {
            'delays': delays,
            'correlation': correlation_windowed,
            'peak_delay': peak_delay,
            'peak_correlation': peak_correlation,
            'correlation_coefficient': peak_correlation / np.sqrt(np.sum(strain1**2) * np.sum(strain2**2)),
            'detectors': [strain_data1['detector'], strain_data2['detector']]
        }

    def estimate_noise_floor(self,
                           strain_data: Dict[str, np.ndarray],
                           segment_duration: float = 32.0) -> Dict[str, np.ndarray]:
        """
        Estimate the noise floor for the detector.

        Args:
            strain_data: Strain data dictionary
            segment_duration: Duration of segments for noise estimation

        Returns:
            Noise floor estimate dictionary
        """
        strain = strain_data['strain']
        time = strain_data['time']

        segment_samples = int(segment_duration * self.sample_rate)
        n_segments = len(strain) // segment_samples

        if n_segments < 1:
            # Use entire data if too short
            freqs, psd = signal.welch(strain, fs=self.sample_rate)
        else:
            # Compute PSD for each segment
            psds = []

            for i in range(n_segments):
                start_idx = i * segment_samples
                end_idx = start_idx + segment_samples
                segment = strain[start_idx:end_idx]

                freqs, psd_segment = signal.welch(segment, fs=self.sample_rate)
                psds.append(psd_segment)

            # Take median PSD as noise floor estimate
            psd = np.median(psds, axis=0)

        # Apply frequency range filter
        freq_mask = (freqs >= self.frequency_range[0]) & (freqs <= self.frequency_range[1])

        return {
            'frequencies': freqs[freq_mask],
            'noise_floor': psd[freq_mask],
            'n_segments': n_segments,
            'segment_duration': segment_duration,
            'detector': self.detector,
            'estimation_time': datetime.now().isoformat()
        }

    def save_analysis_results(self,
                            results: Dict[str, Any],
                            filename: str,
                            format: str = 'hdf5') -> None:
        """
        Save analysis results to file.

        Args:
            results: Analysis results dictionary
            filename: Output filename
            format: Output format ('hdf5' or 'json')
        """
        if format == 'hdf5':
            with h5py.File(filename, 'w') as f:
                self._save_dict_to_hdf5(results, f)
        elif format == 'json':
            # Convert numpy arrays to lists for JSON
            json_results = self._convert_arrays_for_json(results)
            with open(filename, 'w') as f:
                json.dump(json_results, f, indent=2, default=str)
        else:
            raise ValueError(f"Unknown format: {format}")

    def _save_dict_to_hdf5(self, data: Dict[str, Any], hdf5_group) -> None:
        """Recursively save dictionary to HDF5 group."""
        for key, value in data.items():
            if isinstance(value, dict):
                subgroup = hdf5_group.create_group(key)
                self._save_dict_to_hdf5(value, subgroup)
            elif isinstance(value, np.ndarray):
                hdf5_group.create_dataset(key, data=value)
            elif isinstance(value, (list, tuple)):
                try:
                    hdf5_group.create_dataset(key, data=np.array(value))
                except:
                    hdf5_group.attrs[key] = str(value)
            else:
                try:
                    hdf5_group.attrs[key] = value
                except:
                    hdf5_group.attrs[key] = str(value)

    def _convert_arrays_for_json(self, data: Any) -> Any:
        """Convert numpy arrays to lists for JSON serialization."""
        if isinstance(data, dict):
            return {key: self._convert_arrays_for_json(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._convert_arrays_for_json(item) for item in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (np.integer, np.floating)):
            return float(data)
        else:
            return data


# Utility functions for LIGO data analysis
def compute_antenna_pattern(detector: str,
                          ra: float,
                          dec: float,
                          psi: float,
                          gps_time: float) -> Tuple[float, float]:
    """
    Compute antenna pattern functions for a detector.

    Args:
        detector: Detector name ('H1', 'L1', 'V1')
        ra: Right ascension (radians)
        dec: Declination (radians)
        psi: Polarization angle (radians)
        gps_time: GPS time

    Returns:
        Tuple of (F_plus, F_cross) antenna pattern values
    """
    # Simplified antenna pattern calculation
    # In practice, would use proper coordinate transformations

    # Placeholder implementation
    F_plus = 0.5 * (1 + np.cos(dec)**2) * np.cos(2*ra + 2*psi)
    F_cross = np.cos(dec) * np.sin(2*ra + 2*psi)

    return F_plus, F_cross


def time_delay_between_detectors(detector1: str,
                               detector2: str,
                               ra: float,
                               dec: float) -> float:
    """
    Compute time delay between two detectors for a source.

    Args:
        detector1, detector2: Detector names
        ra: Right ascension (radians)
        dec: Declination (radians)

    Returns:
        Time delay in seconds
    """
    # Earth rotation and detector positions would be needed
    # for precise calculation. Simplified version:

    # Typical delays between LIGO detectors
    delays = {
        ('H1', 'L1'): 0.010,  # ~10 ms
        ('H1', 'V1'): 0.027,  # ~27 ms
        ('L1', 'V1'): 0.026   # ~26 ms
    }

    key = tuple(sorted([detector1, detector2]))
    base_delay = delays.get(key, 0.0)

    # Modulate by source direction (simplified)
    direction_factor = np.sin(dec) * np.cos(ra)

    return base_delay * direction_factor
