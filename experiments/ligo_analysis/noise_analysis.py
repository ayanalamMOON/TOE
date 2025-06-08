"""
Advanced Noise Analysis for LIGO Data
=====================================

This module provides sophisticated noise analysis tools for LIGO data,
including non-Gaussian noise characterization, glitch identification,
and noise subtraction techniques optimized for EG-QGEM signal detection.
"""

import numpy as np
import scipy.signal as signal
import scipy.stats as stats
from scipy import ndimage
from scipy.interpolate import interp1d
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import h5py
from collections import defaultdict

from theory.constants import CONSTANTS


@dataclass
class NoiseCharacteristics:
    """Characteristics of detector noise."""
    psd: np.ndarray
    frequencies: np.ndarray
    non_gaussian_parameters: Dict[str, float]
    glitch_rate: float
    spectral_lines: List[Tuple[float, float]]  # (frequency, amplitude)
    coherence_time: float
    stationarity_metric: float


@dataclass
class GlitchCandidate:
    """Detected glitch candidate."""
    time: float
    duration: float
    frequency_range: Tuple[float, float]
    snr: float
    glitch_type: str
    confidence: float
    morphology_params: Dict[str, float]


class AdvancedNoiseAnalyzer:
    """
    Advanced noise analysis for LIGO gravitational wave detectors.

    Provides tools for characterizing non-Gaussian noise, identifying
    glitches, and performing noise subtraction optimized for detecting
    EG-QGEM signatures.
    """

    def __init__(self, sampling_rate: float = 4096):
        """
        Initialize noise analyzer.

        Args:
            sampling_rate: Data sampling rate in Hz
        """
        self.sampling_rate = sampling_rate
        self.constants = CONSTANTS

        # Standard LIGO frequency bands
        self.frequency_bands = {
            'low': (20, 60),
            'mid': (60, 300),
            'high': (300, 1000),
            'very_high': (1000, 2000)
        }

        # Known LIGO noise lines (simplified list)
        self.known_lines = [
            60.0, 120.0, 180.0,  # Power line harmonics
            16.67, 33.33, 50.0,  # Mirror suspension
            393.1, 786.2,        # Violin modes
        ]

    def analyze_noise_characteristics(self, strain_data: np.ndarray,
                                    time_vector: Optional[np.ndarray] = None) -> NoiseCharacteristics:
        """
        Comprehensive noise characterization.

        Args:
            strain_data: Strain time series
            time_vector: Time vector (if None, generated from sampling rate)

        Returns:
            Complete noise characteristics
        """
        if time_vector is None:
            time_vector = np.arange(len(strain_data)) / self.sampling_rate

        # Calculate power spectral density
        frequencies, psd = self._calculate_robust_psd(strain_data)

        # Analyze non-Gaussian properties
        non_gaussian_params = self._analyze_non_gaussian_noise(strain_data)

        # Detect spectral lines
        spectral_lines = self._detect_spectral_lines(frequencies, psd)

        # Estimate glitch rate
        glitch_rate = self._estimate_glitch_rate(strain_data, time_vector)

        # Calculate coherence time
        coherence_time = self._calculate_coherence_time(strain_data)

        # Assess stationarity
        stationarity_metric = self._assess_stationarity(strain_data)

        return NoiseCharacteristics(
            psd=psd,
            frequencies=frequencies,
            non_gaussian_parameters=non_gaussian_params,
            glitch_rate=glitch_rate,
            spectral_lines=spectral_lines,
            coherence_time=coherence_time,
            stationarity_metric=stationarity_metric
        )

    def _calculate_robust_psd(self, strain_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate robust power spectral density using Welch's method."""
        # Use overlapping windows with Hann tapering
        nperseg = int(self.sampling_rate * 4)  # 4-second segments
        noverlap = nperseg // 2

        frequencies, psd = signal.welch(
            strain_data,
            fs=self.sampling_rate,
            window='hann',
            nperseg=nperseg,
            noverlap=noverlap,
            detrend='linear'
        )

        # Apply median filtering to reduce outliers
        psd_smooth = ndimage.median_filter(psd, size=5)

        return frequencies, psd_smooth

    def _analyze_non_gaussian_noise(self, strain_data: np.ndarray) -> Dict[str, float]:
        """Analyze non-Gaussian properties of the noise."""
        # Remove outliers for more robust statistics
        median = np.median(strain_data)
        mad = np.median(np.abs(strain_data - median))
        threshold = 5 * mad
        clean_data = strain_data[np.abs(strain_data - median) < threshold]

        # Calculate higher-order moments
        skewness = stats.skew(clean_data)
        kurtosis = stats.kurtosis(clean_data)

        # Kolmogorov-Smirnov test for normality
        ks_stat, ks_pvalue = stats.kstest(
            (clean_data - np.mean(clean_data)) / np.std(clean_data),
            'norm'
        )

        # Anderson-Darling test for normality
        ad_stat, ad_critical, ad_significance = stats.anderson(clean_data, dist='norm')

        # Estimate noise distribution parameters
        try:
            # Fit to Student's t-distribution (more robust for heavy tails)
            t_params = stats.t.fit(clean_data)
            t_dof = t_params[0]
        except:
            t_dof = np.inf

        # Calculate non-Gaussianity metrics
        ng_metric = np.sqrt(skewness**2 + (kurtosis - 3)**2)

        return {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pvalue,
            'anderson_darling_stat': ad_stat,
            'non_gaussianity_metric': ng_metric,
            'student_t_dof': t_dof,
            'outlier_fraction': 1 - len(clean_data) / len(strain_data)
        }

    def _detect_spectral_lines(self, frequencies: np.ndarray,
                             psd: np.ndarray) -> List[Tuple[float, float]]:
        """Detect narrow spectral lines in the power spectrum."""
        spectral_lines = []

        # Convert to log scale for better peak detection
        log_psd = np.log10(psd + 1e-50)

        # Smooth the spectrum to identify the background
        background = ndimage.gaussian_filter1d(log_psd, sigma=10)

        # Find peaks above background
        excess = log_psd - background
        threshold = 3 * np.std(excess)

        peaks, properties = signal.find_peaks(
            excess,
            height=threshold,
            width=1,
            distance=5
        )

        for peak in peaks:
            freq = frequencies[peak]
            amplitude = 10**(log_psd[peak] - background[peak])

            # Check if it's a known line or harmonics
            is_known = any(abs(freq - line) < 0.1 or
                          abs(freq % line) < 0.1 for line in self.known_lines)

            spectral_lines.append((freq, amplitude))

        return spectral_lines

    def _estimate_glitch_rate(self, strain_data: np.ndarray,
                            time_vector: np.ndarray) -> float:
        """Estimate glitch rate using excess power detection."""
        # Calculate Q-transform for time-frequency analysis
        q_transform = self._calculate_q_transform(strain_data)

        # Detect transient events
        threshold = 5.0  # SNR threshold
        events = q_transform > threshold

        # Count discrete events (connected components)
        labeled_events, num_events = ndimage.label(events)

        # Calculate rate
        duration = time_vector[-1] - time_vector[0]
        glitch_rate = num_events / duration  # events per second

        return glitch_rate

    def _calculate_q_transform(self, strain_data: np.ndarray) -> np.ndarray:
        """Calculate Q-transform for time-frequency analysis."""
        # Simplified Q-transform implementation
        # In practice, would use GWpy or similar specialized library

        # Create frequency array
        f_min, f_max = 20, 1000
        q_factor = 20

        frequencies = np.logspace(np.log10(f_min), np.log10(f_max), 100)
        times = np.arange(len(strain_data)) / self.sampling_rate

        # Time-frequency representation
        q_transform = np.zeros((len(frequencies), len(times)))

        for i, freq in enumerate(frequencies):
            # Create Morlet wavelet
            sigma_t = q_factor / (2 * np.pi * freq)
            window_size = int(6 * sigma_t * self.sampling_rate)

            if window_size > len(strain_data):
                continue

            t_window = np.linspace(-3*sigma_t, 3*sigma_t, window_size)
            wavelet = np.exp(1j * 2 * np.pi * freq * t_window) * \
                     np.exp(-t_window**2 / (2 * sigma_t**2))

            # Convolve with data
            convolution = np.convolve(strain_data, wavelet, mode='same')
            q_transform[i, :] = np.abs(convolution)**2

        return np.max(q_transform, axis=0)  # Max over frequency at each time

    def _calculate_coherence_time(self, strain_data: np.ndarray) -> float:
        """Calculate noise coherence time using autocorrelation."""
        # Calculate autocorrelation function
        autocorr = np.correlate(strain_data, strain_data, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0]  # Normalize

        # Find where autocorrelation drops to 1/e
        threshold = 1.0 / np.e
        coherence_samples = np.argmax(autocorr < threshold)

        if coherence_samples == 0:
            coherence_samples = len(autocorr) // 2

        coherence_time = coherence_samples / self.sampling_rate
        return coherence_time

    def _assess_stationarity(self, strain_data: np.ndarray) -> float:
        """Assess stationarity using sliding window variance."""
        window_size = int(self.sampling_rate * 10)  # 10-second windows
        n_windows = len(strain_data) // window_size

        if n_windows < 2:
            return 1.0  # Assume stationary for short data

        variances = []
        for i in range(n_windows):
            start = i * window_size
            end = start + window_size
            window_var = np.var(strain_data[start:end])
            variances.append(window_var)

        # Calculate coefficient of variation of variances
        mean_var = np.mean(variances)
        std_var = np.std(variances)

        if mean_var == 0:
            return 1.0

        stationarity_metric = 1 - (std_var / mean_var)  # 1 = stationary, 0 = non-stationary
        return max(0, stationarity_metric)

    def detect_glitches(self, strain_data: np.ndarray,
                       time_vector: Optional[np.ndarray] = None,
                       snr_threshold: float = 8.0) -> List[GlitchCandidate]:
        """
        Detect glitches using multiple algorithms.

        Args:
            strain_data: Strain time series
            time_vector: Time vector
            snr_threshold: SNR threshold for detection

        Returns:
            List of glitch candidates
        """
        if time_vector is None:
            time_vector = np.arange(len(strain_data)) / self.sampling_rate

        candidates = []

        # Method 1: Excess power detection
        excess_candidates = self._detect_excess_power_glitches(
            strain_data, time_vector, snr_threshold
        )
        candidates.extend(excess_candidates)

        # Method 2: Morphological detection
        morph_candidates = self._detect_morphological_glitches(
            strain_data, time_vector, snr_threshold
        )
        candidates.extend(morph_candidates)

        # Method 3: Statistical outlier detection
        outlier_candidates = self._detect_statistical_outliers(
            strain_data, time_vector, snr_threshold
        )
        candidates.extend(outlier_candidates)

        # Cluster nearby candidates
        candidates = self._cluster_glitch_candidates(candidates)

        return candidates

    def _detect_excess_power_glitches(self, strain_data: np.ndarray,
                                    time_vector: np.ndarray,
                                    snr_threshold: float) -> List[GlitchCandidate]:
        """Detect glitches using excess power in time-frequency domain."""
        candidates = []

        # Calculate time-frequency representation
        q_transform = self._calculate_detailed_q_transform(strain_data)
        frequencies = np.logspace(np.log10(20), np.log10(1000), q_transform.shape[0])

        # Find excess power regions
        threshold_map = snr_threshold * np.ones_like(q_transform)
        excess_regions = q_transform > threshold_map

        # Label connected components
        labeled_regions, n_regions = ndimage.label(excess_regions)

        for region_id in range(1, n_regions + 1):
            region_mask = labeled_regions == region_id
            region_coords = np.where(region_mask)

            if len(region_coords[0]) < 5:  # Skip small regions
                continue

            # Extract region properties
            freq_indices = region_coords[0]
            time_indices = region_coords[1]

            min_freq = frequencies[np.min(freq_indices)]
            max_freq = frequencies[np.max(freq_indices)]
            start_time = time_vector[np.min(time_indices)]
            end_time = time_vector[np.max(time_indices)]

            # Calculate SNR
            region_power = q_transform[region_mask]
            snr = np.max(region_power)

            # Classify glitch type based on morphology
            duration = end_time - start_time
            bandwidth = max_freq - min_freq

            if duration < 0.1 and bandwidth > 100:
                glitch_type = "blip"
            elif duration > 1.0 and bandwidth < 50:
                glitch_type = "scattered_light"
            elif min_freq < 100 and duration > 0.5:
                glitch_type = "low_frequency_noise"
            else:
                glitch_type = "unclassified"

            candidates.append(GlitchCandidate(
                time=start_time + duration/2,
                duration=duration,
                frequency_range=(min_freq, max_freq),
                snr=snr,
                glitch_type=glitch_type,
                confidence=min(1.0, snr / snr_threshold),
                morphology_params={
                    'bandwidth': bandwidth,
                    'central_frequency': np.sqrt(min_freq * max_freq),
                    'q_factor': np.sqrt(min_freq * max_freq) / bandwidth
                }
            ))

        return candidates

    def _calculate_detailed_q_transform(self, strain_data: np.ndarray) -> np.ndarray:
        """Calculate detailed Q-transform for glitch detection."""
        # More detailed implementation than the simple version
        f_min, f_max = 20, 1000
        n_freqs = 200

        frequencies = np.logspace(np.log10(f_min), np.log10(f_max), n_freqs)
        n_times = len(strain_data)

        q_transform = np.zeros((n_freqs, n_times))

        for i, freq in enumerate(frequencies):
            # Adaptive Q-factor
            q_factor = 20 * np.sqrt(freq / 100)
            sigma_t = q_factor / (2 * np.pi * freq)

            # Create wavelet
            window_size = min(int(6 * sigma_t * self.sampling_rate), len(strain_data))
            if window_size < 10:
                continue

            t_window = np.linspace(-3*sigma_t, 3*sigma_t, window_size)
            wavelet = np.exp(1j * 2 * np.pi * freq * t_window) * \
                     np.exp(-t_window**2 / (2 * sigma_t**2))
            wavelet = wavelet / np.sqrt(np.sum(np.abs(wavelet)**2))

            # Convolve
            padding = len(strain_data) + len(wavelet) - 1
            strain_fft = np.fft.fft(strain_data, n=padding)
            wavelet_fft = np.fft.fft(np.flip(np.conj(wavelet)), n=padding)

            convolution = np.fft.ifft(strain_fft * wavelet_fft)
            convolution = convolution[:len(strain_data)]

            q_transform[i, :] = np.abs(convolution)**2

        # Normalize by expected noise level
        for i in range(n_freqs):
            noise_level = np.median(q_transform[i, :])
            if noise_level > 0:
                q_transform[i, :] = q_transform[i, :] / noise_level

        return q_transform

    def _detect_morphological_glitches(self, strain_data: np.ndarray,
                                     time_vector: np.ndarray,
                                     snr_threshold: float) -> List[GlitchCandidate]:
        """Detect glitches based on waveform morphology."""
        candidates = []

        # High-pass filter to remove low-frequency noise
        sos = signal.butter(8, 20, btype='highpass', fs=self.sampling_rate, output='sos')
        filtered_data = signal.sosfilt(sos, strain_data)

        # Calculate envelope using Hilbert transform
        analytic_signal = signal.hilbert(filtered_data)
        envelope = np.abs(analytic_signal)

        # Smooth envelope
        smoothed_envelope = ndimage.gaussian_filter1d(envelope, sigma=self.sampling_rate/100)

        # Find peaks in envelope
        threshold = snr_threshold * np.std(smoothed_envelope)
        peaks, properties = signal.find_peaks(
            smoothed_envelope,
            height=threshold,
            width=10,
            distance=int(0.1 * self.sampling_rate)  # Minimum 0.1s separation
        )

        for peak in peaks:
            peak_time = time_vector[peak]
            peak_value = smoothed_envelope[peak]

            # Estimate duration from peak width
            width_samples = properties['widths'][np.where(peaks == peak)[0][0]]
            duration = width_samples / self.sampling_rate

            # Simple morphology classification
            if peak_value > 10 * threshold:
                glitch_type = "high_amplitude"
            elif duration < 0.05:
                glitch_type = "short_transient"
            else:
                glitch_type = "generic_transient"

            candidates.append(GlitchCandidate(
                time=peak_time,
                duration=duration,
                frequency_range=(20, 1000),  # Full analysis band
                snr=peak_value / np.std(smoothed_envelope),
                glitch_type=glitch_type,
                confidence=min(1.0, peak_value / threshold),
                morphology_params={
                    'peak_amplitude': peak_value,
                    'envelope_snr': peak_value / np.std(smoothed_envelope)
                }
            ))

        return candidates

    def _detect_statistical_outliers(self, strain_data: np.ndarray,
                                   time_vector: np.ndarray,
                                   snr_threshold: float) -> List[GlitchCandidate]:
        """Detect glitches as statistical outliers."""
        candidates = []

        # Use sliding window to detect local outliers
        window_size = int(0.5 * self.sampling_rate)  # 0.5 second windows
        step_size = window_size // 4

        for i in range(0, len(strain_data) - window_size, step_size):
            window_data = strain_data[i:i+window_size]

            # Calculate z-scores
            median = np.median(window_data)
            mad = np.median(np.abs(window_data - median))

            if mad == 0:
                continue

            z_scores = (window_data - median) / (1.4826 * mad)  # MAD-based z-score

            # Find outliers
            outlier_mask = np.abs(z_scores) > snr_threshold
            outlier_indices = np.where(outlier_mask)[0]

            if len(outlier_indices) > 0:
                # Group consecutive outliers
                outlier_groups = self._group_consecutive_indices(outlier_indices)

                for group in outlier_groups:
                    if len(group) < 3:  # Skip single-sample outliers
                        continue

                    start_idx = i + group[0]
                    end_idx = i + group[-1]

                    start_time = time_vector[start_idx]
                    end_time = time_vector[end_idx]
                    duration = end_time - start_time

                    max_z = np.max(np.abs(z_scores[group]))

                    candidates.append(GlitchCandidate(
                        time=start_time + duration/2,
                        duration=duration,
                        frequency_range=(20, 1000),
                        snr=max_z,
                        glitch_type="statistical_outlier",
                        confidence=min(1.0, max_z / snr_threshold),
                        morphology_params={
                            'max_z_score': max_z,
                            'outlier_samples': len(group)
                        }
                    ))

        return candidates

    def _group_consecutive_indices(self, indices: np.ndarray) -> List[List[int]]:
        """Group consecutive indices into separate lists."""
        if len(indices) == 0:
            return []

        groups = []
        current_group = [indices[0]]

        for i in range(1, len(indices)):
            if indices[i] == indices[i-1] + 1:
                current_group.append(indices[i])
            else:
                groups.append(current_group)
                current_group = [indices[i]]

        groups.append(current_group)
        return groups

    def _cluster_glitch_candidates(self, candidates: List[GlitchCandidate]) -> List[GlitchCandidate]:
        """Cluster nearby glitch candidates to avoid duplicates."""
        if len(candidates) < 2:
            return candidates

        # Extract features for clustering
        features = []
        for candidate in candidates:
            features.append([
                candidate.time,
                candidate.duration,
                candidate.frequency_range[0],
                candidate.frequency_range[1]
            ])

        features = np.array(features)

        # Normalize features
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features)

        # DBSCAN clustering
        clustering = DBSCAN(eps=0.5, min_samples=1)
        cluster_labels = clustering.fit_predict(features_normalized)

        # Merge candidates in the same cluster
        clustered_candidates = []
        unique_labels = np.unique(cluster_labels)

        for label in unique_labels:
            cluster_mask = cluster_labels == label
            cluster_candidates = [candidates[i] for i in range(len(candidates)) if cluster_mask[i]]

            if len(cluster_candidates) == 1:
                clustered_candidates.append(cluster_candidates[0])
            else:
                # Merge candidates - keep the one with highest SNR
                best_candidate = max(cluster_candidates, key=lambda x: x.snr)
                clustered_candidates.append(best_candidate)

        return clustered_candidates

    def subtract_noise_artifacts(self, strain_data: np.ndarray,
                               glitch_candidates: List[GlitchCandidate],
                               method: str = 'inpainting') -> np.ndarray:
        """
        Subtract identified noise artifacts from strain data.

        Args:
            strain_data: Original strain data
            glitch_candidates: Detected glitches to subtract
            method: Subtraction method ('inpainting', 'gating', 'regression')

        Returns:
            Cleaned strain data
        """
        cleaned_data = strain_data.copy()

        for glitch in glitch_candidates:
            if glitch.confidence < 0.7:  # Only subtract high-confidence glitches
                continue

            # Find time indices for this glitch
            start_time = glitch.time - glitch.duration/2
            end_time = glitch.time + glitch.duration/2

            start_idx = int(start_time * self.sampling_rate)
            end_idx = int(end_time * self.sampling_rate)

            # Ensure indices are within bounds
            start_idx = max(0, start_idx)
            end_idx = min(len(cleaned_data), end_idx)

            if start_idx >= end_idx:
                continue

            if method == 'gating':
                # Simple gating - set to zero with tapering
                taper_samples = int(0.01 * self.sampling_rate)  # 10ms taper

                # Apply Tukey window for smooth gating
                gate_length = end_idx - start_idx
                if gate_length > 2 * taper_samples:
                    window = signal.tukey(gate_length, alpha=2*taper_samples/gate_length)
                    cleaned_data[start_idx:end_idx] *= (1 - window)
                else:
                    cleaned_data[start_idx:end_idx] = 0

            elif method == 'inpainting':
                # Linear interpolation across the glitch
                if start_idx > 0 and end_idx < len(cleaned_data):
                    x_points = [start_idx - 1, end_idx]
                    y_points = [cleaned_data[start_idx - 1], cleaned_data[end_idx]]

                    interp_func = interp1d(x_points, y_points, kind='linear')
                    interp_indices = np.arange(start_idx, end_idx)
                    cleaned_data[start_idx:end_idx] = interp_func(interp_indices)

            elif method == 'regression':
                # Autoregressive modeling for subtraction
                # Use surrounding data to predict glitch region
                context_length = int(0.1 * self.sampling_rate)  # 100ms context

                before_start = max(0, start_idx - context_length)
                after_end = min(len(cleaned_data), end_idx + context_length)

                # Simple AR model using surrounding data
                if before_start < start_idx and end_idx < after_end:
                    context_data = np.concatenate([
                        cleaned_data[before_start:start_idx],
                        cleaned_data[end_idx:after_end]
                    ])

                    # Fit AR model (simplified)
                    if len(context_data) > 10:
                        context_mean = np.mean(context_data)
                        context_std = np.std(context_data)

                        # Replace with noise matching surrounding statistics
                        replacement = np.random.normal(
                            context_mean, context_std, end_idx - start_idx
                        )
                        cleaned_data[start_idx:end_idx] = replacement

        return cleaned_data

    def generate_noise_report(self, strain_data: np.ndarray,
                            noise_chars: NoiseCharacteristics,
                            glitch_candidates: List[GlitchCandidate],
                            save_path: Optional[str] = None) -> Dict:
        """Generate comprehensive noise analysis report."""
        report = {
            'data_quality_summary': {
                'total_duration': len(strain_data) / self.sampling_rate,
                'stationarity_metric': noise_chars.stationarity_metric,
                'non_gaussianity_metric': noise_chars.non_gaussian_parameters['non_gaussianity_metric'],
                'glitch_rate': noise_chars.glitch_rate,
                'coherence_time': noise_chars.coherence_time
            },
            'spectral_analysis': {
                'frequency_range': (noise_chars.frequencies[0], noise_chars.frequencies[-1]),
                'spectral_lines_detected': len(noise_chars.spectral_lines),
                'dominant_line_frequencies': [line[0] for line in noise_chars.spectral_lines[:10]]
            },
            'glitch_analysis': {
                'total_glitches_detected': len(glitch_candidates),
                'glitch_types': {gtype: sum(1 for g in glitch_candidates if g.glitch_type == gtype)
                               for gtype in set(g.glitch_type for g in glitch_candidates)},
                'high_confidence_glitches': sum(1 for g in glitch_candidates if g.confidence > 0.8),
                'average_glitch_snr': np.mean([g.snr for g in glitch_candidates]) if glitch_candidates else 0
            },
            'recommendations': self._generate_recommendations(noise_chars, glitch_candidates)
        }

        if save_path:
            # Would save detailed plots and analysis
            pass

        return report

    def _generate_recommendations(self, noise_chars: NoiseCharacteristics,
                                glitch_candidates: List[GlitchCandidate]) -> List[str]:
        """Generate recommendations based on noise analysis."""
        recommendations = []

        if noise_chars.stationarity_metric < 0.7:
            recommendations.append("Data shows significant non-stationarity. Consider shorter analysis segments.")

        if noise_chars.non_gaussian_parameters['non_gaussianity_metric'] > 1.0:
            recommendations.append("Strong non-Gaussian behavior detected. Use robust statistical methods.")

        if noise_chars.glitch_rate > 1.0:  # More than 1 glitch per second
            recommendations.append("High glitch rate detected. Aggressive glitch mitigation recommended.")

        if len(noise_chars.spectral_lines) > 20:
            recommendations.append("Many spectral lines detected. Consider notch filtering.")

        high_snr_glitches = [g for g in glitch_candidates if g.snr > 20]
        if len(high_snr_glitches) > 0:
            recommendations.append("High-SNR glitches present. Manual inspection recommended.")

        if not recommendations:
            recommendations.append("Data quality appears good for GW analysis.")

        return recommendations
