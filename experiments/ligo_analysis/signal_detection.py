"""
Entanglement Signal Detection Module
===================================

Advanced signal detection algorithms specifically designed to identify
EG-QGEM entanglement signatures in gravitational wave data.
"""

import numpy as np
import scipy.signal as signal
from scipy.stats import chi2, norm
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Callable
import warnings
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging # Added import
import sys # Added import for logging handler

# Project imports
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from theory.constants import CONSTANTS


@dataclass
class DetectionCandidate:
    """Data class for detection candidates."""
    time: float
    frequency: float
    snr: float
    confidence: float
    entanglement_strength: float
    detection_statistic: float
    metadata: Dict[str, Any]


class EntanglementSignalDetector:
    """
    Advanced detector for entanglement-modified gravitational wave signatures.
    """

    def __init__(self,
                 sample_rate: float = 4096.0,
                 detection_threshold: float = 8.0):
        """
        Initialize the entanglement signal detector.

        Args:
            sample_rate: Data sampling rate in Hz
            detection_threshold: SNR threshold for detection
        """
        self.sample_rate = sample_rate
        self.detection_threshold = detection_threshold

        # Initialize logger
        self.logger = logging.getLogger(self.__class__.__name__)
        # Basic configuration if no handlers are set for the root logger
        # This helps ensure logs are visible during script execution if not configured elsewhere.
        if not logging.getLogger().hasHandlers(): # Check root logger
            # Configure this specific logger if the root isn't configured
            # Or, if you want to ensure this logger always outputs, configure it directly
            # For simplicity, let's ensure this logger has a handler if root doesn't.
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO) # Set level for this logger
            self.logger.propagate = False # Avoid duplicate messages if root logger also gets configured later


        # Detection algorithm parameters
        self.frequency_range = (20.0, 1024.0)  # Hz
        self.time_window = 1.0  # seconds
        self.frequency_resolution = 0.5  # Hz

        # Entanglement signature models
        self.signature_models = self._initialize_signature_models()

    def _initialize_signature_models(self) -> Dict[str, Callable]:
        """Initialize entanglement signature models."""
        return {
            'phase_modulation': self._phase_modulation_model,
            'amplitude_oscillation': self._amplitude_oscillation_model,
            'frequency_drift': self._frequency_drift_model,
            'polarization_rotation': self._polarization_rotation_model,
            'nonlinear_coupling': self._nonlinear_coupling_model
        }

    def detect_entanglement_signatures(self,
                                     strain_data: Dict[str, np.ndarray],
                                     noise_psd: Optional[np.ndarray] = None,
                                     detection_methods: Optional[List[str]] = None) -> List[DetectionCandidate]:
        """
        Main detection routine for entanglement signatures.

        Args:
            strain_data: Strain data dictionary
            noise_psd: Noise power spectral density
            detection_methods: List of detection methods to use

        Returns:
            List of detection candidates
        """
        if detection_methods is None:
            detection_methods = ['matched_filter', 'excess_power', 'coherent_waveburst', 'bayesian']

        self.logger.info(f"Starting entanglement signature detection with methods: {detection_methods}")
        strain = strain_data['strain']
        time = strain_data['time']
        self.logger.debug(f"Strain data length: {len(strain)}, Time data length: {len(time)}")

        all_candidates = []

        # Apply each detection method
        for method in detection_methods:
            self.logger.info(f"Applying detection method: {method}")
            try:
                if method == 'matched_filter':
                    candidates = self._matched_filter_detection(strain, time, noise_psd)
                elif method == 'excess_power':
                    candidates = self._excess_power_detection(strain, time)
                elif method == 'coherent_waveburst':
                    candidates = self._coherent_waveburst_detection(strain, time)
                elif method == 'bayesian':
                    candidates = self._bayesian_detection(strain, time, noise_psd)
                else:
                    warnings.warn(f"Unknown detection method: {method}")
                    continue

                # Add method information to candidates
                for candidate in candidates:
                    candidate.metadata['detection_method'] = method

                all_candidates.extend(candidates)
                self.logger.info(f"Method {method} found {len(candidates)} candidates. Total candidates so far: {len(all_candidates)}")

            except Exception as e:
                self.logger.error(f"Error in {method} detection: {e}", exc_info=True) # Log traceback
                continue

        self.logger.info(f"All detection methods applied. Total candidates before clustering: {len(all_candidates)}")
        # Cluster and rank candidates
        clustered_candidates = self._cluster_candidates(all_candidates)
        self.logger.info(f"Clustered candidates. Count: {len(clustered_candidates)}")
        ranked_candidates = self._rank_candidates(clustered_candidates)
        self.logger.info(f"Ranked candidates. Final count: {len(ranked_candidates)}")

        return ranked_candidates

    def _matched_filter_detection(self,
                                strain: np.ndarray,
                                time: np.ndarray,
                                noise_psd: Optional[np.ndarray] = None) -> List[DetectionCandidate]:
        """Matched filter detection using entanglement templates."""
        self.logger.info("Starting matched filter detection.")
        candidates = []

        self.logger.info("Generating entanglement templates...")
        templates = self._generate_entanglement_templates()
        self.logger.info(f"Generated {len(templates)} templates.")

        window_samples = int(self.time_window * self.sample_rate)
        hop_samples = window_samples // 4

        if len(strain) < window_samples:
            self.logger.warning(f"Strain length ({len(strain)}) is less than window_samples ({window_samples}). Skipping matched filter.")
            return []

        window_starts = list(range(0, len(strain) - window_samples + 1, hop_samples))
        total_windows_to_process = len(window_starts)

        self.logger.info(f"Strain length: {len(strain)}, Window samples: {window_samples}, Hop samples: {hop_samples}")
        self.logger.info(f"Analyzing {total_windows_to_process} windows with {len(templates)} templates each (Total correlations: {total_windows_to_process * len(templates)}).")

        for window_idx, i in enumerate(window_starts):
            if window_idx % (max(1, total_windows_to_process // 10)) == 0: # Log progress roughly every 10% of windows
                 self.logger.info(f"Processing window {window_idx + 1}/{total_windows_to_process} (start sample: {i})...")
            window_strain = strain[i:i+window_samples]
            window_time = time[i:i+window_samples]

            # Match against each template
            for template_idx, template in enumerate(templates):
                # self.logger.debug(f"  Window {window_idx + 1}: Matching template {template_idx+1}/{len(templates)} ({template.get('type', 'N/A')} f={template.get('frequency', 'N/A')})...")

                correlation = self._compute_correlation(window_strain, template, noise_psd)
                if not correlation.any(): # Check if correlation returned an empty array due to an issue
                    self.logger.warning(f"  Window {window_idx + 1}: Correlation for template {template_idx+1} resulted in empty array. Skipping.")
                    continue

                snr = np.max(correlation)

                if snr > self.detection_threshold:
                    max_idx = np.argmax(correlation)
                    detection_time = window_time[max_idx]
                    self.logger.info(f"  Candidate found in window {window_idx + 1} with template {template_idx+1}: SNR={snr:.2f} at time {detection_time:.2f}s")

                    candidate = DetectionCandidate(
                        time=detection_time,
                        frequency=template['frequency'],
                        snr=snr,
                        confidence=self._snr_to_confidence(snr),
                        entanglement_strength=template['coupling_strength'],
                        detection_statistic=snr,
                        metadata={
                            'template_id': template_id,
                            'template_type': template['type'],
                            'correlation_peak': correlation[max_idx],
                            'window_start': window_time[0]
                        }
                    )
                    candidates.append(candidate)

        self.logger.info("Matched filter detection finished.")
        return candidates

    def _excess_power_detection(self,
                              strain: np.ndarray,
                              time: np.ndarray) -> List[DetectionCandidate]:
        """Excess power detection for burst-like entanglement signatures."""
        candidates = []

        # Compute time-frequency representation
        f, t_spec, Sxx = signal.spectrogram(
            strain,
            fs=self.sample_rate,
            nperseg=int(0.5 * self.sample_rate),
            noverlap=int(0.25 * self.sample_rate),
            window='hann'
        )

        # Apply frequency range filter
        freq_mask = (f >= self.frequency_range[0]) & (f <= self.frequency_range[1])
        f_filtered = f[freq_mask]
        Sxx_filtered = Sxx[freq_mask, :]

        # Compute background power
        background_power = np.median(Sxx_filtered, axis=1, keepdims=True)
        excess_power = Sxx_filtered / background_power

        # Look for entanglement-specific patterns
        for signature_name, signature_model in self.signature_models.items():
            pattern_snr = signature_model(excess_power, f_filtered, t_spec)

            # Find peaks above threshold
            peak_indices = np.where(pattern_snr > self.detection_threshold)

            for j in range(len(peak_indices[0])):
                freq_idx = peak_indices[0][j]
                time_idx = peak_indices[1][j]

                detection_time = time[0] + t_spec[time_idx]
                snr = pattern_snr[freq_idx, time_idx]

                candidate = DetectionCandidate(
                    time=detection_time,
                    frequency=f_filtered[freq_idx],
                    snr=snr,
                    confidence=self._snr_to_confidence(snr),
                    entanglement_strength=self._estimate_coupling_strength(snr),
                    detection_statistic=snr,
                    metadata={
                        'signature_type': signature_name,
                        'frequency_bin': freq_idx,
                        'time_bin': time_idx,
                        'excess_power': excess_power[freq_idx, time_idx]
                    }
                )
                candidates.append(candidate)

        return candidates

    def _coherent_waveburst_detection(self,
                                    strain: np.ndarray,
                                    time: np.ndarray) -> List[DetectionCandidate]:
        """Coherent waveburst detection for entanglement signatures."""
        candidates = []

        # Multi-resolution analysis using wavelets
        scales = np.logspace(0, 3, 50)  # Frequency scales

        # Simplified continuous wavelet transform
        for scale in scales:
            frequency = self.sample_rate / (2 * scale)

            if frequency < self.frequency_range[0] or frequency > self.frequency_range[1]:
                continue

            # Morlet wavelet
            sigma = scale / 6
            wavelet_length = int(10 * sigma)
            t_wavelet = np.arange(-wavelet_length//2, wavelet_length//2) / self.sample_rate
            wavelet = np.exp(2j * np.pi * frequency * t_wavelet) * np.exp(-t_wavelet**2 / (2 * sigma**2))

            # Convolve with strain
            cwt_coeffs = np.convolve(strain, wavelet, mode='same')
            cwt_power = np.abs(cwt_coeffs)**2

            # Threshold detection
            threshold = np.percentile(cwt_power, 99.9)  # 99.9th percentile

            peak_indices = signal.find_peaks(cwt_power, height=threshold)[0]

            for peak_idx in peak_indices:
                if peak_idx < len(time):
                    snr = np.sqrt(cwt_power[peak_idx] / np.median(cwt_power))

                    if snr > self.detection_threshold:
                        candidate = DetectionCandidate(
                            time=time[peak_idx],
                            frequency=frequency,
                            snr=snr,
                            confidence=self._snr_to_confidence(snr),
                            entanglement_strength=self._estimate_coupling_strength(snr),
                            detection_statistic=cwt_power[peak_idx],
                            metadata={
                                'scale': scale,
                                'wavelet_coefficient': cwt_coeffs[peak_idx],
                                'power': cwt_power[peak_idx],
                                'threshold': threshold
                            }
                        )
                        candidates.append(candidate)

        return candidates

    def _bayesian_detection(self,
                          strain: np.ndarray,
                          time: np.ndarray,
                          noise_psd: Optional[np.ndarray] = None) -> List[DetectionCandidate]:
        """Bayesian detection using model comparison."""
        candidates = []

        # Segment data for analysis
        segment_duration = 4.0  # seconds
        segment_samples = int(segment_duration * self.sample_rate)

        for i in range(0, len(strain) - segment_samples, segment_samples//2):
            segment_strain = strain[i:i+segment_samples]
            segment_time = time[i:i+segment_samples]

            # Compute Bayesian evidence for different models
            noise_evidence = self._compute_noise_evidence(segment_strain, noise_psd)
            signal_evidence = self._compute_signal_evidence(segment_strain, noise_psd)

            # Bayes factor
            bayes_factor = signal_evidence / noise_evidence if noise_evidence > 0 else 0

            # Convert to detection statistic
            if bayes_factor > 1.0:
                detection_stat = np.log(bayes_factor)
                snr = np.sqrt(2 * detection_stat)  # Approximate conversion

                if snr > self.detection_threshold:
                    # Estimate parameters
                    central_time = segment_time[len(segment_time)//2]
                    estimated_freq = self._estimate_central_frequency(segment_strain)

                    candidate = DetectionCandidate(
                        time=central_time,
                        frequency=estimated_freq,
                        snr=snr,
                        confidence=self._bayes_factor_to_confidence(bayes_factor),
                        entanglement_strength=self._estimate_coupling_strength(snr),
                        detection_statistic=detection_stat,
                        metadata={
                            'bayes_factor': bayes_factor,
                            'noise_evidence': noise_evidence,
                            'signal_evidence': signal_evidence,
                            'segment_duration': segment_duration
                        }
                    )
                    candidates.append(candidate)

        return candidates

    def _generate_entanglement_templates(self) -> List[Dict[str, Any]]:
        """Generate template bank for entanglement signatures."""
        templates = []

        # Parameter ranges
        coupling_strengths = np.logspace(-6, -3, 10)
        frequencies = np.logspace(np.log10(50), np.log10(500), 20)

        for coupling in coupling_strengths:
            for freq in frequencies:
                # Phase modulation template
                template = self._create_phase_modulation_template(coupling, freq)
                templates.append(template)

                # Amplitude modulation template
                template = self._create_amplitude_modulation_template(coupling, freq)
                templates.append(template)

        return templates

    def _create_phase_modulation_template(self,
                                        coupling: float,
                                        frequency: float) -> Dict[str, Any]:
        """Create phase modulation template."""
        duration = self.time_window
        t = np.arange(0, duration, 1/self.sample_rate)

        # Basic carrier wave
        carrier = np.cos(2*np.pi*frequency*t)

        # Entanglement-induced phase modulation
        modulation_freq = frequency / 10  # Modulation frequency
        phase_mod = coupling * np.sin(2*np.pi*modulation_freq*t)

        template_signal = np.cos(2*np.pi*frequency*t + phase_mod)

        return {
            'signal': template_signal,
            'time': t,
            'frequency': frequency,
            'coupling_strength': coupling,
            'type': 'phase_modulation',
            'modulation_frequency': modulation_freq
        }

    def _create_amplitude_modulation_template(self,
                                            coupling: float,
                                            frequency: float) -> Dict[str, Any]:
        """Create amplitude modulation template."""
        duration = self.time_window
        t = np.arange(0, duration, 1/self.sample_rate)

        # Basic carrier wave with amplitude modulation
        modulation_freq = frequency / 20
        amplitude = 1 + coupling * np.sin(2*np.pi*modulation_freq*t)

        template_signal = amplitude * np.cos(2*np.pi*frequency*t)

        return {
            'signal': template_signal,
            'time': t,
            'frequency': frequency,
            'coupling_strength': coupling,
            'type': 'amplitude_modulation',
            'modulation_frequency': modulation_freq
        }

    def _compute_correlation(self,
                           data: np.ndarray,
                           template: Dict[str, Any],
                           noise_psd: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute correlation between data and template."""
        template_signal = template['signal']

        # Ensure data and template_signal are numpy arrays
        if not isinstance(data, np.ndarray):
            self.logger.warning(f"Data input to _compute_correlation is not a numpy array, type: {type(data)}. Attempting conversion.")
            data = np.array(data, dtype=float)
        if not isinstance(template_signal, np.ndarray):
            self.logger.warning(f"Template signal input to _compute_correlation is not a numpy array, type: {type(template_signal)}. Attempting conversion.")
            template_signal = np.array(template_signal, dtype=float)

        # Ensure they are 1D
        if data.ndim != 1:
            self.logger.warning(f"Data input is not 1D (shape: {data.shape}). Flattening.")
            data = data.flatten()
        if template_signal.ndim != 1:
            self.logger.warning(f"Template signal is not 1D (shape: {template_signal.shape}). Flattening.")
            template_signal = template_signal.flatten()

        # Ensure non-empty
        if len(data) == 0 or len(template_signal) == 0:
            self.logger.error("Empty data or template signal in _compute_correlation. Returning empty array.")
            return np.array([])

        # Ensure same length
        min_length = min(len(data), len(template_signal))
        if min_length == 0: # Should be caught by above, but as a safeguard
            self.logger.error("Zero min_length after attempting to get data and template. Returning empty array.")
            return np.array([])

        data_truncated = data[:min_length]
        template_truncated = template_signal[:min_length]

        # Check for non-finite values
        if not np.all(np.isfinite(data_truncated)):
            self.logger.error("Non-finite values (NaN or inf) found in data_truncated. Returning empty array.")
            # Optionally, you could try to clean the data, e.g., np.nan_to_num(data_truncated)
            return np.array([])
        if not np.all(np.isfinite(template_truncated)):
            self.logger.error("Non-finite values (NaN or inf) found in template_truncated. Returning empty array.")
            # Optionally, np.nan_to_num(template_truncated)
            return np.array([])

        if noise_psd is None:
            # Simple correlation without noise weighting
            correlation = signal.correlate(data_truncated, template_truncated, mode='same')
            normalization = np.sqrt(np.sum(template_truncated**2))
        else:
            # Noise-weighted correlation (frequency domain)
            # This is a simplified implementation
            correlation = signal.correlate(data_truncated, template_truncated, mode='same')
            normalization = np.sqrt(np.sum(template_truncated**2))

        if normalization > 0:
            correlation = correlation / normalization

        return correlation

    def _phase_modulation_model(self,
                              power_map: np.ndarray,
                              frequencies: np.ndarray,
                              times: np.ndarray) -> np.ndarray:
        """Model for phase modulation signatures."""
        # Look for oscillatory patterns in phase
        phase_map = np.angle(signal.hilbert(power_map, axis=1))

        # Compute phase derivative (instantaneous frequency)
        freq_dev = np.gradient(phase_map, axis=1)

        # Look for sinusoidal modulation in frequency
        modulation_pattern = np.abs(np.fft.fft(freq_dev, axis=1))

        return np.mean(modulation_pattern, axis=1, keepdims=True) * np.ones_like(power_map)

    def _amplitude_oscillation_model(self,
                                   power_map: np.ndarray,
                                   frequencies: np.ndarray,
                                   times: np.ndarray) -> np.ndarray:
        """Model for amplitude oscillation signatures."""
        # Look for periodic amplitude variations
        amplitude_variation = np.std(power_map, axis=1, keepdims=True)
        oscillation_strength = amplitude_variation / np.mean(power_map, axis=1, keepdims=True)

        return oscillation_strength * np.ones_like(power_map)

    def _frequency_drift_model(self,
                             power_map: np.ndarray,
                             frequencies: np.ndarray,
                             times: np.ndarray) -> np.ndarray:
        """Model for frequency drift signatures."""
        # Compute centroid frequency as function of time
        freq_centroid = np.sum(power_map * frequencies[:, np.newaxis], axis=0) / np.sum(power_map, axis=0)

        # Look for systematic drifts
        drift_rate = np.gradient(freq_centroid)
        drift_strength = np.abs(drift_rate)

        return np.outer(np.ones(len(frequencies)), drift_strength)

    def _polarization_rotation_model(self,
                                   power_map: np.ndarray,
                                   frequencies: np.ndarray,
                                   times: np.ndarray) -> np.ndarray:
        """Model for polarization rotation signatures."""
        # This would require analysis of both polarizations
        # Simplified implementation
        return 0.1 * np.ones_like(power_map)

    def _nonlinear_coupling_model(self,
                                power_map: np.ndarray,
                                frequencies: np.ndarray,
                                times: np.ndarray) -> np.ndarray:
        """Model for nonlinear coupling signatures."""
        # Look for harmonic content
        harmonic_strength = np.zeros_like(power_map)

        for i, f in enumerate(frequencies):
            # Look for power at 2f, 3f, etc.
            harmonic_freqs = [2*f, 3*f]

            for h_freq in harmonic_freqs:
                # Find closest frequency bin
                freq_idx = np.argmin(np.abs(frequencies - h_freq))
                if freq_idx < len(frequencies):
                    harmonic_strength[i, :] += power_map[freq_idx, :]

        return harmonic_strength

    def _cluster_candidates(self, candidates: List[DetectionCandidate]) -> List[DetectionCandidate]:
        """Cluster nearby candidates to remove duplicates."""
        if not candidates:
            return []

        clustered = []
        sorted_candidates = sorted(candidates, key=lambda x: x.snr, reverse=True)

        time_threshold = 0.1  # seconds
        freq_threshold = 10.0  # Hz

        for candidate in sorted_candidates:
            # Check if this candidate is close to an existing cluster
            is_duplicate = False

            for existing in clustered:
                time_diff = abs(candidate.time - existing.time)
                freq_diff = abs(candidate.frequency - existing.frequency)

                if time_diff < time_threshold and freq_diff < freq_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                clustered.append(candidate)

        return clustered

    def _rank_candidates(self, candidates: List[DetectionCandidate]) -> List[DetectionCandidate]:
        """Rank candidates by detection significance."""
        # Combine SNR and confidence for ranking
        for candidate in candidates:
            candidate.metadata['ranking_score'] = candidate.snr * candidate.confidence

        return sorted(candidates, key=lambda x: x.metadata['ranking_score'], reverse=True)

    def _snr_to_confidence(self, snr: float) -> float:
        """Convert SNR to confidence level."""
        # Assuming Gaussian noise, convert to p-value and then confidence
        p_value = 2 * (1 - norm.cdf(snr))  # Two-sided test
        confidence = 1 - p_value
        return np.clip(confidence, 0.0, 1.0)

    def _bayes_factor_to_confidence(self, bayes_factor: float) -> float:
        """Convert Bayes factor to confidence level."""
        # Simple mapping from Bayes factor to confidence
        confidence = bayes_factor / (1 + bayes_factor)
        return np.clip(confidence, 0.0, 1.0)

    def _estimate_coupling_strength(self, snr: float) -> float:
        """Estimate entanglement coupling strength from SNR."""
        # Empirical relationship (to be calibrated with simulations)
        coupling = 1e-5 * (snr / 10)**2
        return np.clip(coupling, 1e-8, 1e-2)

    def _compute_noise_evidence(self,
                              data: np.ndarray,
                              noise_psd: Optional[np.ndarray] = None) -> float:
        """Compute Bayesian evidence for noise-only model."""
        # Simplified implementation
        if noise_psd is None:
            noise_variance = np.var(data)
        else:
            noise_variance = np.mean(noise_psd)

        # Gaussian likelihood
        log_evidence = -0.5 * len(data) * np.log(2*np.pi*noise_variance) - 0.5 * np.sum(data**2) / noise_variance
        return np.exp(log_evidence)

    def _compute_signal_evidence(self,
                               data: np.ndarray,
                               noise_psd: Optional[np.ndarray] = None) -> float:
        """Compute Bayesian evidence for signal+noise model."""
        # This would involve marginalization over signal parameters
        # Simplified implementation using maximum likelihood

        # Assume some signal is present and compute likelihood
        signal_variance = np.var(data) * 1.1  # Slightly higher than noise

        log_evidence = -0.5 * len(data) * np.log(2*np.pi*signal_variance) - 0.5 * np.sum(data**2) / signal_variance

        # Add prior penalty for model complexity
        log_evidence -= 5.0  # Penalty for additional parameters

        return np.exp(log_evidence)

    def _estimate_central_frequency(self, data: np.ndarray) -> float:
        """Estimate central frequency of a signal."""
        # Compute power spectral density
        freqs, psd = signal.welch(data, fs=self.sample_rate)

        # Find peak frequency
        peak_idx = np.argmax(psd)
        return freqs[peak_idx]

    def save_detection_results(self,
                             candidates: List[DetectionCandidate],
                             filename: str) -> None:
        """Save detection results to file."""
        results = {
            'detection_summary': {
                'n_candidates': len(candidates),
                'detection_threshold': self.detection_threshold,
                'frequency_range': self.frequency_range,
                'analysis_time': str(np.datetime64('now'))
            },
            'candidates': []
        }

        for i, candidate in enumerate(candidates):
            candidate_dict = {
                'candidate_id': i,
                'time': float(candidate.time),
                'frequency': float(candidate.frequency),
                'snr': float(candidate.snr),
                'confidence': float(candidate.confidence),
                'entanglement_strength': float(candidate.entanglement_strength),
                'detection_statistic': float(candidate.detection_statistic),
                'metadata': candidate.metadata
            }
            results['candidates'].append(candidate_dict)

        import json
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
