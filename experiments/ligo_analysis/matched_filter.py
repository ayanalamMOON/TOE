"""
Entanglement-Optimized Matched Filtering for LIGO Analysis
==========================================================

This module provides matched filtering algorithms specifically optimized
for detecting EG-QGEM signatures in gravitational wave data, including
template banks and coherent multi-detector analysis.
"""

import numpy as np
import scipy.signal as signal
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from theory.constants import CONSTANTS
from theory.entanglement_tensor import EntanglementTensor
from experiments.ligo_analysis.waveform_analysis import EGQGEMWaveformAnalyzer


@dataclass
class MatchedFilterResult:
    """Result from matched filtering analysis."""
    snr_time_series: np.ndarray
    peak_snr: float
    peak_time: float
    template_params: Dict[str, float]
    chi_squared: float
    detection_statistic: float
    coherent_snr: Optional[float] = None
    network_snr: Optional[float] = None


@dataclass
class TemplateBank:
    """Template bank for matched filtering."""
    templates: List[Dict[str, np.ndarray]]
    parameter_space: List[Dict[str, float]]
    frequencies: np.ndarray
    template_spacing: Dict[str, float]
    coverage_metric: float


class EntanglementMatchedFilter:
    """
    Matched filtering optimized for EG-QGEM gravitational wave signatures.

    Implements both single-detector and coherent multi-detector matched
    filtering with templates incorporating entanglement modifications.
    """

    def __init__(self,
                 detector_network: List[str] = ['H1', 'L1'],
                 sampling_rate: float = 4096):
        """
        Initialize matched filter.

        Args:
            detector_network: List of detector names
            sampling_rate: Data sampling rate in Hz
        """
        self.detector_network = detector_network
        self.sampling_rate = sampling_rate
        self.constants = CONSTANTS
        self.entanglement_tensor = EntanglementTensor()
        self.waveform_analyzer = EGQGEMWaveformAnalyzer()

        # Frequency range for analysis
        self.f_min = 20.0  # Hz
        self.f_max = 1000.0  # Hz

        # Template bank parameters
        self.minimal_match = 0.97  # Minimum overlap between neighboring templates

        # Detection thresholds
        self.snr_threshold = 8.0
        self.chi_squared_threshold = 10.0

    def generate_template_bank(self,
                             parameter_ranges: Dict[str, Tuple[float, float]],
                             egqgem_enabled: bool = True,
                             max_templates: int = 10000) -> TemplateBank:
        """
        Generate optimal template bank for EG-QGEM signals.

        Args:
            parameter_ranges: Parameter ranges for template placement
            egqgem_enabled: Whether to include EG-QGEM modifications
            max_templates: Maximum number of templates

        Returns:
            Generated template bank
        """
        print("Generating EG-QGEM template bank...")

        # Calculate metric for parameter space
        metric_tensor = self._calculate_parameter_metric(parameter_ranges)

        # Generate template placement using stochastic placement
        template_params = self._stochastic_template_placement(
            parameter_ranges, metric_tensor, max_templates
        )

        # Generate frequency array
        frequencies = np.linspace(self.f_min, self.f_max, 1024)

        # Generate templates
        templates = []

        print(f"Generating {len(template_params)} templates...")

        # Use multiprocessing for template generation
        with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
            futures = []

            for params in template_params:
                future = executor.submit(
                    self._generate_single_template,
                    params, frequencies, egqgem_enabled
                )
                futures.append(future)

            # Collect results
            for i, future in enumerate(futures):
                try:
                    template = future.result()
                    templates.append(template)

                    if (i + 1) % 100 == 0:
                        print(f"Generated {i + 1}/{len(template_params)} templates")

                except Exception as e:
                    print(f"Failed to generate template {i}: {e}")
                    continue

        # Calculate coverage metric
        coverage_metric = self._calculate_coverage_metric(templates, template_params)

        # Calculate template spacing
        template_spacing = self._calculate_template_spacing(template_params)

        return TemplateBank(
            templates=templates,
            parameter_space=template_params,
            frequencies=frequencies,
            template_spacing=template_spacing,
            coverage_metric=coverage_metric
        )

    def _calculate_parameter_metric(self,
                                  parameter_ranges: Dict[str, Tuple[float, float]]) -> np.ndarray:
        """Calculate Fisher information metric for parameter space."""
        # Simplified metric calculation - would need full Fisher matrix computation
        param_names = list(parameter_ranges.keys())
        n_params = len(param_names)

        # Create identity metric as approximation
        metric = np.eye(n_params)

        # Scale by parameter ranges (larger ranges need finer spacing)
        for i, param in enumerate(param_names):
            range_size = parameter_ranges[param][1] - parameter_ranges[param][0]
            metric[i, i] = 1.0 / range_size**2

        return metric

    def _stochastic_template_placement(self,
                                     parameter_ranges: Dict[str, Tuple[float, float]],
                                     metric_tensor: np.ndarray,
                                     max_templates: int) -> List[Dict[str, float]]:
        """Place templates using stochastic bank generation."""
        param_names = list(parameter_ranges.keys())
        templates = []

        # Start with random seed template
        seed_template = {}
        for param in param_names:
            min_val, max_val = parameter_ranges[param]
            seed_template[param] = np.random.uniform(min_val, max_val)

        templates.append(seed_template)

        # Iteratively add templates
        for _ in range(max_templates - 1):
            # Propose new template
            proposal = {}
            for param in param_names:
                min_val, max_val = parameter_ranges[param]
                proposal[param] = np.random.uniform(min_val, max_val)

            # Check if proposal is sufficiently separated
            min_overlap = 1.0
            for existing_template in templates:
                overlap = self._calculate_template_overlap(
                    proposal, existing_template, metric_tensor, param_names
                )
                min_overlap = min(min_overlap, overlap)

            # Accept if overlap is below threshold
            if min_overlap < self.minimal_match:
                templates.append(proposal)

            # Early termination if we're placing too many rejected templates
            if len(templates) < (_ + 1) // 100:
                break

        return templates

    def _calculate_template_overlap(self,
                                  template1: Dict[str, float],
                                  template2: Dict[str, float],
                                  metric_tensor: np.ndarray,
                                  param_names: List[str]) -> float:
        """Calculate overlap between two templates using Fisher metric."""
        # Convert parameter differences to vector
        diff_vector = np.array([
            template1[param] - template2[param] for param in param_names
        ])

        # Calculate metric distance
        metric_distance = np.sqrt(diff_vector.T @ metric_tensor @ diff_vector)

        # Convert to overlap (exponential approximation)
        overlap = np.exp(-0.5 * metric_distance**2)

        return overlap

    def _generate_single_template(self,
                                params: Dict[str, float],
                                frequencies: np.ndarray,
                                egqgem_enabled: bool) -> Dict[str, np.ndarray]:
        """Generate a single template waveform."""
        if egqgem_enabled:
            # Generate EG-QGEM modified template
            h_plus, h_cross = self.waveform_analyzer.generate_inspiral_waveform(
                mass1=params.get('mass1', 30.0),
                mass2=params.get('mass2', 25.0),
                spin1z=params.get('spin1z', 0.0),
                spin2z=params.get('spin2z', 0.0),
                distance=params.get('distance', 400.0),
                inclination=params.get('inclination', np.pi/3),
                entanglement_coupling=params.get('entanglement_coupling', 1e-4),
                frequencies=frequencies,
                reference_phase=0.0
            )
        else:
            # Generate standard GR template
            h_plus, h_cross = self._generate_gr_template(params, frequencies)

        # Normalize templates
        h_plus_norm = h_plus / np.sqrt(np.sum(np.abs(h_plus)**2))
        h_cross_norm = h_cross / np.sqrt(np.sum(np.abs(h_cross)**2))

        return {
            'h_plus': h_plus_norm,
            'h_cross': h_cross_norm,
            'parameters': params.copy()
        }

    def _generate_gr_template(self,
                            params: Dict[str, float],
                            frequencies: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate standard GR template for comparison."""
        # Simplified GR template generation
        mass1 = params.get('mass1', 30.0)
        mass2 = params.get('mass2', 25.0)
        distance = params.get('distance', 400.0)
        inclination = params.get('inclination', np.pi/3)

        total_mass = mass1 + mass2
        chirp_mass = (mass1 * mass2)**(3/5) / total_mass**(1/5)
        eta = (mass1 * mass2) / total_mass**2

        # Post-Newtonian amplitude and phase
        M_geo = chirp_mass * self.constants.SOLAR_MASS_GEO
        D_geo = distance * 1e6 * self.constants.PARSEC_GEO

        f_geo = frequencies * 2 * np.pi * M_geo

        # Amplitude
        amplitude = np.sqrt(5*np.pi/24) * M_geo**(5/6) / (D_geo * f_geo**(7/6))

        # Phase (2PN approximation)
        phase = (3/(128*eta)) * f_geo**(-5/3) * (
            1 + (3715/756 + 55*eta/9) * f_geo**(2/3) +
            (-16*np.pi) * f_geo +
            (15293365/508032 + 27145*eta/504 + 3085*eta**2/72) * f_geo**(4/3)
        )

        # Polarizations
        F_plus = 0.5 * (1 + np.cos(inclination)**2)
        F_cross = np.cos(inclination)

        h_plus = amplitude * F_plus * np.exp(1j * phase)
        h_cross = amplitude * F_cross * np.exp(1j * (phase + np.pi/2))

        return h_plus, h_cross

    def _calculate_coverage_metric(self,
                                 templates: List[Dict[str, np.ndarray]],
                                 template_params: List[Dict[str, float]]) -> float:
        """Calculate template bank coverage metric."""
        if len(templates) < 2:
            return 0.0

        # Calculate average template spacing
        total_overlaps = 0
        overlap_count = 0

        for i in range(len(templates)):
            for j in range(i + 1, min(i + 10, len(templates))):  # Check nearest neighbors
                # Calculate waveform overlap
                h1_plus = templates[i]['h_plus']
                h2_plus = templates[j]['h_plus']

                overlap = np.abs(np.sum(np.conj(h1_plus) * h2_plus))**2
                overlap /= (np.sum(np.abs(h1_plus)**2) * np.sum(np.abs(h2_plus)**2))

                total_overlaps += overlap
                overlap_count += 1

        if overlap_count == 0:
            return 0.0

        average_overlap = total_overlaps / overlap_count

        # Coverage metric: closer to minimal_match is better
        coverage_metric = 1.0 - abs(average_overlap - self.minimal_match)

        return coverage_metric

    def _calculate_template_spacing(self,
                                  template_params: List[Dict[str, float]]) -> Dict[str, float]:
        """Calculate average template spacing in each parameter."""
        if len(template_params) < 2:
            return {}

        param_names = list(template_params[0].keys())
        spacing = {}

        for param in param_names:
            values = [tp[param] for tp in template_params]
            values.sort()

            if len(values) > 1:
                differences = [values[i+1] - values[i] for i in range(len(values)-1)]
                spacing[param] = np.mean(differences)
            else:
                spacing[param] = 0.0

        return spacing

    def matched_filter_search(self,
                            strain_data: Dict[str, np.ndarray],
                            noise_psd: Dict[str, np.ndarray],
                            template_bank: TemplateBank,
                            time_vector: Optional[np.ndarray] = None) -> List[MatchedFilterResult]:
        """
        Perform matched filter search over template bank.

        Args:
            strain_data: Strain data for each detector
            noise_psd: Noise PSD for each detector
            template_bank: Template bank to search over
            time_vector: Time vector for data

        Returns:
            List of matched filter results above threshold
        """
        if time_vector is None:
            time_vector = np.arange(len(list(strain_data.values())[0])) / self.sampling_rate

        results = []

        print(f"Searching over {len(template_bank.templates)} templates...")

        for i, template in enumerate(template_bank.templates):
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(template_bank.templates)} templates")

            # Perform matched filtering for each detector
            detector_results = {}

            for detector in self.detector_network:
                if detector not in strain_data or detector not in noise_psd:
                    continue

                # Calculate matched filter SNR
                snr_ts = self._calculate_snr_time_series(
                    strain_data[detector],
                    template['h_plus'],  # Use plus polarization as reference
                    noise_psd[detector],
                    template_bank.frequencies
                )

                detector_results[detector] = snr_ts

            if not detector_results:
                continue

            # Combine results from multiple detectors
            combined_result = self._combine_detector_results(
                detector_results, template['parameters'], time_vector
            )

            # Check if above threshold
            if combined_result.peak_snr > self.snr_threshold:
                # Calculate chi-squared test
                combined_result.chi_squared = self._calculate_chi_squared(
                    strain_data, template, combined_result.peak_time,
                    noise_psd, template_bank.frequencies
                )

                # Calculate detection statistic
                combined_result.detection_statistic = self._calculate_detection_statistic(
                    combined_result.peak_snr, combined_result.chi_squared
                )

                results.append(combined_result)

        # Sort by detection statistic
        results.sort(key=lambda x: x.detection_statistic, reverse=True)

        return results

    def _calculate_snr_time_series(self,
                                 strain_data: np.ndarray,
                                 template: np.ndarray,
                                 noise_psd: np.ndarray,
                                 frequencies: np.ndarray) -> np.ndarray:
        """Calculate SNR time series using matched filtering."""
        # Convert to frequency domain
        strain_fft = np.fft.fft(strain_data)
        template_fft = np.fft.fft(template, n=len(strain_data))

        # Interpolate PSD to match frequency resolution
        freq_data = np.fft.fftfreq(len(strain_data), 1/self.sampling_rate)

        # Only use positive frequencies
        positive_freqs = freq_data > 0
        freq_data_pos = freq_data[positive_freqs]

        # Interpolate noise PSD
        psd_interp = interp1d(
            frequencies, noise_psd,
            kind='linear', bounds_error=False, fill_value=np.inf
        )(freq_data_pos)

        # Construct full PSD array (symmetric)
        psd_full = np.ones(len(strain_data)) * np.inf
        psd_full[1:len(freq_data_pos)+1] = psd_interp
        psd_full[-len(freq_data_pos):] = psd_interp[::-1]
        psd_full[0] = np.inf  # DC component

        # Matched filter in frequency domain
        optimal_filter = np.conj(template_fft) / psd_full

        # Avoid division by zero
        optimal_filter[psd_full == np.inf] = 0

        # Calculate SNR time series
        snr_fft = strain_fft * optimal_filter
        snr_ts = 2 * np.fft.ifft(snr_fft).real  # Factor of 2 for one-sided PSD

        # Normalize by template norm
        template_norm = np.sqrt(2 * np.sum(
            np.abs(template_fft)**2 / psd_full *
            (psd_full != np.inf)
        ) / len(strain_data))

        if template_norm > 0:
            snr_ts /= template_norm

        return snr_ts

    def _combine_detector_results(self,
                                detector_results: Dict[str, np.ndarray],
                                template_params: Dict[str, float],
                                time_vector: np.ndarray) -> MatchedFilterResult:
        """Combine matched filter results from multiple detectors."""
        # Simple coherent combination - sum SNR time series
        combined_snr = np.zeros(len(list(detector_results.values())[0]))

        for detector, snr_ts in detector_results.items():
            combined_snr += snr_ts**2

        combined_snr = np.sqrt(combined_snr)

        # Find peak
        peak_idx = np.argmax(combined_snr)
        peak_snr = combined_snr[peak_idx]
        peak_time = time_vector[peak_idx]

        # Calculate network SNR
        network_snr = np.sqrt(sum(
            np.max(snr_ts)**2 for snr_ts in detector_results.values()
        ))

        return MatchedFilterResult(
            snr_time_series=combined_snr,
            peak_snr=peak_snr,
            peak_time=peak_time,
            template_params=template_params.copy(),
            chi_squared=0.0,  # Will be calculated later
            detection_statistic=0.0,  # Will be calculated later
            coherent_snr=peak_snr,
            network_snr=network_snr
        )

    def _calculate_chi_squared(self,
                             strain_data: Dict[str, np.ndarray],
                             template: Dict[str, np.ndarray],
                             peak_time: float,
                             noise_psd: Dict[str, np.ndarray],
                             frequencies: np.ndarray) -> float:
        """Calculate chi-squared goodness-of-fit test."""
        # Simplified chi-squared calculation
        # In practice, would use more sophisticated signal consistency tests

        total_chi_squared = 0.0
        dof = 0

        for detector in self.detector_network:
            if detector not in strain_data:
                continue

            # Extract data around peak time
            peak_idx = int(peak_time * self.sampling_rate)
            window_size = int(0.1 * self.sampling_rate)  # 100ms window

            start_idx = max(0, peak_idx - window_size // 2)
            end_idx = min(len(strain_data[detector]), peak_idx + window_size // 2)

            data_segment = strain_data[detector][start_idx:end_idx]

            # Generate template segment
            template_segment = template['h_plus'][:len(data_segment)]

            # Scale template to match data
            template_fft = np.fft.fft(template_segment)
            data_fft = np.fft.fft(data_segment)

            # Simple scaling using cross-correlation
            cross_corr = np.sum(np.conj(template_fft) * data_fft)
            template_norm = np.sum(np.abs(template_fft)**2)

            if template_norm > 0:
                scale_factor = np.real(cross_corr) / template_norm
                scaled_template = scale_factor * template_segment
            else:
                scaled_template = np.zeros_like(template_segment)

            # Calculate residual
            residual = data_segment - scaled_template

            # Chi-squared contribution
            chi_squared_contrib = np.sum(residual**2)
            total_chi_squared += chi_squared_contrib
            dof += len(residual)

        # Normalize by degrees of freedom
        if dof > 0:
            return total_chi_squared / dof
        else:
            return np.inf

    def _calculate_detection_statistic(self, snr: float, chi_squared: float) -> float:
        """Calculate combined detection statistic."""
        # NewSNR-like statistic that penalizes poor chi-squared
        if chi_squared <= 1.0:
            return snr
        elif chi_squared < self.chi_squared_threshold:
            # Gradual penalty
            penalty = 1.0 - 0.5 * (chi_squared - 1.0) / (self.chi_squared_threshold - 1.0)
            return snr * penalty
        else:
            # Strong penalty for very poor fits
            return snr * 0.1

    def coherent_multi_detector_search(self,
                                     strain_data: Dict[str, np.ndarray],
                                     noise_psd: Dict[str, np.ndarray],
                                     template_bank: TemplateBank,
                                     time_delays: Optional[Dict[str, float]] = None) -> List[MatchedFilterResult]:
        """
        Perform coherent multi-detector matched filter search.

        Args:
            strain_data: Strain data for each detector
            noise_psd: Noise PSD for each detector
            template_bank: Template bank
            time_delays: Time delays between detectors

        Returns:
            Coherent search results
        """
        if time_delays is None:
            time_delays = {detector: 0.0 for detector in self.detector_network}

        results = []

        print("Performing coherent multi-detector search...")

        for i, template in enumerate(template_bank.templates):
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(template_bank.templates)} templates")

            # Calculate SNR for each detector with time delays
            detector_snrs = {}

            for detector in self.detector_network:
                if detector not in strain_data:
                    continue

                # Apply time delay
                delay_samples = int(time_delays[detector] * self.sampling_rate)

                if delay_samples != 0:
                    delayed_data = np.roll(strain_data[detector], delay_samples)
                else:
                    delayed_data = strain_data[detector]

                # Calculate SNR
                snr_ts = self._calculate_snr_time_series(
                    delayed_data,
                    template['h_plus'],
                    noise_psd[detector],
                    template_bank.frequencies
                )

                detector_snrs[detector] = snr_ts

            # Coherent combination
            if len(detector_snrs) >= 2:
                coherent_snr = self._coherent_combination(detector_snrs)

                peak_idx = np.argmax(coherent_snr)
                peak_snr = coherent_snr[peak_idx]

                if peak_snr > self.snr_threshold:
                    peak_time = peak_idx / self.sampling_rate

                    result = MatchedFilterResult(
                        snr_time_series=coherent_snr,
                        peak_snr=peak_snr,
                        peak_time=peak_time,
                        template_params=template['parameters'].copy(),
                        chi_squared=0.0,
                        detection_statistic=peak_snr,
                        coherent_snr=peak_snr
                    )

                    results.append(result)

        return sorted(results, key=lambda x: x.peak_snr, reverse=True)

    def _coherent_combination(self, detector_snrs: Dict[str, np.ndarray]) -> np.ndarray:
        """Perform coherent combination of detector SNR time series."""
        # Simple coherent sum - in practice would use optimal weighting
        snr_arrays = list(detector_snrs.values())

        # Ensure all arrays have same length
        min_length = min(len(snr) for snr in snr_arrays)
        truncated_snrs = [snr[:min_length] for snr in snr_arrays]

        # Coherent sum
        coherent_snr = np.sum(truncated_snrs, axis=0)

        return coherent_snr

    def optimize_template_parameters(self,
                                   strain_data: Dict[str, np.ndarray],
                                   noise_psd: Dict[str, np.ndarray],
                                   initial_template: Dict[str, float],
                                   frequencies: np.ndarray) -> Dict[str, float]:
        """
        Optimize template parameters using maximum likelihood.

        Args:
            strain_data: Strain data
            noise_psd: Noise PSD
            initial_template: Initial parameter guess
            frequencies: Frequency array

        Returns:
            Optimized parameters
        """
        def negative_log_likelihood(params_array):
            # Convert array to parameter dictionary
            param_names = list(initial_template.keys())
            params = dict(zip(param_names, params_array))

            # Generate template
            try:
                template = self._generate_single_template(params, frequencies, True)

                # Calculate likelihood
                log_l = 0.0

                for detector in self.detector_network:
                    if detector not in strain_data:
                        continue

                    # Simple likelihood calculation
                    snr_ts = self._calculate_snr_time_series(
                        strain_data[detector],
                        template['h_plus'],
                        noise_psd[detector],
                        frequencies
                    )

                    log_l += np.max(snr_ts)**2

                return -log_l  # Negative for minimization

            except Exception:
                return 1e10  # Large penalty for invalid parameters

        # Initial parameter array
        param_names = list(initial_template.keys())
        initial_params = [initial_template[param] for param in param_names]

        # Optimize
        result = minimize(
            negative_log_likelihood,
            initial_params,
            method='Nelder-Mead',
            options={'maxiter': 1000}
        )

        # Convert back to dictionary
        optimized_params = dict(zip(param_names, result.x))

        return optimized_params

    def calculate_detection_efficiency(self,
                                     template_bank: TemplateBank,
                                     injection_params: List[Dict[str, float]],
                                     noise_psd: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate detection efficiency of template bank."""
        detected = 0
        total = len(injection_params)

        for params in injection_params:
            # Generate injection signal
            injection_template = self._generate_single_template(
                params, template_bank.frequencies, True
            )

            # Find best matching template in bank
            best_overlap = 0.0

            for bank_template in template_bank.templates:
                overlap = np.abs(np.sum(
                    np.conj(injection_template['h_plus']) *
                    bank_template['h_plus']
                ))**2

                overlap /= (
                    np.sum(np.abs(injection_template['h_plus'])**2) *
                    np.sum(np.abs(bank_template['h_plus'])**2)
                )

                best_overlap = max(best_overlap, overlap)

            # Count as detected if overlap exceeds minimal match
            if best_overlap >= self.minimal_match:
                detected += 1

        efficiency = detected / total if total > 0 else 0.0

        return {
            'efficiency': efficiency,
            'detected': detected,
            'total': total,
            'best_overlap_mean': np.mean([
                max(np.abs(np.sum(
                    np.conj(self._generate_single_template(params, template_bank.frequencies, True)['h_plus']) *
                    template['h_plus']
                ))**2 / (
                    np.sum(np.abs(self._generate_single_template(params, template_bank.frequencies, True)['h_plus'])**2) *
                    np.sum(np.abs(template['h_plus'])**2)
                ) for template in template_bank.templates)
                for params in injection_params
            ])
        }
