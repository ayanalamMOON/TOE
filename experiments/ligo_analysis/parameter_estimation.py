"""
EG-QGEM Parameter Estimation for LIGO Analysis
==============================================

This module provides Bayesian parameter estimation specifically for
EG-QGEM modified gravitational waveforms, including entanglement coupling
strength, modified orbital dynamics, and quantum coherence effects.
"""

import numpy as np
import scipy.optimize as opt
from scipy.stats import multivariate_normal
from scipy.interpolate import interp1d
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union, Callable
import emcee
import corner
import matplotlib.pyplot as plt
from collections import defaultdict

from theory.constants import CONSTANTS
from theory.entanglement_tensor import EntanglementTensor


@dataclass
class ParameterPrior:
    """Prior distribution for a parameter."""
    name: str
    dist_type: str  # 'uniform', 'normal', 'log_uniform'
    bounds: Tuple[float, float]  # (min, max) or (mean, std)

    def log_prob(self, value: float) -> float:
        """Calculate log probability for given value."""
        if self.dist_type == 'uniform':
            if self.bounds[0] <= value <= self.bounds[1]:
                return -np.log(self.bounds[1] - self.bounds[0])
            return -np.inf
        elif self.dist_type == 'normal':
            mean, std = self.bounds
            return -0.5 * ((value - mean) / std)**2 - np.log(std * np.sqrt(2*np.pi))
        elif self.dist_type == 'log_uniform':
            log_min, log_max = np.log(self.bounds)
            if log_min <= np.log(value) <= log_max:
                return -np.log(value) - np.log(log_max - log_min)
            return -np.inf
        else:
            raise ValueError(f"Unknown distribution type: {self.dist_type}")


@dataclass
class ParameterEstimationResult:
    """Results from parameter estimation."""
    parameter_names: List[str]
    samples: np.ndarray
    log_likelihood_samples: np.ndarray
    log_evidence: float
    best_fit_params: Dict[str, float]
    credible_intervals: Dict[str, Tuple[float, float]]
    correlation_matrix: np.ndarray
    convergence_stats: Dict[str, float]


class EGQGEMParameterEstimator:
    """
    Bayesian parameter estimation for EG-QGEM modified gravitational waves.

    Estimates both standard GR parameters and EG-QGEM specific parameters
    including entanglement coupling strength and quantum coherence effects.
    """

    def __init__(self, detector_network: List[str] = ['H1', 'L1']):
        """
        Initialize parameter estimator.

        Args:
            detector_network: List of detector names
        """
        self.detector_network = detector_network
        self.constants = CONSTANTS
        self.entanglement_tensor = EntanglementTensor()

        # Parameter definitions
        self.standard_params = ['mass1', 'mass2', 'spin1z', 'spin2z',
                               'luminosity_distance', 'inclination',
                               'polarization_angle', 'phase', 'geocent_time']

        self.egqgem_params = ['entanglement_coupling', 'coherence_time',
                             'quantum_phase_shift', 'nonlinear_coupling',
                             'entanglement_decay_rate']

        self.all_params = self.standard_params + self.egqgem_params

        # Setup default priors
        self._setup_default_priors()

    def _setup_default_priors(self):
        """Setup default parameter priors."""
        self.priors = {
            # Standard GR parameters
            'mass1': ParameterPrior('mass1', 'uniform', (10, 80)),  # Solar masses
            'mass2': ParameterPrior('mass2', 'uniform', (10, 80)),
            'spin1z': ParameterPrior('spin1z', 'uniform', (-0.99, 0.99)),
            'spin2z': ParameterPrior('spin2z', 'uniform', (-0.99, 0.99)),
            'luminosity_distance': ParameterPrior('luminosity_distance', 'log_uniform', (100, 2000)),  # Mpc
            'inclination': ParameterPrior('inclination', 'uniform', (0, np.pi)),
            'polarization_angle': ParameterPrior('polarization_angle', 'uniform', (0, np.pi)),
            'phase': ParameterPrior('phase', 'uniform', (0, 2*np.pi)),
            'geocent_time': ParameterPrior('geocent_time', 'uniform', (-0.1, 0.1)),  # seconds

            # EG-QGEM specific parameters
            'entanglement_coupling': ParameterPrior('entanglement_coupling', 'log_uniform', (1e-6, 1e-2)),
            'coherence_time': ParameterPrior('coherence_time', 'log_uniform', (1e-3, 1.0)),  # seconds
            'quantum_phase_shift': ParameterPrior('quantum_phase_shift', 'uniform', (0, 2*np.pi)),
            'nonlinear_coupling': ParameterPrior('nonlinear_coupling', 'log_uniform', (1e-8, 1e-4)),
            'entanglement_decay_rate': ParameterPrior('entanglement_decay_rate', 'log_uniform', (0.1, 100))  # Hz
        }

    def set_custom_priors(self, custom_priors: Dict[str, ParameterPrior]):
        """Set custom parameter priors."""
        self.priors.update(custom_priors)

    def log_prior(self, params: Dict[str, float]) -> float:
        """Calculate log prior probability for parameter set."""
        log_p = 0.0
        for name, value in params.items():
            if name in self.priors:
                log_p += self.priors[name].log_prob(value)
            else:
                return -np.inf

        # Physical constraints
        if params['mass1'] < params['mass2']:
            return -np.inf  # Enforce mass1 >= mass2

        return log_p

    def log_likelihood(self, params: Dict[str, float],
                      data: Dict[str, np.ndarray],
                      noise_psd: Dict[str, np.ndarray],
                      frequencies: np.ndarray) -> float:
        """
        Calculate log likelihood for given parameters.

        Args:
            params: Parameter dictionary
            data: Strain data for each detector
            noise_psd: Power spectral density for each detector
            frequencies: Frequency array

        Returns:
            Log likelihood value
        """
        log_l = 0.0

        # Generate template waveforms
        try:
            templates = self._generate_template_waveforms(params, frequencies)
        except Exception:
            return -np.inf

        # Calculate likelihood for each detector
        for detector in self.detector_network:
            if detector not in data or detector not in noise_psd:
                continue

            # Get detector response
            template_strain = self._apply_detector_response(
                templates, detector, params
            )

            # Calculate matched filter SNR
            df = frequencies[1] - frequencies[0]
            inner_product = 4 * np.real(
                np.sum(np.conj(data[detector]) * template_strain / noise_psd[detector]) * df
            )

            template_norm = 4 * np.real(
                np.sum(np.abs(template_strain)**2 / noise_psd[detector]) * df
            )

            data_norm = 4 * np.real(
                np.sum(np.abs(data[detector])**2 / noise_psd[detector]) * df
            )

            # Log likelihood (up to normalization constant)
            log_l += inner_product - 0.5 * template_norm

        return log_l

    def _generate_template_waveforms(self, params: Dict[str, float],
                                   frequencies: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate EG-QGEM modified template waveforms."""
        # Extract masses and spins
        m1, m2 = params['mass1'], params['mass2']
        s1z, s2z = params['spin1z'], params['spin2z']

        # Calculate derived quantities
        total_mass = m1 + m2
        chirp_mass = (m1 * m2)**(3/5) / total_mass**(1/5)
        eta = (m1 * m2) / total_mass**2

        # EG-QGEM parameters
        alpha = params['entanglement_coupling']
        tau_c = params['coherence_time']
        phi_q = params['quantum_phase_shift']
        beta = params['nonlinear_coupling']
        gamma = params['entanglement_decay_rate']

        # Generate base inspiral waveform
        phase = self._inspiral_phase(frequencies, chirp_mass, eta, s1z, s2z)
        amplitude = self._inspiral_amplitude(frequencies, chirp_mass,
                                           params['luminosity_distance'])

        # Apply EG-QGEM modifications
        # 1. Entanglement phase corrections
        entanglement_phase = alpha * np.log(frequencies) * np.sin(phi_q)

        # 2. Quantum coherence amplitude modulation
        coherence_factor = np.exp(-frequencies * tau_c)

        # 3. Nonlinear coupling corrections
        nonlinear_phase = beta * frequencies**2 * np.cos(2 * phi_q)

        # 4. Frequency-dependent entanglement decay
        decay_factor = 1 / (1 + (frequencies / gamma)**2)

        # Combine modifications
        total_phase = phase + entanglement_phase + nonlinear_phase
        total_amplitude = amplitude * coherence_factor * decay_factor

        # Generate plus and cross polarizations
        h_plus = total_amplitude * np.exp(1j * total_phase)
        h_cross = total_amplitude * np.exp(1j * (total_phase + np.pi/2))

        return {'plus': h_plus, 'cross': h_cross}

    def _inspiral_phase(self, frequencies: np.ndarray, chirp_mass: float,
                       eta: float, spin1z: float, spin2z: float) -> np.ndarray:
        """Calculate post-Newtonian inspiral phase."""
        # Convert to geometric units
        M_geo = chirp_mass * self.constants.SOLAR_MASS_GEO
        f_geo = frequencies * 2 * np.pi * M_geo

        # Post-Newtonian phase coefficients
        psi_0 = 3 / (128 * eta * f_geo**(5/3))
        psi_2 = (3715/756 + 55*eta/9) * f_geo**(-1)
        psi_3 = -16*np.pi * f_geo**(-2/3)
        psi_4 = (15293365/508032 + 27145*eta/504 + 3085*eta**2/72) * f_geo**(-1/3)

        # Spin corrections (simplified)
        chi_s = 0.5 * (spin1z + spin2z)
        chi_a = 0.5 * (spin1z - spin2z)

        psi_spin = (113*chi_s/3 + 113*chi_a/3) * f_geo**(-2/3)

        return psi_0 + psi_2 + psi_3 + psi_4 + psi_spin

    def _inspiral_amplitude(self, frequencies: np.ndarray,
                          chirp_mass: float, distance: float) -> np.ndarray:
        """Calculate inspiral amplitude."""
        M_geo = chirp_mass * self.constants.SOLAR_MASS_GEO
        f_geo = frequencies * 2 * np.pi * M_geo

        # Distance in geometric units
        D_geo = distance * 1e6 * self.constants.PARSEC_GEO

        # Amplitude coefficient
        A_0 = np.sqrt(5*np.pi/24) * M_geo**(5/6) / (D_geo * f_geo**(7/6))

        return A_0

    def _apply_detector_response(self, templates: Dict[str, np.ndarray],
                               detector: str, params: Dict[str, float]) -> np.ndarray:
        """Apply detector antenna response."""
        # Simplified detector response - would need full antenna patterns
        inclination = params['inclination']
        polarization = params['polarization_angle']

        # Antenna response functions (simplified)
        F_plus = 0.5 * (1 + np.cos(inclination)**2) * np.cos(2*polarization)
        F_cross = np.cos(inclination) * np.sin(2*polarization)

        return F_plus * templates['plus'] + F_cross * templates['cross']

    def run_mcmc_estimation(self, data: Dict[str, np.ndarray],
                          noise_psd: Dict[str, np.ndarray],
                          frequencies: np.ndarray,
                          n_walkers: int = 100,
                          n_steps: int = 1000,
                          burn_in: int = 500,
                          active_params: Optional[List[str]] = None) -> ParameterEstimationResult:
        """
        Run MCMC parameter estimation.

        Args:
            data: Strain data for each detector
            noise_psd: Noise power spectral density
            frequencies: Frequency array
            n_walkers: Number of MCMC walkers
            n_steps: Number of MCMC steps
            burn_in: Number of burn-in steps
            active_params: Parameters to estimate (None = all)

        Returns:
            Parameter estimation results
        """
        if active_params is None:
            active_params = self.all_params

        n_dim = len(active_params)

        # Initialize walkers
        initial_positions = self._initialize_walkers(active_params, n_walkers)

        # Setup MCMC sampler
        def log_probability(theta):
            params = dict(zip(active_params, theta))

            # Fill in fixed parameters with default values
            for param in self.all_params:
                if param not in params:
                    params[param] = self._get_default_value(param)

            log_p = self.log_prior(params)
            if not np.isfinite(log_p):
                return -np.inf

            log_l = self.log_likelihood(params, data, noise_psd, frequencies)
            return log_p + log_l

        sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_probability)

        # Run MCMC
        print("Running MCMC sampling...")
        sampler.run_mcmc(initial_positions, n_steps, progress=True)

        # Extract results
        samples = sampler.get_chain(discard=burn_in, flat=True)
        log_prob_samples = sampler.get_log_prob(discard=burn_in, flat=True)

        # Calculate best fit and credible intervals
        best_fit_idx = np.argmax(log_prob_samples)
        best_fit_params = dict(zip(active_params, samples[best_fit_idx]))

        credible_intervals = {}
        for i, param in enumerate(active_params):
            param_samples = samples[:, i]
            credible_intervals[param] = (
                np.percentile(param_samples, 16),
                np.percentile(param_samples, 84)
            )

        # Calculate correlation matrix
        correlation_matrix = np.corrcoef(samples.T)

        # Convergence statistics
        convergence_stats = {
            'acceptance_fraction': np.mean(sampler.acceptance_fraction),
            'autocorr_time': np.mean(sampler.get_autocorr_time(quiet=True))
        }

        # Estimate evidence (simplified)
        log_evidence = np.max(log_prob_samples)  # Rough approximation

        return ParameterEstimationResult(
            parameter_names=active_params,
            samples=samples,
            log_likelihood_samples=log_prob_samples,
            log_evidence=log_evidence,
            best_fit_params=best_fit_params,
            credible_intervals=credible_intervals,
            correlation_matrix=correlation_matrix,
            convergence_stats=convergence_stats
        )

    def _initialize_walkers(self, active_params: List[str],
                          n_walkers: int) -> np.ndarray:
        """Initialize MCMC walkers around prior means."""
        positions = []

        for _ in range(n_walkers):
            walker_position = []
            for param in active_params:
                prior = self.priors[param]

                if prior.dist_type == 'uniform':
                    # Start near center of uniform prior
                    center = 0.5 * (prior.bounds[0] + prior.bounds[1])
                    width = 0.1 * (prior.bounds[1] - prior.bounds[0])
                    value = np.random.normal(center, width)
                    value = np.clip(value, prior.bounds[0], prior.bounds[1])
                elif prior.dist_type == 'log_uniform':
                    # Start near geometric mean
                    log_center = 0.5 * (np.log(prior.bounds[0]) + np.log(prior.bounds[1]))
                    log_width = 0.1 * (np.log(prior.bounds[1]) - np.log(prior.bounds[0]))
                    log_value = np.random.normal(log_center, log_width)
                    value = np.exp(log_value)
                    value = np.clip(value, prior.bounds[0], prior.bounds[1])
                else:  # normal
                    mean, std = prior.bounds
                    value = np.random.normal(mean, 0.1 * std)

                walker_position.append(value)

            positions.append(walker_position)

        return np.array(positions)

    def _get_default_value(self, param: str) -> float:
        """Get default value for parameter."""
        defaults = {
            'mass1': 30.0, 'mass2': 25.0,
            'spin1z': 0.0, 'spin2z': 0.0,
            'luminosity_distance': 400.0,
            'inclination': np.pi/3,
            'polarization_angle': 0.0,
            'phase': 0.0,
            'geocent_time': 0.0,
            'entanglement_coupling': 1e-4,
            'coherence_time': 0.1,
            'quantum_phase_shift': 0.0,
            'nonlinear_coupling': 1e-6,
            'entanglement_decay_rate': 10.0
        }
        return defaults.get(param, 0.0)

    def plot_corner_plot(self, result: ParameterEstimationResult,
                        save_path: Optional[str] = None) -> plt.Figure:
        """Generate corner plot of parameter posteriors."""
        fig = corner.corner(
            result.samples,
            labels=result.parameter_names,
            truths=[result.best_fit_params[p] for p in result.parameter_names],
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 12}
        )

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def calculate_bayes_factors(self,
                              gr_result: ParameterEstimationResult,
                              egqgem_result: ParameterEstimationResult) -> Dict[str, float]:
        """Calculate Bayes factors comparing GR and EG-QGEM models."""
        log_bf_egqgem_gr = egqgem_result.log_evidence - gr_result.log_evidence

        return {
            'log_bayes_factor_egqgem_vs_gr': log_bf_egqgem_gr,
            'bayes_factor_egqgem_vs_gr': np.exp(log_bf_egqgem_gr),
            'evidence_ratio': np.exp(log_bf_egqgem_gr)
        }

    def estimate_parameter_constraints(self,
                                     result: ParameterEstimationResult) -> Dict[str, Dict[str, float]]:
        """Estimate parameter constraints from posteriors."""
        constraints = {}

        for i, param in enumerate(result.parameter_names):
            samples = result.samples[:, i]

            constraints[param] = {
                'mean': np.mean(samples),
                'std': np.std(samples),
                'median': np.median(samples),
                'mode': samples[np.argmax(result.log_likelihood_samples)],
                '90_percent_lower': np.percentile(samples, 5),
                '90_percent_upper': np.percentile(samples, 95),
                '95_percent_lower': np.percentile(samples, 2.5),
                '95_percent_upper': np.percentile(samples, 97.5)
            }

        return constraints
