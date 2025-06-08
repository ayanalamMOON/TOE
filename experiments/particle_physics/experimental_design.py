"""
Experimental Design Module for EG-QGEM Theory Testing
Designs optimal experiments to test EG-QGEM predictions

This module provides comprehensive experimental design capabilities including:
- Statistical power analysis for EG-QGEM signatures
- Optimal detector configuration design
- Background rejection strategies
- Systematic uncertainty estimation
- Data analysis strategy optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from scipy import stats, optimize
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import EG-QGEM theory components
from theory.constants import *
from theory.entanglement_tensor import EntanglementTensor
from theory.modified_einstein import ModifiedEinsteinSolver

@dataclass
class ExperimentalParameters:
    """Parameters defining an experimental setup"""
    name: str
    energy_range: Tuple[float, float]  # GeV
    luminosity: float                  # cm^-2 s^-1
    running_time: float               # seconds
    detector_efficiency: float        # 0-1
    energy_resolution: float          # fractional
    angular_resolution: float         # radians
    timing_resolution: float          # seconds
    background_rate: float            # Hz
    systematic_uncertainty: float     # fractional

@dataclass
class PhysicsSignature:
    """Definition of a physics signature to search for"""
    name: str
    signal_rate: float                 # Hz
    background_rate: float            # Hz
    signature_function: Callable      # Function that returns signature strength
    discriminating_variables: List[str]
    threshold_optimization: bool = True

class EGQGEMExperimentalDesign:
    """Main experimental design class for EG-QGEM theory testing"""

    def __init__(self):
        self.entanglement_tensor = EntanglementTensor()
        self.modified_einstein = ModifiedEinsteinSolver()

        # Standard Model backgrounds
        self.sm_backgrounds = {
            'qed_bhabha': 1e-33,      # cm^2
            'qed_pair_production': 5e-34,
            'weak_scattering': 1e-38,
            'strong_qcd': 1e-30
        }

        # EG-QGEM signal cross-sections (energy-dependent)
        self.eg_qgem_signals = {
            'entanglement_production': lambda E: 1e-40 * (E/100)**2,  # cm^2
            'modified_higgs': lambda E: 1e-42 * np.exp(-((E-125)/10)**2),
            'quantum_gravity': lambda E: 1e-45 * (E/1000)**4,
            'entangled_fermions': lambda E: 1e-39 * (E/50)**1.5
        }

    def calculate_statistical_significance(self, signal: float, background: float,
                                         systematic: float = 0.0) -> Dict[str, float]:
        """
        Calculate statistical significance using various methods

        Args:
            signal: Expected number of signal events
            background: Expected number of background events
            systematic: Systematic uncertainty fraction

        Returns:
            Dictionary with significance calculations
        """
        total_background = background * (1 + systematic)**2

        # Simple Gaussian approximation
        if total_background > 0:
            gaussian_significance = signal / np.sqrt(total_background)
        else:
            gaussian_significance = np.inf

        # Poisson significance (exact)
        if background > 0:
            # Using Poisson probability
            observed = signal + background
            expected_bg = background

            # Profile likelihood ratio
            if signal > 0:
                poisson_significance = np.sqrt(2 * (observed * np.log(observed / expected_bg) -
                                                  (observed - expected_bg)))
            else:
                poisson_significance = 0.0
        else:
            poisson_significance = np.inf

        # With systematic uncertainties (Asimov significance)
        if systematic > 0 and background > 0:
            sigma_b = systematic * background
            asimov_significance = signal / np.sqrt(background + sigma_b**2)
        else:
            asimov_significance = gaussian_significance

        # Discovery significance (5 sigma requirement)
        discovery_time = None
        if signal > 0:
            required_events = (5.0 * np.sqrt(total_background))**2 / signal**2
            if required_events > 0:
                discovery_time = required_events * (signal + background) / signal

        return {
            'gaussian_significance': gaussian_significance,
            'poisson_significance': poisson_significance,
            'asimov_significance': asimov_significance,
            'discovery_time_events': discovery_time,
            'signal_to_background': signal / background if background > 0 else np.inf,
            'signal_events': signal,
            'background_events': background
        }

    def optimize_experimental_parameters(self, signature: PhysicsSignature,
                                       constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize experimental parameters for maximum sensitivity

        Args:
            signature: Physics signature to optimize for
            constraints: Experimental constraints (budget, technology limits, etc.)

        Returns:
            Optimized experimental parameters
        """
        def figure_of_merit(params):
            """Figure of merit for optimization (negative significance)"""
            energy, luminosity, runtime, efficiency = params

            # Calculate signal rate
            signal_cross_section = signature.signature_function(energy)
            signal_rate = signal_cross_section * luminosity * efficiency
            signal_events = signal_rate * runtime

            # Calculate background rate (simplified)
            background_rate = signature.background_rate * luminosity
            background_events = background_rate * runtime

            # Add systematic uncertainty
            systematic = 0.05  # 5% systematic uncertainty

            # Calculate significance
            significance = self.calculate_statistical_significance(
                signal_events, background_events, systematic
            )

            # Return negative significance for minimization
            return -significance['asimov_significance']

        # Set up constraints
        energy_min = constraints.get('energy_min', 10)
        energy_max = constraints.get('energy_max', 1000)
        luminosity_min = constraints.get('luminosity_min', 1e32)
        luminosity_max = constraints.get('luminosity_max', 1e36)
        runtime_min = constraints.get('runtime_min', 86400)  # 1 day
        runtime_max = constraints.get('runtime_max', 3.15e7)  # 1 year
        efficiency_min = constraints.get('efficiency_min', 0.1)
        efficiency_max = constraints.get('efficiency_max', 0.9)

        # Initial guess
        initial_params = [
            (energy_min + energy_max) / 2,
            (luminosity_min + luminosity_max) / 2,
            (runtime_min + runtime_max) / 2,
            (efficiency_min + efficiency_max) / 2
        ]

        # Bounds
        bounds = [
            (energy_min, energy_max),
            (luminosity_min, luminosity_max),
            (runtime_min, runtime_max),
            (efficiency_min, efficiency_max)
        ]

        # Optimize
        result = optimize.minimize(
            figure_of_merit,
            initial_params,
            method='L-BFGS-B',
            bounds=bounds
        )

        if result.success:
            optimal_energy, optimal_luminosity, optimal_runtime, optimal_efficiency = result.x

            # Calculate final performance metrics
            signal_cs = signature.signature_function(optimal_energy)
            signal_rate = signal_cs * optimal_luminosity * optimal_efficiency
            signal_events = signal_rate * optimal_runtime
            background_events = signature.background_rate * optimal_luminosity * optimal_runtime

            significance = self.calculate_statistical_significance(
                signal_events, background_events, 0.05
            )

            return {
                'success': True,
                'optimal_energy': optimal_energy,
                'optimal_luminosity': optimal_luminosity,
                'optimal_runtime': optimal_runtime,
                'optimal_efficiency': optimal_efficiency,
                'expected_significance': -result.fun,
                'signal_events': signal_events,
                'background_events': background_events,
                'signal_cross_section': signal_cs,
                'optimization_result': result
            }
        else:
            return {
                'success': False,
                'message': result.message
            }

    def design_detector_configuration(self, target_signatures: List[PhysicsSignature],
                                    budget_constraint: float) -> Dict[str, Any]:
        """
        Design optimal detector configuration for multiple signatures

        Args:
            target_signatures: List of physics signatures to detect
            budget_constraint: Maximum budget in arbitrary units

        Returns:
            Optimal detector configuration
        """
        # Define detector components and their costs/benefits
        detector_components = {
            'silicon_tracker': {
                'cost': 100,
                'benefits': {'angular_resolution': 0.95, 'efficiency': 0.98}
            },
            'electromagnetic_calorimeter': {
                'cost': 200,
                'benefits': {'energy_resolution': 0.02, 'efficiency': 0.95}
            },
            'hadronic_calorimeter': {
                'cost': 300,
                'benefits': {'energy_resolution': 0.1, 'efficiency': 0.90}
            },
            'muon_chambers': {
                'cost': 150,
                'benefits': {'efficiency': 0.95, 'background_rejection': 100}
            },
            'timing_detector': {
                'cost': 80,
                'benefits': {'timing_resolution': 1e-11, 'background_rejection': 10}
            },
            'vertex_detector': {
                'cost': 120,
                'benefits': {'angular_resolution': 0.99, 'background_rejection': 5}
            }
        }

        def configuration_performance(component_selection):
            """Calculate overall performance for component selection"""
            total_cost = 0
            combined_efficiency = 1.0
            combined_resolution = 0.1  # Start with moderate resolution
            background_rejection = 1.0

            for component, include in zip(detector_components.keys(), component_selection):
                if include:
                    comp_data = detector_components[component]
                    total_cost += comp_data['cost']

                    # Combine efficiencies multiplicatively
                    if 'efficiency' in comp_data['benefits']:
                        combined_efficiency *= comp_data['benefits']['efficiency']

                    # Improve resolution (smaller is better)
                    if 'energy_resolution' in comp_data['benefits']:
                        combined_resolution = min(combined_resolution,
                                                comp_data['benefits']['energy_resolution'])

                    # Improve background rejection multiplicatively
                    if 'background_rejection' in comp_data['benefits']:
                        background_rejection *= comp_data['benefits']['background_rejection']

            # Calculate figure of merit for all target signatures
            total_significance = 0

            for signature in target_signatures:
                # Estimate signal events with this configuration
                baseline_signal = 1000  # Baseline expected events
                signal_events = baseline_signal * combined_efficiency

                # Estimate background with rejection
                baseline_background = 10000
                background_events = baseline_background / background_rejection

                # Calculate significance
                if background_events > 0:
                    significance = signal_events / np.sqrt(background_events)
                    total_significance += significance

            # Penalize if over budget
            if total_cost > budget_constraint:
                total_significance *= 0.1

            return total_significance, total_cost

        # Enumerate all possible configurations
        n_components = len(detector_components)
        best_config = None
        best_performance = 0
        best_cost = 0

        # Use bit manipulation to try all combinations
        for config_int in range(1, 2**n_components):
            config = [(config_int >> i) & 1 for i in range(n_components)]
            performance, cost = configuration_performance(config)

            if performance > best_performance and cost <= budget_constraint:
                best_performance = performance
                best_config = config
                best_cost = cost

        # Translate best configuration back to component names
        if best_config:
            selected_components = [comp for comp, selected in
                                 zip(detector_components.keys(), best_config) if selected]

            return {
                'success': True,
                'selected_components': selected_components,
                'total_cost': best_cost,
                'expected_performance': best_performance,
                'budget_utilization': best_cost / budget_constraint,
                'configuration_vector': best_config
            }
        else:
            return {
                'success': False,
                'message': 'No feasible configuration found within budget'
            }

    def estimate_systematic_uncertainties(self, experiment: ExperimentalParameters) -> Dict[str, float]:
        """Estimate systematic uncertainties for an experiment"""

        uncertainties = {}

        # Luminosity uncertainty
        uncertainties['luminosity'] = 0.02  # 2% typical

        # Energy scale uncertainty
        uncertainties['energy_scale'] = experiment.energy_resolution * 0.5

        # Efficiency uncertainty
        # Scales with complexity of measurement
        base_efficiency_unc = 0.01
        efficiency_unc = base_efficiency_unc * (2 - experiment.detector_efficiency)
        uncertainties['efficiency'] = efficiency_unc

        # Background modeling uncertainty
        # Depends on background rate
        if experiment.background_rate > 0:
            bg_unc = 0.1 + 0.05 * np.log10(experiment.background_rate)
            uncertainties['background'] = min(bg_unc, 0.5)  # Cap at 50%
        else:
            uncertainties['background'] = 0.05

        # Acceptance uncertainty
        uncertainties['acceptance'] = experiment.angular_resolution * 0.1

        # Trigger efficiency uncertainty
        uncertainties['trigger'] = 0.02

        # Pile-up uncertainty (depends on luminosity)
        pileup_factor = experiment.luminosity / 1e34  # Normalized to typical LHC
        uncertainties['pileup'] = 0.01 * np.sqrt(pileup_factor)

        # Calibration uncertainty
        uncertainties['calibration'] = experiment.systematic_uncertainty * 0.5

        # Total systematic uncertainty (add in quadrature)
        total_systematic = np.sqrt(sum(unc**2 for unc in uncertainties.values()))
        uncertainties['total'] = total_systematic

        return uncertainties

    def simulate_experiment(self, experiment: ExperimentalParameters,
                          signatures: List[PhysicsSignature],
                          num_trials: int = 1000) -> Dict[str, Any]:
        """
        Simulate full experimental run with statistical fluctuations

        Args:
            experiment: Experimental parameters
            signatures: Physics signatures to simulate
            num_trials: Number of Monte Carlo trials

        Returns:
            Simulation results with statistics
        """
        results = {
            'trials': [],
            'discovery_probability': {},
            'expected_exclusion_limits': {},
            'measurement_precision': {}
        }

        for signature in signatures:
            discovery_count = 0
            significance_values = []
            measured_cross_sections = []

            # Calculate expected rates
            true_signal_rate = signature.signal_rate * experiment.detector_efficiency
            true_bg_rate = signature.background_rate

            expected_signal = true_signal_rate * experiment.running_time
            expected_background = true_bg_rate * experiment.running_time

            for trial in range(num_trials):
                # Generate random event counts
                observed_signal = np.random.poisson(expected_signal)
                observed_background = np.random.poisson(expected_background)
                total_observed = observed_signal + observed_background

                # Add systematic fluctuations
                systematic_unc = self.estimate_systematic_uncertainties(experiment)
                systematic_factor = np.random.normal(1.0, systematic_unc['total'])
                total_observed *= systematic_factor

                # Calculate significance for this trial
                if expected_background > 0:
                    significance = (total_observed - expected_background) / np.sqrt(expected_background)
                else:
                    significance = np.sqrt(total_observed)

                significance_values.append(significance)

                # Check for discovery (5 sigma)
                if significance > 5.0:
                    discovery_count += 1

                # Estimate measured cross-section
                if experiment.luminosity > 0:
                    measured_cs = total_observed / (experiment.luminosity * experiment.running_time)
                    measured_cross_sections.append(measured_cs)

            # Calculate statistics
            discovery_prob = discovery_count / num_trials
            mean_significance = np.mean(significance_values)
            significance_std = np.std(significance_values)

            # Exclusion limit (95% CL upper limit)
            exclusion_limit = np.percentile(measured_cross_sections, 95) if measured_cross_sections else 0

            # Measurement precision
            if measured_cross_sections:
                measurement_precision = np.std(measured_cross_sections) / np.mean(measured_cross_sections)
            else:
                measurement_precision = np.inf

            results['discovery_probability'][signature.name] = discovery_prob
            results['expected_exclusion_limits'][signature.name] = exclusion_limit
            results['measurement_precision'][signature.name] = measurement_precision

            # Store detailed results for this signature
            results['trials'].append({
                'signature': signature.name,
                'significance_values': significance_values,
                'mean_significance': mean_significance,
                'significance_std': significance_std,
                'measured_cross_sections': measured_cross_sections,
                'expected_signal': expected_signal,
                'expected_background': expected_background
            })

        return results

    def optimize_background_rejection(self, signal_features: np.ndarray,
                                    background_features: np.ndarray,
                                    feature_names: List[str]) -> Dict[str, Any]:
        """
        Optimize background rejection using machine learning

        Args:
            signal_features: Signal event features (n_events x n_features)
            background_features: Background event features
            feature_names: Names of the features

        Returns:
            Optimization results and trained models
        """
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler

        # Combine datasets
        X_signal = signal_features
        X_background = background_features
        X = np.vstack([X_signal, X_background])

        # Create labels (1 for signal, 0 for background)
        y = np.hstack([np.ones(len(X_signal)), np.zeros(len(X_background))])

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train multiple classifiers
        classifiers = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }

        results = {}

        for name, clf in classifiers.items():
            # Cross-validation
            cv_scores = cross_val_score(clf, X_scaled, y, cv=5, scoring='roc_auc')

            # Fit classifier
            clf.fit(X_scaled, y)

            # Get feature importances
            if hasattr(clf, 'feature_importances_'):
                feature_importance = dict(zip(feature_names, clf.feature_importances_))
            else:
                feature_importance = {}

            # Calculate ROC curve
            y_proba = clf.predict_proba(X_scaled)[:, 1]
            fpr, tpr, thresholds = roc_curve(y, y_proba)
            roc_auc = auc(fpr, tpr)

            # Find optimal threshold for best signal/background separation
            # Optimize for maximum significance
            best_significance = 0
            best_threshold = 0.5

            for threshold in thresholds:
                y_pred = (y_proba >= threshold).astype(int)

                # Calculate confusion matrix elements
                tp = np.sum((y == 1) & (y_pred == 1))  # True positives
                fp = np.sum((y == 0) & (y_pred == 1))  # False positives

                if fp > 0:
                    significance = tp / np.sqrt(fp)
                    if significance > best_significance:
                        best_significance = significance
                        best_threshold = threshold

            results[name] = {
                'cv_scores': cv_scores,
                'mean_cv_score': np.mean(cv_scores),
                'std_cv_score': np.std(cv_scores),
                'roc_auc': roc_auc,
                'feature_importance': feature_importance,
                'best_threshold': best_threshold,
                'best_significance': best_significance,
                'classifier': clf,
                'scaler': scaler,
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds
            }

        # Find best classifier
        best_classifier = max(results.keys(), key=lambda k: results[k]['mean_cv_score'])

        return {
            'best_classifier': best_classifier,
            'results': results,
            'feature_names': feature_names
        }

    def generate_experiment_comparison_plots(self, experiments: List[ExperimentalParameters],
                                           signatures: List[PhysicsSignature],
                                           save_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate plots comparing different experimental configurations"""

        n_experiments = len(experiments)
        n_signatures = len(signatures)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Significance comparison
        significance_matrix = np.zeros((n_experiments, n_signatures))

        for i, exp in enumerate(experiments):
            for j, sig in enumerate(signatures):
                # Calculate expected significance
                signal_events = sig.signal_rate * exp.detector_efficiency * exp.running_time
                bg_events = sig.background_rate * exp.running_time

                if bg_events > 0:
                    significance = signal_events / np.sqrt(bg_events)
                else:
                    significance = np.sqrt(signal_events)

                significance_matrix[i, j] = significance

        im1 = axes[0, 0].imshow(significance_matrix, aspect='auto', cmap='viridis')
        axes[0, 0].set_xlabel('Signatures')
        axes[0, 0].set_ylabel('Experiments')
        axes[0, 0].set_title('Expected Significance')
        axes[0, 0].set_xticks(range(n_signatures))
        axes[0, 0].set_xticklabels([sig.name[:10] for sig in signatures], rotation=45)
        axes[0, 0].set_yticks(range(n_experiments))
        axes[0, 0].set_yticklabels([exp.name[:10] for exp in experiments])
        plt.colorbar(im1, ax=axes[0, 0])

        # Energy vs Luminosity
        energies = [(exp.energy_range[0] + exp.energy_range[1])/2 for exp in experiments]
        luminosities = [exp.luminosity for exp in experiments]

        axes[0, 1].scatter(energies, luminosities, s=100, alpha=0.7)
        for i, exp in enumerate(experiments):
            axes[0, 1].annotate(exp.name[:5], (energies[i], luminosities[i]),
                              xytext=(5, 5), textcoords='offset points')
        axes[0, 1].set_xlabel('Center-of-Mass Energy (GeV)')
        axes[0, 1].set_ylabel('Luminosity (cm⁻²s⁻¹)')
        axes[0, 1].set_title('Energy vs Luminosity')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True)

        # Resolution comparison
        energy_resolutions = [exp.energy_resolution for exp in experiments]
        angular_resolutions = [exp.angular_resolution for exp in experiments]

        axes[1, 0].scatter(energy_resolutions, angular_resolutions, s=100, alpha=0.7)
        for i, exp in enumerate(experiments):
            axes[1, 0].annotate(exp.name[:5],
                              (energy_resolutions[i], angular_resolutions[i]),
                              xytext=(5, 5), textcoords='offset points')
        axes[1, 0].set_xlabel('Energy Resolution')
        axes[1, 0].set_ylabel('Angular Resolution (rad)')
        axes[1, 0].set_title('Resolution Comparison')
        axes[1, 0].grid(True)

        # Cost-benefit analysis
        # Estimate relative costs based on parameters
        costs = []
        benefits = []

        for exp in experiments:
            # Simple cost model
            cost = (exp.luminosity / 1e34 +
                   1 / exp.energy_resolution +
                   1 / exp.angular_resolution +
                   exp.running_time / 1e6)
            costs.append(cost)

            # Benefit = average significance across all signatures
            avg_significance = np.mean([significance_matrix[i, j]
                                      for j in range(n_signatures)])
            benefits.append(avg_significance)

        axes[1, 1].scatter(costs, benefits, s=100, alpha=0.7)
        for i, exp in enumerate(experiments):
            axes[1, 1].annotate(exp.name[:5], (costs[i], benefits[i]),
                              xytext=(5, 5), textcoords='offset points')
        axes[1, 1].set_xlabel('Relative Cost')
        axes[1, 1].set_ylabel('Average Significance')
        axes[1, 1].set_title('Cost-Benefit Analysis')
        axes[1, 1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return {
            'figure': fig,
            'significance_matrix': significance_matrix,
            'best_experiment_per_signature': {
                signatures[j].name: experiments[np.argmax(significance_matrix[:, j])].name
                for j in range(n_signatures)
            },
            'most_cost_effective': experiments[np.argmax(np.array(benefits) / np.array(costs))].name
        }
