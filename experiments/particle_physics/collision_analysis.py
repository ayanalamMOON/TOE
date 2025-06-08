"""
EG-QGEM Particle Collision Analysis
===================================

This module provides comprehensive tools for analyzing particle collisions
in the EG-QGEM framework, including entanglement effects on scattering
cross-sections, modified particle interactions, and new signature predictions.
"""

import numpy as np
import scipy.special as special
from scipy.integrate import quad, dblquad
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from collections import defaultdict

from theory.constants import CONSTANTS
from theory.entanglement_tensor import EntanglementTensor


@dataclass
class ParticleState:
    """Particle state with EG-QGEM modifications."""
    energy: float
    momentum: np.ndarray
    mass: float
    charge: float
    spin: float
    entanglement_amplitude: complex
    quantum_numbers: Dict[str, int]


@dataclass
class CollisionEvent:
    """Complete collision event information."""
    initial_particles: List[ParticleState]
    final_particles: List[ParticleState]
    center_of_mass_energy: float
    scattering_angle: float
    impact_parameter: float
    entanglement_transfer: float
    cross_section: float
    quantum_corrections: Dict[str, float]


@dataclass
class CrossSectionResult:
    """Cross-section calculation results."""
    total_cross_section: float
    differential_cross_section: np.ndarray
    angles: np.ndarray
    entanglement_enhancement: float
    quantum_corrections: Dict[str, float]
    uncertainty: float


class EGQGEMCollisionAnalyzer:
    """
    Particle collision analyzer with EG-QGEM modifications.

    Calculates scattering cross-sections, analyzes collision dynamics,
    and predicts new signatures arising from entanglement effects.
    """

    def __init__(self):
        """Initialize collision analyzer."""
        self.constants = CONSTANTS
        self.entanglement_tensor = EntanglementTensor()

        # Standard Model parameters (simplified)
        self.coupling_constants = {
            'electromagnetic': 1/137.036,  # Fine structure constant
            'weak': 0.65,  # GF * MW^2 / (sqrt(2) * pi * alpha)
            'strong': 0.12,  # Strong coupling at MZ
            'entanglement': self.constants.kappa_E  # EG-QGEM coupling
        }

        # Particle masses (GeV/c²)
        self.particle_masses = {
            'electron': 0.000511,
            'muon': 0.105658,
            'tau': 1.77686,
            'neutrino_e': 0.0,  # Approximately massless
            'neutrino_mu': 0.0,
            'neutrino_tau': 0.0,
            'up': 0.0022,
            'down': 0.0047,
            'charm': 1.275,
            'strange': 0.095,
            'top': 173.1,
            'bottom': 4.18,
            'photon': 0.0,
            'W': 80.379,
            'Z': 91.1876,
            'higgs': 125.1,
            'proton': 0.938272,
            'neutron': 0.939565
        }

    def calculate_scattering_cross_section(self,
                                         initial_particles: List[str],
                                         final_particles: List[str],
                                         center_of_mass_energy: float,
                                         entanglement_strength: float = 1e-6) -> CrossSectionResult:
        """
        Calculate scattering cross-section with EG-QGEM modifications.

        Args:
            initial_particles: List of initial particle types
            final_particles: List of final particle types
            center_of_mass_energy: Center-of-mass energy in GeV
            entanglement_strength: Entanglement coupling strength

        Returns:
            Complete cross-section results
        """
        # Calculate standard model cross-section
        sm_cross_section = self._calculate_sm_cross_section(
            initial_particles, final_particles, center_of_mass_energy
        )

        # Calculate EG-QGEM corrections
        entanglement_correction = self._calculate_entanglement_correction(
            initial_particles, final_particles, center_of_mass_energy,
            entanglement_strength
        )

        # Calculate differential cross-section
        angles = np.linspace(0, np.pi, 100)
        differential_cs = []

        for angle in angles:
            diff_cs = self._calculate_differential_cross_section(
                initial_particles, final_particles, center_of_mass_energy,
                angle, entanglement_strength
            )
            differential_cs.append(diff_cs)

        differential_cs = np.array(differential_cs)

        # Total cross-section with corrections
        total_cross_section = sm_cross_section * (1 + entanglement_correction)

        # Enhancement factor
        enhancement = entanglement_correction

        # Quantum corrections
        quantum_corrections = self._calculate_quantum_corrections(
            initial_particles, final_particles, center_of_mass_energy,
            entanglement_strength
        )

        # Uncertainty estimation
        uncertainty = self._estimate_uncertainty(
            center_of_mass_energy, entanglement_strength
        )

        return CrossSectionResult(
            total_cross_section=total_cross_section,
            differential_cross_section=differential_cs,
            angles=angles,
            entanglement_enhancement=enhancement,
            quantum_corrections=quantum_corrections,
            uncertainty=uncertainty
        )

    def _calculate_sm_cross_section(self,
                                  initial_particles: List[str],
                                  final_particles: List[str],
                                  energy: float) -> float:
        """Calculate Standard Model cross-section."""
        # Simplified cross-section calculations for common processes

        if set(initial_particles) == {'electron', 'positron'}:
            if set(final_particles) == {'muon', 'antimuon'}:
                # e+ e- -> mu+ mu-
                return self._ee_to_mumu_cross_section(energy)
            elif set(final_particles) == {'electron', 'positron'}:
                # e+ e- -> e+ e- (Bhabha)
                return self._bhabha_cross_section(energy)
            elif 'photon' in final_particles:
                # e+ e- -> gamma gamma
                return self._ee_to_photons_cross_section(energy)

        elif 'proton' in initial_particles:
            if 'proton' in initial_particles and len(initial_particles) == 2:
                # pp collisions
                return self._pp_cross_section(energy)

        # Default: use dimensional analysis estimate
        alpha = self.coupling_constants['electromagnetic']
        hbar_c = 0.197327  # GeV·fm

        # Rough estimate: σ ~ α²(ħc/E)²
        cross_section = alpha**2 * (hbar_c / energy)**2  # in fm²

        return cross_section * 1e-39  # Convert to m²

    def _ee_to_mumu_cross_section(self, energy: float) -> float:
        """e+ e- -> mu+ mu- cross-section."""
        alpha = self.coupling_constants['electromagnetic']
        hbar_c = 0.197327  # GeV·fm
        m_mu = self.particle_masses['muon']

        if energy < 2 * m_mu:
            return 0.0

        beta = np.sqrt(1 - (2 * m_mu / energy)**2)

        # Leading order QED
        sigma = (4 * np.pi * alpha**2 * hbar_c**2) / (3 * energy**2)
        sigma *= beta * (3 - beta**2) / 2

        return sigma * 1e-39  # Convert to m²

    def _bhabha_cross_section(self, energy: float) -> float:
        """e+ e- -> e+ e- Bhabha scattering."""
        alpha = self.coupling_constants['electromagnetic']
        hbar_c = 0.197327

        # Simplified Bhabha formula
        sigma = (4 * np.pi * alpha**2 * hbar_c**2) / energy**2
        sigma *= np.log(energy / (2 * self.particle_masses['electron']))

        return sigma * 1e-39

    def _ee_to_photons_cross_section(self, energy: float) -> float:
        """e+ e- -> gamma gamma cross-section."""
        alpha = self.coupling_constants['electromagnetic']
        hbar_c = 0.197327

        # QED result
        sigma = (2 * np.pi * alpha**2 * hbar_c**2) / energy**2

        return sigma * 1e-39

    def _pp_cross_section(self, energy: float) -> float:
        """Proton-proton total cross-section."""
        # Empirical parameterization (simplified)
        # Real pp cross-section has complex energy dependence

        s = energy**2  # Mandelstam s

        # Approximate total cross-section in mb
        sigma_mb = 35 + 0.3 * np.log(s/100)**2

        return sigma_mb * 1e-31  # Convert mb to m²

    def _calculate_entanglement_correction(self,
                                         initial_particles: List[str],
                                         final_particles: List[str],
                                         energy: float,
                                         entanglement_strength: float) -> float:
        """Calculate entanglement correction to cross-section."""
        # EG-QGEM correction factor

        # Energy scale for entanglement effects
        E_planck = np.sqrt(self.constants.hbar * self.constants.c**5 / self.constants.G)  # Planck energy
        E_entanglement = entanglement_strength * E_planck

        # Correction scales with (E / E_entanglement)^2
        correction = (energy / E_entanglement)**2

        # Additional factors for particle types
        particle_factor = 1.0

        # Leptons vs hadrons
        if all(p in ['electron', 'muon', 'tau', 'positron', 'antimuon', 'antitau']
               for p in initial_particles + final_particles):
            particle_factor *= 0.1  # Weaker for leptons

        # High-energy enhancement
        if energy > 100:  # GeV
            particle_factor *= np.log(energy / 100)

        total_correction = correction * particle_factor

        # Ensure correction is reasonable
        return min(total_correction, 1.0)  # Cap at 100% correction

    def _calculate_differential_cross_section(self,
                                            initial_particles: List[str],
                                            final_particles: List[str],
                                            energy: float,
                                            angle: float,
                                            entanglement_strength: float) -> float:
        """Calculate differential cross-section dσ/dΩ."""
        # Get total cross-section
        total_cs = self._calculate_sm_cross_section(
            initial_particles, final_particles, energy
        )

        # Angular distribution
        if set(initial_particles) == {'electron', 'positron'}:
            # QED processes have (1 + cos²θ) distribution
            angular_factor = (1 + np.cos(angle)**2) / 2
        else:
            # Default isotropic
            angular_factor = 1.0 / (4 * np.pi)

        # Entanglement modification to angular distribution
        entanglement_angular = self._entanglement_angular_correction(
            angle, energy, entanglement_strength
        )

        differential_cs = total_cs * angular_factor * (1 + entanglement_angular)

        return differential_cs

    def _entanglement_angular_correction(self,
                                       angle: float,
                                       energy: float,
                                       entanglement_strength: float) -> float:
        """Calculate entanglement correction to angular distribution."""
        # EG-QGEM predicts modifications to angular distributions

        # Correction amplitude
        amplitude = entanglement_strength * (energy / 100)**0.5  # GeV scale

        # Angular modulation
        # Entanglement tends to enhance forward/backward scattering
        modulation = amplitude * (np.cos(angle)**2 - 0.5)

        return modulation

    def _calculate_quantum_corrections(self,
                                     initial_particles: List[str],
                                     final_particles: List[str],
                                     energy: float,
                                     entanglement_strength: float) -> Dict[str, float]:
        """Calculate various quantum corrections."""
        corrections = {}

        # Loop corrections
        alpha = self.coupling_constants['electromagnetic']
        corrections['radiative'] = alpha / np.pi * np.log(energy / 0.1)  # GeV scale

        # Entanglement-induced corrections
        corrections['entanglement_loop'] = entanglement_strength * np.log(energy)

        # Vacuum polarization
        corrections['vacuum_polarization'] = alpha / (3 * np.pi) * np.log(energy / 0.5)

        # Running coupling corrections
        corrections['running_coupling'] = alpha / np.pi * np.log(energy / 91.2)  # Z mass

        # EG-QGEM specific corrections
        corrections['spacetime_modification'] = entanglement_strength**2 * energy / 1000

        return corrections

    def _estimate_uncertainty(self,
                            energy: float,
                            entanglement_strength: float) -> float:
        """Estimate theoretical uncertainty."""
        # Base uncertainty from higher-order corrections
        base_uncertainty = 0.05  # 5%

        # Energy-dependent uncertainty
        energy_uncertainty = 0.01 * np.log(energy / 10)  # GeV

        # Entanglement uncertainty
        entanglement_uncertainty = entanglement_strength * 10

        total_uncertainty = np.sqrt(
            base_uncertainty**2 +
            energy_uncertainty**2 +
            entanglement_uncertainty**2
        )

        return total_uncertainty

    def analyze_collision_event(self,
                              initial_states: List[ParticleState],
                              final_states: List[ParticleState]) -> CollisionEvent:
        """Analyze complete collision event."""
        # Calculate center-of-mass energy
        total_initial_energy = sum(state.energy for state in initial_states)
        total_initial_momentum = sum(state.momentum for state in initial_states)

        cms_energy = np.sqrt(total_initial_energy**2 - np.linalg.norm(total_initial_momentum)**2)

        # Calculate scattering angle (simplified for 2->2 process)
        if len(initial_states) == 2 and len(final_states) == 2:
            p1_initial = initial_states[0].momentum
            p1_final = final_states[0].momentum

            cos_theta = np.dot(p1_initial, p1_final) / (
                np.linalg.norm(p1_initial) * np.linalg.norm(p1_final)
            )
            scattering_angle = np.arccos(np.clip(cos_theta, -1, 1))
        else:
            scattering_angle = 0.0

        # Calculate entanglement transfer
        initial_entanglement = sum(
            abs(state.entanglement_amplitude)**2 for state in initial_states
        )
        final_entanglement = sum(
            abs(state.entanglement_amplitude)**2 for state in final_states
        )

        entanglement_transfer = final_entanglement - initial_entanglement

        # Estimate impact parameter (simplified)
        impact_parameter = self.constants.hbar / (cms_energy * 1e9 * 1.602e-19)  # Convert GeV to J

        # Calculate cross-section for this process
        initial_types = [self._identify_particle_type(state) for state in initial_states]
        final_types = [self._identify_particle_type(state) for state in final_states]

        cs_result = self.calculate_scattering_cross_section(
            initial_types, final_types, cms_energy
        )

        return CollisionEvent(
            initial_particles=initial_states,
            final_particles=final_states,
            center_of_mass_energy=cms_energy,
            scattering_angle=scattering_angle,
            impact_parameter=impact_parameter,
            entanglement_transfer=entanglement_transfer,
            cross_section=cs_result.total_cross_section,
            quantum_corrections=cs_result.quantum_corrections
        )

    def _identify_particle_type(self, state: ParticleState) -> str:
        """Identify particle type from state."""
        # Simple identification based on mass and charge
        mass = state.mass
        charge = state.charge

        # Match to known particles (simplified)
        for particle, particle_mass in self.particle_masses.items():
            if abs(mass - particle_mass) < 0.01 * particle_mass:
                if particle == 'electron' and charge > 0:
                    return 'positron'
                return particle

        return 'unknown'

    def predict_new_physics_signatures(self,
                                     collision_energy: float,
                                     entanglement_strength: float) -> Dict[str, List[float]]:
        """Predict new physics signatures in EG-QGEM."""
        signatures = {}

        # 1. Entanglement-mediated particle production
        entanglement_production_rate = self._calculate_entanglement_production(
            collision_energy, entanglement_strength
        )
        signatures['entanglement_production'] = entanglement_production_rate

        # 2. Modified Higgs production and decay
        higgs_modifications = self._calculate_higgs_modifications(
            collision_energy, entanglement_strength
        )
        signatures['higgs_modifications'] = higgs_modifications

        # 3. Dark matter production through entanglement
        dark_matter_production = self._calculate_dark_matter_production(
            collision_energy, entanglement_strength
        )
        signatures['dark_matter_production'] = dark_matter_production

        # 4. Modified gauge boson properties
        gauge_modifications = self._calculate_gauge_modifications(
            collision_energy, entanglement_strength
        )
        signatures['gauge_modifications'] = gauge_modifications

        # 5. Quantum gravity effects
        quantum_gravity_effects = self._calculate_quantum_gravity_effects(
            collision_energy, entanglement_strength
        )
        signatures['quantum_gravity_effects'] = quantum_gravity_effects

        return signatures

    def _calculate_entanglement_production(self,
                                         energy: float,
                                         entanglement_strength: float) -> List[float]:
        """Calculate entanglement-mediated particle production rates."""
        # Production rate for entanglement-correlated particle pairs

        production_rates = []
        energy_range = np.logspace(1, 4, 50)  # 10 GeV to 10 TeV

        for E in energy_range:
            if E <= energy:
                # Rate scales with entanglement strength and energy
                rate = entanglement_strength * (E / 100)**2 * np.exp(-E / (10 * energy))
                production_rates.append(rate)
            else:
                production_rates.append(0.0)

        return production_rates

    def _calculate_higgs_modifications(self,
                                     energy: float,
                                     entanglement_strength: float) -> List[float]:
        """Calculate modifications to Higgs physics."""
        higgs_mass = self.particle_masses['higgs']

        modifications = []

        # Mass shift
        mass_shift = entanglement_strength * (energy / 1000) * higgs_mass
        modifications.append(mass_shift)

        # Production cross-section modification
        production_modification = entanglement_strength * np.log(energy / higgs_mass)
        modifications.append(production_modification)

        # Decay rate modifications
        decay_modification = entanglement_strength * (energy / 100)**0.5
        modifications.append(decay_modification)

        # New decay channels
        new_channel_rate = entanglement_strength**2 * (energy / 1000)
        modifications.append(new_channel_rate)

        return modifications

    def _calculate_dark_matter_production(self,
                                        energy: float,
                                        entanglement_strength: float) -> List[float]:
        """Calculate dark matter production through entanglement."""
        production_data = []

        # Mass range for dark matter candidates
        dm_masses = np.logspace(0, 3, 30)  # 1 GeV to 1 TeV

        for dm_mass in dm_masses:
            if energy > 2 * dm_mass:  # Threshold production
                # Production rate through entanglement portal
                rate = entanglement_strength**2 * np.sqrt(1 - (2 * dm_mass / energy)**2)
                rate *= (energy / 1000)  # TeV scale enhancement
                production_data.append(rate)
            else:
                production_data.append(0.0)

        return production_data

    def _calculate_gauge_modifications(self,
                                     energy: float,
                                     entanglement_strength: float) -> List[float]:
        """Calculate modifications to gauge boson properties."""
        modifications = []

        # W boson modifications
        w_mass = self.particle_masses['W']
        w_mass_shift = entanglement_strength * (energy / 1000) * w_mass
        modifications.append(w_mass_shift)

        # Z boson modifications
        z_mass = self.particle_masses['Z']
        z_mass_shift = entanglement_strength * (energy / 1000) * z_mass
        modifications.append(z_mass_shift)

        # Photon effective mass (from entanglement)
        photon_eff_mass = entanglement_strength * (energy / 10000)  # Very small
        modifications.append(photon_eff_mass)

        # Coupling constant modifications
        em_coupling_shift = entanglement_strength * np.log(energy / 91.2)
        modifications.append(em_coupling_shift)

        return modifications

    def _calculate_quantum_gravity_effects(self,
                                         energy: float,
                                         entanglement_strength: float) -> List[float]:
        """Calculate quantum gravity effects in collisions."""
        effects = []

        # Planck scale effects
        planck_energy = 1e19  # GeV (approximate)
        planck_suppression = (energy / planck_energy)**2

        # Black hole production threshold (modified by entanglement)
        bh_threshold = planck_energy * np.sqrt(entanglement_strength)
        bh_production_rate = 0.0
        if energy > bh_threshold:
            bh_production_rate = (energy / bh_threshold)**2 * entanglement_strength
        effects.append(bh_production_rate)

        # Extra dimension effects
        extra_dim_scale = 1000 * np.sqrt(entanglement_strength)  # GeV
        extra_dim_effects = (energy / extra_dim_scale)**4 if energy > extra_dim_scale else 0.0
        effects.append(extra_dim_effects)

        # Graviton production
        graviton_production = entanglement_strength * (energy / 1000)**3
        effects.append(graviton_production)

        # Spacetime noncommutativity
        noncomm_scale = 10000 * entanglement_strength  # GeV
        noncomm_effects = (energy / noncomm_scale)**2 if energy > 100 else 0.0
        effects.append(noncomm_effects)

        return effects

    def generate_monte_carlo_events(self,
                                   process: str,
                                   n_events: int,
                                   collision_energy: float,
                                   entanglement_strength: float = 1e-6) -> List[CollisionEvent]:
        """Generate Monte Carlo collision events."""
        events = []

        for _ in range(n_events):
            # Generate random initial conditions
            if process == 'ee_to_mumu':
                initial_states = self._generate_ee_initial_states(collision_energy)
                final_states = self._generate_mumu_final_states(
                    initial_states, entanglement_strength
                )
            elif process == 'pp_to_jets':
                initial_states = self._generate_pp_initial_states(collision_energy)
                final_states = self._generate_jet_final_states(
                    initial_states, entanglement_strength
                )
            else:
                continue

            # Analyze event
            event = self.analyze_collision_event(initial_states, final_states)
            events.append(event)

        return events

    def _generate_ee_initial_states(self, energy: float) -> List[ParticleState]:
        """Generate e+ e- initial states."""
        beam_energy = energy / 2
        electron_mass = self.particle_masses['electron']

        # Electron
        e_momentum = np.array([0, 0, np.sqrt(beam_energy**2 - electron_mass**2)])
        electron = ParticleState(
            energy=beam_energy,
            momentum=e_momentum,
            mass=electron_mass,
            charge=-1,
            spin=0.5,
            entanglement_amplitude=complex(1, 0),
            quantum_numbers={'lepton_number': 1}
        )

        # Positron
        p_momentum = np.array([0, 0, -np.sqrt(beam_energy**2 - electron_mass**2)])
        positron = ParticleState(
            energy=beam_energy,
            momentum=p_momentum,
            mass=electron_mass,
            charge=1,
            spin=0.5,
            entanglement_amplitude=complex(0, 1),
            quantum_numbers={'lepton_number': -1}
        )

        return [electron, positron]

    def _generate_mumu_final_states(self,
                                   initial_states: List[ParticleState],
                                   entanglement_strength: float) -> List[ParticleState]:
        """Generate muon pair final states."""
        total_energy = sum(state.energy for state in initial_states)
        muon_mass = self.particle_masses['muon']

        # Random scattering angle
        cos_theta = np.random.uniform(-1, 1)
        sin_theta = np.sqrt(1 - cos_theta**2)
        phi = np.random.uniform(0, 2*np.pi)

        # Muon energy and momentum
        muon_energy = total_energy / 2
        muon_momentum_mag = np.sqrt(muon_energy**2 - muon_mass**2)

        # Muon
        mu_momentum = muon_momentum_mag * np.array([
            sin_theta * np.cos(phi),
            sin_theta * np.sin(phi),
            cos_theta
        ])

        muon = ParticleState(
            energy=muon_energy,
            momentum=mu_momentum,
            mass=muon_mass,
            charge=-1,
            spin=0.5,
            entanglement_amplitude=complex(
                np.sqrt(1 - entanglement_strength),
                np.sqrt(entanglement_strength)
            ),
            quantum_numbers={'lepton_number': 1}
        )

        # Antimuon
        antimuon = ParticleState(
            energy=muon_energy,
            momentum=-mu_momentum,
            mass=muon_mass,
            charge=1,
            spin=0.5,
            entanglement_amplitude=complex(
                np.sqrt(entanglement_strength),
                np.sqrt(1 - entanglement_strength)
            ),
            quantum_numbers={'lepton_number': -1}
        )

        return [muon, antimuon]

    def _generate_pp_initial_states(self, energy: float) -> List[ParticleState]:
        """Generate proton-proton initial states."""
        beam_energy = energy / 2
        proton_mass = self.particle_masses['proton']

        # Proton 1
        p1_momentum = np.array([0, 0, np.sqrt(beam_energy**2 - proton_mass**2)])
        proton1 = ParticleState(
            energy=beam_energy,
            momentum=p1_momentum,
            mass=proton_mass,
            charge=1,
            spin=0.5,
            entanglement_amplitude=complex(1, 0),
            quantum_numbers={'baryon_number': 1}
        )

        # Proton 2
        p2_momentum = np.array([0, 0, -np.sqrt(beam_energy**2 - proton_mass**2)])
        proton2 = ParticleState(
            energy=beam_energy,
            momentum=p2_momentum,
            mass=proton_mass,
            charge=1,
            spin=0.5,
            entanglement_amplitude=complex(0, 1),
            quantum_numbers={'baryon_number': 1}
        )

        return [proton1, proton2]

    def _generate_jet_final_states(self,
                                 initial_states: List[ParticleState],
                                 entanglement_strength: float) -> List[ParticleState]:
        """Generate jet final states (simplified)."""
        total_energy = sum(state.energy for state in initial_states)

        # Generate 2 jets for simplicity
        n_jets = 2
        jet_energies = np.random.dirichlet([1] * n_jets) * total_energy

        jets = []
        for i, energy in enumerate(jet_energies):
            # Random direction
            cos_theta = np.random.uniform(-1, 1)
            sin_theta = np.sqrt(1 - cos_theta**2)
            phi = np.random.uniform(0, 2*np.pi)

            # Approximate jet as massless
            momentum_mag = energy
            momentum = momentum_mag * np.array([
                sin_theta * np.cos(phi),
                sin_theta * np.sin(phi),
                cos_theta
            ])

            jet = ParticleState(
                energy=energy,
                momentum=momentum,
                mass=0.0,  # Massless jet approximation
                charge=0,
                spin=1,
                entanglement_amplitude=complex(
                    np.random.normal(0, np.sqrt(entanglement_strength)),
                    np.random.normal(0, np.sqrt(entanglement_strength))
                ),
                quantum_numbers={'color': i}
            )
            jets.append(jet)

        return jets
