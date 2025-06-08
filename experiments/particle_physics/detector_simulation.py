"""
Particle Detector Simulation Module for EG-QGEM Theory
Simulates detector responses to EG-QGEM modified particle interactions

This module provides comprehensive detector simulation capabilities including:
- Multi-layer detector geometries
- Energy deposition modeling with EG-QGEM corrections
- Entanglement signature detection in detector signals
- Calorimeter and tracking detector responses
- Machine learning-based event reconstruction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy import stats, optimize
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import EG-QGEM theory components
from theory.constants import *
from theory.entanglement_tensor import EntanglementTensor
from theory.modified_einstein import ModifiedEinsteinSolver

@dataclass
class DetectorLayer:
    """Represents a single detector layer"""
    name: str
    material: str
    thickness: float  # cm
    radius: float     # cm
    z_position: float # cm
    efficiency: float
    resolution: float # energy resolution (%)

@dataclass
class ParticleHit:
    """Represents a particle hit in detector"""
    layer_id: int
    position: Tuple[float, float, float]  # x, y, z
    energy: float
    time: float
    particle_type: str
    entanglement_strength: float

class DetectorGeometry:
    """Defines detector geometry and materials"""

    def __init__(self):
        self.layers = []
        self.magnetic_field = 4.0  # Tesla
        self.setup_default_geometry()

    def setup_default_geometry(self):
        """Setup default CMS-like detector geometry"""
        # Silicon tracker layers
        for i in range(6):
            self.layers.append(DetectorLayer(
                name=f"silicon_layer_{i+1}",
                material="silicon",
                thickness=0.03,
                radius=4.0 + i * 2.0,
                z_position=0.0,
                efficiency=0.98,
                resolution=0.1
            ))

        # ECAL layers
        for i in range(25):
            self.layers.append(DetectorLayer(
                name=f"ecal_layer_{i+1}",
                material="PbWO4",
                thickness=2.3,
                radius=129.0 + i * 2.3,
                z_position=0.0,
                efficiency=0.95,
                resolution=2.0
            ))

        # HCAL layers
        for i in range(17):
            self.layers.append(DetectorLayer(
                name=f"hcal_layer_{i+1}",
                material="brass",
                thickness=5.0,
                radius=177.0 + i * 5.0,
                z_position=0.0,
                efficiency=0.90,
                resolution=15.0
            ))

        # Muon chambers
        for i in range(4):
            self.layers.append(DetectorLayer(
                name=f"muon_layer_{i+1}",
                material="RPC",
                thickness=2.0,
                radius=400.0 + i * 50.0,
                z_position=0.0,
                efficiency=0.95,
                resolution=5.0
            ))

class EGQGEMDetectorSimulation:
    """Main detector simulation class with EG-QGEM modifications"""

    def __init__(self, geometry: Optional[DetectorGeometry] = None):
        self.geometry = geometry or DetectorGeometry()
        self.entanglement_tensor = EntanglementTensor()
        self.modified_einstein = ModifiedEinsteinSolver()

        # EG-QGEM detector response parameters
        self.entanglement_coupling = 1e-18  # GeV^-2
        self.coherence_length = 1e-15       # meters
        self.decoherence_time = 1e-12       # seconds

        # Machine learning models
        self.event_classifier = None
        self.feature_scaler = StandardScaler()

    def simulate_particle_propagation(self, particle_type: str, energy: float,
                                    momentum: np.ndarray, position: np.ndarray,
                                    entanglement_state: Optional[np.ndarray] = None) -> List[ParticleHit]:
        """
        Simulate particle propagation through detector with EG-QGEM effects

        Args:
            particle_type: Type of particle ('electron', 'muon', 'photon', etc.)
            energy: Initial energy in GeV
            momentum: 3D momentum vector in GeV
            position: Initial 3D position in cm
            entanglement_state: Optional entanglement state vector

        Returns:
            List of ParticleHit objects
        """
        hits = []
        current_energy = energy
        current_position = position.copy()
        current_momentum = momentum.copy()

        # Calculate initial entanglement strength
        if entanglement_state is not None:
            entanglement_strength = np.linalg.norm(entanglement_state)
        else:
            entanglement_strength = 0.0

        for i, layer in enumerate(self.geometry.layers):
            # Check if particle reaches this layer
            distance_to_layer = np.sqrt(current_position[0]**2 + current_position[1]**2)

            if distance_to_layer < layer.radius and current_energy > 0.001:  # 1 MeV threshold

                # Calculate EG-QGEM modified energy loss
                energy_loss = self._calculate_energy_loss(
                    particle_type, current_energy, layer, entanglement_strength
                )

                # Apply detector efficiency
                if np.random.random() < layer.efficiency:
                    # Calculate hit position
                    phi = np.arctan2(current_position[1], current_position[0])
                    hit_x = layer.radius * np.cos(phi)
                    hit_y = layer.radius * np.sin(phi)
                    hit_z = current_position[2]

                    # Apply position resolution
                    pos_res = 0.01  # 100 microns
                    hit_x += np.random.normal(0, pos_res)
                    hit_y += np.random.normal(0, pos_res)
                    hit_z += np.random.normal(0, pos_res)

                    # Apply energy resolution
                    measured_energy = energy_loss * (1 + np.random.normal(0, layer.resolution/100))

                    # Calculate time with EG-QGEM corrections
                    time = self._calculate_hit_time(distance_to_layer, current_energy, entanglement_strength)

                    # Update entanglement strength due to decoherence
                    entanglement_strength *= np.exp(-time / self.decoherence_time)

                    hit = ParticleHit(
                        layer_id=i,
                        position=(hit_x, hit_y, hit_z),
                        energy=measured_energy,
                        time=time,
                        particle_type=particle_type,
                        entanglement_strength=entanglement_strength
                    )
                    hits.append(hit)

                # Update particle state
                current_energy -= energy_loss

                # Multiple scattering
                if particle_type in ['electron', 'muon']:
                    scatter_angle = self._calculate_multiple_scattering(
                        particle_type, current_energy, layer.thickness
                    )
                    current_momentum = self._apply_scattering(current_momentum, scatter_angle)

                # Update position
                current_position = self._propagate_to_next_layer(
                    current_position, current_momentum, layer.radius
                )

        return hits

    def _calculate_energy_loss(self, particle_type: str, energy: float,
                             layer: DetectorLayer, entanglement_strength: float) -> float:
        """Calculate energy loss with EG-QGEM corrections"""

        # Standard Bethe-Bloch formula
        if particle_type == 'electron':
            # Bremsstrahlung dominant for electrons
            X0 = self._get_radiation_length(layer.material)
            standard_loss = energy * (layer.thickness / X0)
        elif particle_type == 'muon':
            # Ionization for muons
            standard_loss = 2.0 * layer.thickness  # MeV per cm
        elif particle_type == 'photon':
            # Pair production and Compton scattering
            standard_loss = energy * (1 - np.exp(-layer.thickness / 2.0))
        else:
            standard_loss = 1.0 * layer.thickness

        # EG-QGEM correction
        eg_qgem_correction = 1.0 + self.entanglement_coupling * entanglement_strength**2 * energy

        return standard_loss * eg_qgem_correction

    def _get_radiation_length(self, material: str) -> float:
        """Get radiation length for material in cm"""
        radiation_lengths = {
            'silicon': 9.36,
            'PbWO4': 0.89,
            'brass': 1.49,
            'RPC': 10.0
        }
        return radiation_lengths.get(material, 10.0)

    def _calculate_multiple_scattering(self, particle_type: str, energy: float, thickness: float) -> float:
        """Calculate multiple scattering angle"""
        # Highland formula
        if particle_type == 'electron':
            mass = 0.000511  # GeV
        elif particle_type == 'muon':
            mass = 0.10566   # GeV
        else:
            return 0.0

        momentum = np.sqrt(energy**2 - mass**2)
        beta = momentum / energy

        # Simplified calculation
        theta0 = 13.6 / (beta * momentum) * np.sqrt(thickness / 10.0)  # mrad
        return np.random.normal(0, theta0 / 1000)  # Convert to radians

    def _apply_scattering(self, momentum: np.ndarray, scatter_angle: float) -> np.ndarray:
        """Apply scattering to momentum vector"""
        # Random azimuthal angle
        phi = np.random.uniform(0, 2 * np.pi)

        # Create rotation matrix
        cos_theta = np.cos(scatter_angle)
        sin_theta = np.sin(scatter_angle)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        # Simple rotation around random axis
        scattered = momentum.copy()
        scattered[0] += momentum[2] * sin_theta * cos_phi
        scattered[1] += momentum[2] * sin_theta * sin_phi
        scattered[2] *= cos_theta

        return scattered

    def _propagate_to_next_layer(self, position: np.ndarray, momentum: np.ndarray,
                                radius: float) -> np.ndarray:
        """Propagate particle to next detector layer"""
        # Simple straight-line propagation
        p_mag = np.linalg.norm(momentum)
        if p_mag == 0:
            return position

        direction = momentum / p_mag

        # Calculate intersection with cylinder at given radius
        current_r = np.sqrt(position[0]**2 + position[1]**2)
        if current_r >= radius:
            return position

        # Propagate outward
        scale_factor = radius / current_r
        new_position = position.copy()
        new_position[0] *= scale_factor
        new_position[1] *= scale_factor

        return new_position

    def _calculate_hit_time(self, distance: float, energy: float, entanglement_strength: float) -> float:
        """Calculate hit time with EG-QGEM corrections"""
        # Speed of light in cm/ns
        c = 29.98

        # Classical time
        classical_time = distance / c

        # EG-QGEM time correction due to entanglement
        time_correction = self.entanglement_coupling * entanglement_strength * energy

        return classical_time + time_correction

    def simulate_event(self, event_type: str, center_of_mass_energy: float,
                      num_particles: int = 2) -> Dict[str, Any]:
        """
        Simulate complete collision event in detector

        Args:
            event_type: Type of event ('ee_to_mumu', 'pp_to_higgs', etc.)
            center_of_mass_energy: CM energy in GeV
            num_particles: Number of final state particles

        Returns:
            Dictionary containing event information and hits
        """
        event_data = {
            'event_type': event_type,
            'cm_energy': center_of_mass_energy,
            'particles': [],
            'hits': [],
            'entanglement_signatures': []
        }

        # Generate final state particles based on event type
        if event_type == 'ee_to_mumu':
            particles = self._generate_ee_to_mumu(center_of_mass_energy)
        elif event_type == 'pp_to_higgs':
            particles = self._generate_pp_to_higgs(center_of_mass_energy)
        elif event_type == 'entangled_pair':
            particles = self._generate_entangled_pair(center_of_mass_energy)
        else:
            particles = self._generate_generic_event(center_of_mass_energy, num_particles)

        # Simulate detector response for each particle
        for particle in particles:
            hits = self.simulate_particle_propagation(
                particle['type'],
                particle['energy'],
                particle['momentum'],
                particle['position'],
                particle.get('entanglement_state')
            )
            event_data['hits'].extend(hits)
            event_data['particles'].append(particle)

        # Detect entanglement signatures
        entanglement_sigs = self.detect_entanglement_signatures(event_data['hits'])
        event_data['entanglement_signatures'] = entanglement_sigs

        return event_data

    def _generate_ee_to_mumu(self, cm_energy: float) -> List[Dict]:
        """Generate e+e- -> mu+mu- event"""
        # Simple back-to-back muons
        muon_energy = cm_energy / 2
        muon_momentum = np.sqrt(muon_energy**2 - 0.10566**2)

        # Random polar angle
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2 * np.pi)

        momentum1 = np.array([
            muon_momentum * np.sin(theta) * np.cos(phi),
            muon_momentum * np.sin(theta) * np.sin(phi),
            muon_momentum * np.cos(theta)
        ])
        momentum2 = -momentum1

        particles = [
            {
                'type': 'muon',
                'charge': -1,
                'energy': muon_energy,
                'momentum': momentum1,
                'position': np.array([0.0, 0.0, 0.0]),
                'entanglement_state': None
            },
            {
                'type': 'muon',
                'charge': 1,
                'energy': muon_energy,
                'momentum': momentum2,
                'position': np.array([0.0, 0.0, 0.0]),
                'entanglement_state': None
            }
        ]

        return particles

    def _generate_pp_to_higgs(self, cm_energy: float) -> List[Dict]:
        """Generate pp -> H + X event with Higgs decay"""
        # Simplified Higgs production
        higgs_mass = 125.0  # GeV
        higgs_energy = min(cm_energy * 0.3, higgs_mass + 50.0)

        # Higgs decay to two photons (simplified)
        photon_energy = higgs_energy / 2

        # Random angles
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2 * np.pi)

        momentum1 = photon_energy * np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])
        momentum2 = -momentum1

        particles = [
            {
                'type': 'photon',
                'charge': 0,
                'energy': photon_energy,
                'momentum': momentum1,
                'position': np.array([0.0, 0.0, 0.0]),
                'entanglement_state': None
            },
            {
                'type': 'photon',
                'charge': 0,
                'energy': photon_energy,
                'momentum': momentum2,
                'position': np.array([0.0, 0.0, 0.0]),
                'entanglement_state': None
            }
        ]

        return particles

    def _generate_entangled_pair(self, cm_energy: float) -> List[Dict]:
        """Generate quantum entangled particle pair"""
        # Create entangled state (Bell state)
        entangled_state1 = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
        entangled_state2 = np.array([1/np.sqrt(2), 0, 0, -1/np.sqrt(2)])

        particle_energy = cm_energy / 2
        momentum_mag = particle_energy  # Assume massless for simplicity

        # Random direction
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2 * np.pi)

        momentum1 = momentum_mag * np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])
        momentum2 = -momentum1

        particles = [
            {
                'type': 'electron',
                'charge': -1,
                'energy': particle_energy,
                'momentum': momentum1,
                'position': np.array([0.0, 0.0, 0.0]),
                'entanglement_state': entangled_state1
            },
            {
                'type': 'electron',
                'charge': 1,
                'energy': particle_energy,
                'momentum': momentum2,
                'position': np.array([0.0, 0.0, 0.0]),
                'entanglement_state': entangled_state2
            }
        ]

        return particles

    def _generate_generic_event(self, cm_energy: float, num_particles: int) -> List[Dict]:
        """Generate generic multi-particle event"""
        particles = []

        for i in range(num_particles):
            # Random particle type
            particle_types = ['electron', 'muon', 'photon']
            particle_type = np.random.choice(particle_types)

            # Random energy distribution
            energy = cm_energy * np.random.exponential(0.3)
            energy = min(energy, cm_energy * 0.8)

            # Random momentum direction
            theta = np.random.uniform(0, np.pi)
            phi = np.random.uniform(0, 2 * np.pi)

            if particle_type == 'electron':
                mass = 0.000511
            elif particle_type == 'muon':
                mass = 0.10566
            else:
                mass = 0.0

            momentum_mag = np.sqrt(max(0, energy**2 - mass**2))
            momentum = momentum_mag * np.array([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta)
            ])

            particles.append({
                'type': particle_type,
                'charge': np.random.choice([-1, 1]) if particle_type != 'photon' else 0,
                'energy': energy,
                'momentum': momentum,
                'position': np.array([0.0, 0.0, 0.0]),
                'entanglement_state': None
            })

        return particles

    def detect_entanglement_signatures(self, hits: List[ParticleHit]) -> List[Dict]:
        """Detect signatures of quantum entanglement in detector hits"""
        signatures = []

        # Look for correlated energy depositions
        energy_correlations = self._analyze_energy_correlations(hits)
        if energy_correlations['significance'] > 3.0:
            signatures.append({
                'type': 'energy_correlation',
                'significance': energy_correlations['significance'],
                'details': energy_correlations
            })

        # Look for time correlations
        time_correlations = self._analyze_time_correlations(hits)
        if time_correlations['significance'] > 3.0:
            signatures.append({
                'type': 'time_correlation',
                'significance': time_correlations['significance'],
                'details': time_correlations
            })

        # Look for angular correlations
        angular_correlations = self._analyze_angular_correlations(hits)
        if angular_correlations['significance'] > 2.5:
            signatures.append({
                'type': 'angular_correlation',
                'significance': angular_correlations['significance'],
                'details': angular_correlations
            })

        return signatures

    def _analyze_energy_correlations(self, hits: List[ParticleHit]) -> Dict:
        """Analyze energy correlations between hits"""
        if len(hits) < 2:
            return {'significance': 0.0, 'correlation': 0.0}

        energies = [hit.energy for hit in hits]
        entanglements = [hit.entanglement_strength for hit in hits]

        # Calculate correlation between energy and entanglement strength
        if len(set(entanglements)) > 1:
            correlation, p_value = stats.pearsonr(energies, entanglements)
            significance = abs(correlation) * np.sqrt(len(hits) - 2) / np.sqrt(1 - correlation**2)
        else:
            correlation = 0.0
            significance = 0.0
            p_value = 1.0

        return {
            'significance': significance,
            'correlation': correlation,
            'p_value': p_value,
            'num_hits': len(hits)
        }

    def _analyze_time_correlations(self, hits: List[ParticleHit]) -> Dict:
        """Analyze time correlations between hits"""
        if len(hits) < 2:
            return {'significance': 0.0}

        times = [hit.time for hit in hits]
        entanglements = [hit.entanglement_strength for hit in hits]

        # Look for simultaneous hits with high entanglement
        simultaneous_threshold = 1e-9  # 1 ns
        entangled_pairs = 0
        total_pairs = 0

        for i in range(len(hits)):
            for j in range(i+1, len(hits)):
                total_pairs += 1
                time_diff = abs(hits[i].time - hits[j].time)
                if (time_diff < simultaneous_threshold and
                    hits[i].entanglement_strength > 0.1 and
                    hits[j].entanglement_strength > 0.1):
                    entangled_pairs += 1

        if total_pairs > 0:
            fraction = entangled_pairs / total_pairs
            # Use binomial test for significance
            significance = abs(fraction - 0.1) * np.sqrt(total_pairs)
        else:
            significance = 0.0
            fraction = 0.0

        return {
            'significance': significance,
            'simultaneous_fraction': fraction,
            'entangled_pairs': entangled_pairs,
            'total_pairs': total_pairs
        }

    def _analyze_angular_correlations(self, hits: List[ParticleHit]) -> Dict:
        """Analyze angular correlations between hits"""
        if len(hits) < 2:
            return {'significance': 0.0}

        # Calculate angles between hit positions
        positions = [np.array(hit.position) for hit in hits]
        entanglements = [hit.entanglement_strength for hit in hits]

        back_to_back_pairs = 0
        total_pairs = 0

        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                total_pairs += 1

                # Calculate angle between position vectors
                pos1 = positions[i][:2]  # x, y only
                pos2 = positions[j][:2]

                if np.linalg.norm(pos1) > 0 and np.linalg.norm(pos2) > 0:
                    cos_angle = np.dot(pos1, pos2) / (np.linalg.norm(pos1) * np.linalg.norm(pos2))
                    angle = np.arccos(np.clip(cos_angle, -1, 1))

                    # Check for back-to-back configuration (π ± 0.2 rad)
                    if (abs(angle - np.pi) < 0.2 and
                        entanglements[i] > 0.05 and entanglements[j] > 0.05):
                        back_to_back_pairs += 1

        if total_pairs > 0:
            fraction = back_to_back_pairs / total_pairs
            # Expected fraction for random events
            expected_fraction = 0.1
            significance = abs(fraction - expected_fraction) * np.sqrt(total_pairs)
        else:
            significance = 0.0
            fraction = 0.0

        return {
            'significance': significance,
            'back_to_back_fraction': fraction,
            'back_to_back_pairs': back_to_back_pairs,
            'total_pairs': total_pairs
        }

    def train_event_classifier(self, training_events: List[Dict]) -> Dict:
        """Train machine learning classifier for event identification"""
        features = []
        labels = []

        for event in training_events:
            # Extract features from event
            event_features = self._extract_event_features(event)
            features.append(event_features)
            labels.append(event['event_type'])

        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(labels)

        # Scale features
        X_scaled = self.feature_scaler.fit_transform(X)

        # Train classifier
        self.event_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.event_classifier.fit(X_scaled, y)

        # Calculate training accuracy
        train_accuracy = self.event_classifier.score(X_scaled, y)

        return {
            'training_accuracy': train_accuracy,
            'num_features': X.shape[1],
            'num_events': len(training_events),
            'feature_importance': self.event_classifier.feature_importances_
        }

    def _extract_event_features(self, event: Dict) -> List[float]:
        """Extract features from event for ML classification"""
        features = []

        hits = event['hits']
        particles = event['particles']

        # Basic event features
        features.append(len(hits))  # Number of hits
        features.append(len(particles))  # Number of particles
        features.append(event['cm_energy'])  # Center of mass energy

        # Energy features
        total_energy = sum(hit.energy for hit in hits)
        features.append(total_energy)
        features.append(total_energy / event['cm_energy'] if event['cm_energy'] > 0 else 0)

        if hits:
            energies = [hit.energy for hit in hits]
            features.append(np.mean(energies))
            features.append(np.std(energies))
            features.append(np.max(energies))
            features.append(np.min(energies))
        else:
            features.extend([0, 0, 0, 0])

        # Spatial features
        if hits:
            positions = [hit.position for hit in hits]
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            z_coords = [pos[2] for pos in positions]

            features.append(np.std(x_coords))
            features.append(np.std(y_coords))
            features.append(np.std(z_coords))
        else:
            features.extend([0, 0, 0])

        # Entanglement features
        entanglements = [hit.entanglement_strength for hit in hits]
        if entanglements:
            features.append(np.mean(entanglements))
            features.append(np.max(entanglements))
            features.append(np.sum(entanglements))
            features.append(len([e for e in entanglements if e > 0.1]))
        else:
            features.extend([0, 0, 0, 0])

        # Timing features
        if hits:
            times = [hit.time for hit in hits]
            features.append(np.mean(times))
            features.append(np.std(times))
            features.append(np.max(times) - np.min(times))
        else:
            features.extend([0, 0, 0])

        return features

    def classify_event(self, event: Dict) -> Dict:
        """Classify event using trained ML model"""
        if self.event_classifier is None:
            return {'error': 'Classifier not trained'}

        # Extract features
        features = self._extract_event_features(event)
        X = np.array([features])
        X_scaled = self.feature_scaler.transform(X)

        # Make prediction
        prediction = self.event_classifier.predict(X_scaled)[0]
        probabilities = self.event_classifier.predict_proba(X_scaled)[0]

        # Get class names
        classes = self.event_classifier.classes_

        return {
            'predicted_type': prediction,
            'confidence': np.max(probabilities),
            'probabilities': dict(zip(classes, probabilities))
        }

    def generate_detector_response_plots(self, event: Dict, save_path: Optional[str] = None) -> Dict:
        """Generate comprehensive detector response plots"""
        hits = event['hits']
        if not hits:
            return {'error': 'No hits to plot'}

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Energy vs radius plot
        radii = [np.sqrt(hit.position[0]**2 + hit.position[1]**2) for hit in hits]
        energies = [hit.energy for hit in hits]
        axes[0, 0].scatter(radii, energies, alpha=0.7)
        axes[0, 0].set_xlabel('Radius (cm)')
        axes[0, 0].set_ylabel('Energy (GeV)')
        axes[0, 0].set_title('Energy vs Radius')

        # Hit time distribution
        times = [hit.time for hit in hits]
        axes[0, 1].hist(times, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Time (ns)')
        axes[0, 1].set_ylabel('Number of Hits')
        axes[0, 1].set_title('Hit Time Distribution')

        # Entanglement strength distribution
        entanglements = [hit.entanglement_strength for hit in hits]
        axes[0, 2].hist(entanglements, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 2].set_xlabel('Entanglement Strength')
        axes[0, 2].set_ylabel('Number of Hits')
        axes[0, 2].set_title('Entanglement Distribution')

        # 2D hit pattern (x-y view)
        x_coords = [hit.position[0] for hit in hits]
        y_coords = [hit.position[1] for hit in hits]
        scatter = axes[1, 0].scatter(x_coords, y_coords, c=energies, alpha=0.7, cmap='viridis')
        axes[1, 0].set_xlabel('X (cm)')
        axes[1, 0].set_ylabel('Y (cm)')
        axes[1, 0].set_title('Hit Pattern (X-Y View)')
        plt.colorbar(scatter, ax=axes[1, 0], label='Energy (GeV)')

        # Energy vs entanglement correlation
        axes[1, 1].scatter(entanglements, energies, alpha=0.7)
        axes[1, 1].set_xlabel('Entanglement Strength')
        axes[1, 1].set_ylabel('Energy (GeV)')
        axes[1, 1].set_title('Energy vs Entanglement')

        # Layer occupancy
        layer_ids = [hit.layer_id for hit in hits]
        unique_layers, counts = np.unique(layer_ids, return_counts=True)
        axes[1, 2].bar(unique_layers, counts, alpha=0.7, edgecolor='black')
        axes[1, 2].set_xlabel('Layer ID')
        axes[1, 2].set_ylabel('Number of Hits')
        axes[1, 2].set_title('Layer Occupancy')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return {
            'figure': fig,
            'total_hits': len(hits),
            'total_energy': sum(energies),
            'max_entanglement': max(entanglements) if entanglements else 0,
            'active_layers': len(unique_layers)
        }
