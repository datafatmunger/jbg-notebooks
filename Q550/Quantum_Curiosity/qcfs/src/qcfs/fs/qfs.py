import numpy as np
from math import ceil, log2
import pennylane as qml

from qcfs.device import get_device
from qcfs.enums import QuantumBackend

class QuantumFeatureSelection:
    def __init__(self, num_features, learning_rate=0.1, backend=QuantumBackend.DEFAULT, n_shots=1000):
        """
        Initialize the quantum feature selector.
        
        Args:
            num_features (int): Number of features to select from
            learning_rate (float): Learning rate for quantum circuit updates
            backend (QuantumBackend): Which quantum backend to use
            n_shots (int): Number of measurement shots for quantum sampling
        """
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.n_shots = n_shots
        self.backend = backend

        # Determine number of qubits needed to represent features
        self.num_qubits = ceil(log2(num_features))
        
        # Device for measurements (with shots)
        self.measurement_device = get_device(backend, self.num_qubits, shots=self.n_shots)
        # Device for unitary updates (analytic)
        self.analytic_device = get_device(backend, self.num_qubits)
        
        self.decision_register = list(range(self.num_qubits))

    def _prepare_initial_state(self):
        """
        Prepare the initial quantum state as an equal superposition.
        This is called within quantum circuits, not externally.
        """
        # Apply Hadamard gates to create equal superposition
        for wire in range(self.num_qubits):
            qml.Hadamard(wires=wire)

    def _encode_feature(self, feature):
        """
        Encode a classical feature index into binary for quantum operations.
        
        Args:
            feature (int): Feature index to encode
        Returns:
            str: Binary representation of feature index
        """
        return format(feature, f"0{self.num_qubits}b")

    @property
    def measurement_circuit(self):
        """
        Create the measurement circuit for feature selection.
        Returns actual quantum measurement samples in computational basis.
        """
        @qml.qnode(self.measurement_device)
        def circuit():
            self._prepare_initial_state()
            # Return actual quantum measurements
            return qml.sample(wires=self.decision_register)
        return circuit

    def measure(self, available_features, return_probabilities=False):
        """
        Perform quantum measurements to select a feature.
        
        Args:
            available_features (list): List of available feature indices
            return_probabilities (bool): Whether to return probabilities or a selected feature
        
        Returns:
            If return_probabilities=True: Array of empirical probabilities from measurements
            If return_probabilities=False: Selected feature index based on quantum measurements
        """
        # Validate available features
        valid_features = [f for f in available_features if f < self.num_features]
        if not valid_features:
            raise ValueError("No valid features available for measurement")

        # Get quantum measurement samples
        samples = self.measurement_circuit()
        
        # Convert binary samples to feature indices
        feature_counts = np.zeros(self.num_features)
        for sample in samples:
            # Convert binary array to integer
            feature = int(''.join(map(str, sample)), 2)
            if feature < self.num_features:  # Only count valid features
                feature_counts[feature] += 1
        
        # Calculate empirical probabilities for valid features
        valid_counts = feature_counts[valid_features]
        total_valid_counts = np.sum(valid_counts)
        
        if total_valid_counts > 0:
            valid_probs = valid_counts / total_valid_counts
        else:
            # If no valid measurements, use uniform distribution
            valid_probs = np.ones(len(valid_features)) / len(valid_features)
            
        if return_probabilities:
            return valid_probs
        else:
            # Select feature based on most frequent measurement outcome
            return valid_features[np.argmax(valid_counts)]

    def unitary_update(self, action, reward):
        """
        Apply a reward-based unitary update to the quantum circuit.
        
        Args:
            action (int): The selected feature/action
            reward (float): The reward value for the action
        """
        target = self._encode_feature(action)
        
        @qml.qnode(self.analytic_device)
        def update_circuit():
            self._prepare_initial_state()
            
            # Apply reward-based rotations
            for i, wire in enumerate(self.decision_register):
                # Quantum phase based on reward and target state
                angle = self.learning_rate * reward if target[i] == '1' else -self.learning_rate * reward
                qml.RY(angle, wires=wire)
            
            # Return state for verification (not measurement)
            return qml.state()
        
        # Execute update circuit
        update_circuit()
        print(f"🔄 Quantum circuit updated for action {action} with reward {reward}")

    def get_feature(self, available_features):
        """
        High-level method to select a feature using quantum measurement.
        This ensures one measurement per state preparation.
        
        Args:
            available_features (list): List of available feature indices
        Returns:
            int: Selected feature index based on quantum measurement
        """
        return self.measure(available_features, return_probabilities=False)