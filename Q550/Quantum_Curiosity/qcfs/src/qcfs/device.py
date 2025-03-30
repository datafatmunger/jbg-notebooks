import os
import pennylane as qml
from qiskit_ibm_runtime import QiskitRuntimeService
from qcfs.enums import QuantumBackend

def get_device(backend, num_features, shots=None):
    """
    Get the appropriate Pennylane device based on the backend.

    Args:
        backend (QuantumBackend): The quantum backend to use (e.g., DEFAULT, LIGHTNING, QISKIT).
        num_features (int): The number of qubits (wires) required for the device.

    Returns:
        qml.Device: The configured Pennylane device.
    """
    if backend == QuantumBackend.LIGHTNING:
        return qml.device("lightning.gpu", wires=num_features) if shots is None else qml.device("lightning.gpu", wires=num_features, shots=shots)
    elif backend == QuantumBackend.QISKIT_AER:
        return qml.device("qiskit.aer", wires=num_features) if shots is None else qml.device("qiskit.aer", wires=num_features, shots=shots)
    elif backend == QuantumBackend.QISKIT_IBM:
        # Load the API token from the environment
        api_token = os.environ.get("IBM_QUANTUM_API_TOKEN")
        if not api_token:
            raise ValueError("IBM_QUANTUM_API_TOKEN environment variable is not set.")

        # Use Qiskit Runtime Service with the token
        service = QiskitRuntimeService(channel="ibm_quantum", token=api_token)
        backend = service.least_busy(operational=True, simulator=False)
        return qml.device("qiskit.ibmq", wires=num_features, backend=backend) if shots is None else qml.device("qiskit.ibmq", wires=num_features, backend=backend, shots=shots)
    else:
        return qml.device("default.qubit", wires=num_features) if shots is None else qml.device("default.qubit", wires=num_features, shots=shots)