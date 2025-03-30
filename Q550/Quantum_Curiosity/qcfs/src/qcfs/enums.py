from enum import Enum

class QuantumBackend(Enum):
    DEFAULT = "default"
    LIGHTNING = "lightning"
    QISKIT_AER = "aer"
    QISKIT_IBM = "ibm"