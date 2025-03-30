import numpy as np
import pennylane as qml

from qcfs.device import get_device
from qcfs.enums import QuantumBackend

class QuantumLearner:
    def __init__(self, num_layers, backend=QuantumBackend.DEFAULT):
        self.num_layers = num_layers
        self.backend = backend.value  # Use the value of the enum
        print(f"QuantumLearner initialized with backend: {self.backend}")
        self.weights = None  # Weights will be initialized dynamically
        self.bias = None  # Bias will be initialized dynamically
        self.device = None  # Device will be initialized dynamically
        self.circuit = None  # Circuit will be built dynamically

    def _build_circuit(self, num_features):
        """Build the quantum circuit with parameterized weights."""
        self.device = get_device(self.backend, num_features)

        @qml.qnode(self.device)
        def circuit(inputs, weights, bias):
            # Encode inputs into the quantum state
            for i in range(num_features):
                qml.RY(inputs[i], wires=i)

            # Apply parameterized layers
            for layer in range(self.num_layers):
                for i in range(num_features):
                    qml.RY(weights[layer, i], wires=i)
                for i in range(num_features - 1):
                    qml.CNOT(wires=[i, i + 1])

            # Apply bias rotation to the first qubit
            qml.RY(bias, wires=0)

            # Measure the expectation value of the first qubit
            return qml.expval(qml.PauliZ(0))

        return circuit

    def fit(self, X, y, num_it=10, learning_rate=0.1):
        """Train the quantum model using gradient descent."""
        # Ensure the input is a numeric numpy array
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        # Dynamically adjust the number of features
        num_features = X.shape[1]
        self.weights = qml.numpy.tensor(
            np.random.uniform(-np.pi, np.pi, (self.num_layers, num_features)),
            requires_grad=True
        )
        self.bias = qml.numpy.tensor(np.random.uniform(-np.pi, np.pi), requires_grad=True)
        self.circuit = self._build_circuit(num_features)

        opt = qml.GradientDescentOptimizer(stepsize=learning_rate)

        #print(f"Weights before training: {self.weights}")
        #print(f"Bias before training: {self.bias}")

        for it in range(num_it):
            for i in range(len(X)):
                inputs = X[i]
                target = y[i]

                def cost_fn(weights, bias):
                    prediction = self.circuit(inputs, weights, bias)
                    return (prediction - target) ** 2  # Return a scalar value

                # Perform an optimization step
                self.weights, self.bias = opt.step(cost_fn, self.weights, self.bias)

            # Debugging information
            #print(f"Iteration {it + 1}/{num_it}")
            #print(f"Weights: {self.weights}")
            #print(f"Bias: {self.bias}")

        return self

    def predict(self, X):
        """Predict labels for the given inputs."""
        # Ensure the input is a numeric numpy array
        X = np.array(X, dtype=float)

        predictions = []
        for inputs in X:
            output = self.circuit(inputs, self.weights, self.bias)
            predictions.append(np.sign(output))  # Convert to binary label {-1, 1}
        return np.array(predictions)