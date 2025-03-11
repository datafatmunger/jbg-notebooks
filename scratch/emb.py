import openai
import json
import os
import numpy as np
from scipy.linalg import expm

# Define the file to store embeddings
EMBEDDINGS_FILE = "embeddings.json"

# Initialize OpenAI client
client = openai.OpenAI(api_key="your-api-key-here")

def load_embeddings():
    """Loads stored embeddings from a JSON file if it exists."""
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_embeddings(embeddings):
    """Saves embeddings to a JSON file."""
    with open(EMBEDDINGS_FILE, "w") as f:
        json.dump(embeddings, f, indent=4)

def get_embeddings(word, stored_embeddings):
    """Retrieves embedding from cache or fetches it from OpenAI if not cached."""
    if word in stored_embeddings:
        print(f"Loaded cached embedding for '{word}'")
        return stored_embeddings[word]

    print(f"Fetching embedding for '{word}' from OpenAI...")
    response = client.embeddings.create(
        input=word,
        model="text-embedding-ada-002"
    )
    embedding = response.data[0].embedding

    # Store the new embedding
    stored_embeddings[word] = embedding
    save_embeddings(stored_embeddings)

    return embedding

def pair_real_to_complex(vector):
    """
    Converts a 1536-dimensional real vector into a 768-dimensional complex vector.
    Uses pairs of adjacent reals to form complex numbers.
    """
    vector = np.array(vector)
    if len(vector) % 2 != 0:
        raise ValueError("Vector length must be even to form complex pairs.")

    real_parts = vector[0::2]  # Take even indices as real parts
    imag_parts = vector[1::2]  # Take odd indices as imaginary parts
    complex_vector = real_parts + 1j * imag_parts

    # Normalize to ensure quantum state validity
    norm = np.linalg.norm(complex_vector)
    return complex_vector / norm if norm != 0 else complex_vector

def transverse_field_hamiltonian(dim):
    """Generates a transverse field Hamiltonian H_I with Pauli-X terms."""
    H_I = np.zeros((dim, dim), dtype=complex)
    for i in range(dim - 1):
        H_I[i, i + 1] = 1.0  # X interaction
        H_I[i + 1, i] = 1.0  # X interaction (symmetric)
    return -H_I  # Negative sign for transverse field

def encoding_hamiltonian(complex_vector):
    """Constructs the final Hamiltonian H_final from the complex embedding."""
    return np.outer(complex_vector, complex_vector.conj())  # Projector-based H_final

def annealing_schedule(t, T):
    """Linear annealing schedule s(t) = t/T."""
    return t / T

def time_evolution_operator(H_I, H_final, T, steps=100):
    """Computes the time evolution unitary U using Trotterization."""
    dim = H_I.shape[0]
    U = np.eye(dim, dtype=complex)  # Start with identity matrix
    dt = T / steps  # Time step
    
    for t in np.linspace(0, T, steps):
        s_t = annealing_schedule(t, T)
        H_t = (1 - s_t) * H_I + s_t * H_final  # Interpolated Hamiltonian
        U = expm(-1j * H_t * dt) @ U  # Time evolution step
    
    return U

def quantum_encoding(real_vector, T=1.0, steps=100):
    """Encodes a real-valued embedding into a quantum state using complex amplitudes and evolution."""
    complex_vector = pair_real_to_complex(real_vector)
    dim = len(complex_vector)

    H_I = transverse_field_hamiltonian(dim)
    H_final = encoding_hamiltonian(complex_vector)
    U = time_evolution_operator(H_I, H_final, T, steps)

    # Start in equal superposition state
    initial_state = np.ones(dim, dtype=complex) / np.sqrt(dim)

    # Apply time evolution
    encoded_state = U @ initial_state
    return encoded_state

def quantum_fidelity(state1, state2):
    """Computes quantum fidelity between two quantum states."""
    return np.abs(np.vdot(state1, state2))**2  # |<ψ|ϕ>|²

def hadamard_gate():
    """Single-qubit Hadamard gate."""
    return np.array([[1, 1], [1, -1]]) / np.sqrt(2)

def controlled_swap(num_qubits):
    """Constructs a controlled-SWAP gate for `num_qubits` per state."""
    dim = 2 ** (1 + 2 * num_qubits)  # 1 control + 2 registers
    CSWAP = np.eye(dim)

    for i in range(2 ** num_qubits):  # Iterate over basis states
        swap_a = (1 << num_qubits) + i
        swap_b = (1 << num_qubits) + (i + (1 << num_qubits))
        CSWAP[[swap_a, swap_b], [swap_b, swap_a]] = 1  # Swap states
        CSWAP[swap_a, swap_a] = CSWAP[swap_b, swap_b] = 0

    return CSWAP

def pad_to_power_of_2(state):
    """Pads a state vector to the next power of 2 (if necessary)."""
    original_size = len(state)
    next_power_of_2 = 2 ** int(np.ceil(np.log2(original_size)))  # Find next power of 2

    if original_size == next_power_of_2:
        return state  # No padding needed

    # Pad with zeros to reach the next power of 2
    padded_state = np.zeros(next_power_of_2, dtype=complex)
    padded_state[:original_size] = state  # Copy original state
    return padded_state

def swap_test(state1, state2):
    """Performs a swap test for two quantum states, padding if necessary."""
    state1 = pad_to_power_of_2(state1) / np.linalg.norm(state1)  # Normalize after padding
    state2 = pad_to_power_of_2(state2) / np.linalg.norm(state2)

    # Number of qubits needed (log2 of padded state size)
    num_qubits = int(np.log2(len(state1)))

    # Define full state: |0⟩ ⊗ |ψ⟩ ⊗ |ϕ⟩
    control_qubit = np.array([1, 0])  # |0⟩
    full_state = np.kron(control_qubit, np.kron(state1, state2))
    full_state = full_state.reshape(-1, 1)  # Column vector

    # Apply Hadamard to control qubit
    H = np.kron(hadamard_gate(), np.eye(len(state1) ** 2))
    state_after_H = H @ full_state

    # Apply Controlled-SWAP gate
    CSWAP = controlled_swap(num_qubits)
    state_after_CSWAP = CSWAP @ state_after_H

    # Apply Hadamard to control qubit again
    state_final = H @ state_after_CSWAP

    # Measure control qubit probability
    num_control_states = len(state1) ** 2 // 2
    P0 = np.sum(np.abs(state_final[:num_control_states])**2)
    P1 = 1 - P0

    # Extract fidelity
    fidelity = 2 * P0 - 1
    return fidelity, P0, P1

def main():
    words = ["cat", "dog", "car"]
    
    # Load existing embeddings
    stored_embeddings = load_embeddings()

    # Retrieve embeddings
    embeddings = {word: get_embeddings(word, stored_embeddings) for word in words}

    # Encode into quantum states
    quantum_states = {word: quantum_encoding(embeddings[word]) for word in words}

    # Compute quantum fidelities
    #fidelity_cat_dog = quantum_fidelity(quantum_states["cat"], quantum_states["dog"])
    #fidelity_cat_car = quantum_fidelity(quantum_states["cat"], quantum_states["car"])
    swap_cat_dog, cat_dog_p0, cat_dog_p1 = swap_test(quantum_states["cat"], quantum_states["dog"])
    swap_cat_car, cat_car_p0, cat_car_p1 = swap_test(quantum_states["cat"], quantum_states["car"])

    # Compute classical cosine similarities
    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    cosine_cat_dog = cosine_similarity(embeddings["cat"], embeddings["dog"])
    cosine_cat_car = cosine_similarity(embeddings["cat"], embeddings["car"])

    # Print results
    print(f"Cosine similarity between 'cat' and 'dog': {cosine_cat_dog:.4f}")
    print(f"Cosine similarity between 'cat' and 'car': {cosine_cat_car:.4f}")
    #print(f"Quantum Fidelity between 'cat' and 'dog': {fidelity_cat_dog:.4f}")
    #print(f"Quantum Fidelity between 'cat' and 'car': {fidelity_cat_car:.4f}")
    print(f"Quantum Swap Test between 'cat' and 'dog': {swap_cat_dog:.4f}")
    print(f"Quantum Swap Test between 'cat' and 'car': {swap_cat_car:.4f}")

if __name__ == "__main__":
    main()

