import numpy as np
from squanch import *

_0 = np.array([1, 0], dtype = np.complex64)
_1 = np.array([0, 1], dtype = np.complex64)
_M0 = np.outer(_0, _0)
_M1 = np.outer(_1, _1)

a = 0.6  # Amplitude for |0> in qubit0
b = 0.8  # Amplitude for |1> in qubit0
_psi = np.array([a, b], dtype = np.complex64)
_Mpsi = np.outer(_psi, _psi)

# Identity operator
_I = np.array([[1, 0],
               [0, 1]])

# Pauli-Z gate (phase flip)
_Z = np.array([[1, 0],
               [0, -1]])

_Z8 = np.kron(np.kron(_Z, _I), _I)


def E(qsystem, operator=None):
    """
    Applies an operator E to the current quantum system.
    If no operator is provided, the identity matrix is used (np.eye).

    Parameters:
    qsystem (QSystem): The quantum system to apply the operator on.
    operator (np.ndarray, optional): The matrix operator to apply. Defaults to np.eye (identity matrix).
    """
    # If no operator is provided, use the identity matrix
    if operator is None:
        operator = np.eye(2 ** qsystem.num_qubits, dtype=np.complex64)

    # Apply the operator using squanch's apply method
    qsystem.apply(operator)




if __name__ == "__main__":
    # Example: Single qubit with 50-50 measurement probability

    '''is_init_state_0 = 0
    if is_init_state_0 == 1:
        initial_state = np.outer(_0, _0)
    else:
        initial_state = np.outer(_1, _1)
    '''
    state = np.kron(np.kron(_Mpsi, _M0), _M0)
    num_qubits = 5
    qsystem = QSystem(num_qubits, state=state)
    qubit0 = Qubit(qsystem, 0)
    qubit1 = Qubit(qsystem, 1)
    qubit2 = Qubit(qsystem, 2)

    #print(f"Initial quantum state (|psi>):\n", qsystem.state)

    CNOT(qubit0, qubit1)
    #print("Quantum state after applying CNOT gate:\n", qsystem.state)

    CNOT(qubit1, qubit2)

    E(qsystem)

    syndrome_sys = QSystem(num_qubits=2, state=np.kron(_M0, _M0))
    a1 = Qubit(syndrome_sys, 0)
    a2 = Qubit(syndrome_sys, 1)
    H(a1)
    H(a2)
    #print("Quantum state:\n", syndrome_sys.state)
    CPHASE(a1, qubit0, np.pi)
    #CU(a1, qubit1, _Z)
    #qsystem.control(a1, Z(qubit0))

    '''
    Z(qubit0)
    print("Quantum state after applying Z gate:\n", qsystem.state)

    # Apply Hadamard gate to put the qubit into superposition
    H(qubit0)

    print("Quantum state after applying Hadamard gate:\n", qsystem.state)
    
    # Measure the qubit
    measurement_result = qubit.measure()
    print(f"Measured qubit: {measurement_result}")
    print("Quantum state after measurement:\n", qsystem.state)
    '''

'''
    system_size = 2  # 2 qubits per system
    num_systems = 3  # 3 separate systems in the stream

    # Test the shared_hilbert_space method
    hilbert_space = QStream.shared_hilbert_space(system_size, num_systems)

    print("Shared Hilbert Space Shape:", hilbert_space.shape)
    print("Shared Hilbert Space Content (first system):\n", hilbert_space[0])

'''