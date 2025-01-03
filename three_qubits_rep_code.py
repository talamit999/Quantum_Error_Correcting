import numpy as np
from squanch import *

_0 = np.array([1, 0], dtype=np.complex64)
_1 = np.array([0, 1], dtype=np.complex64)
_M0 = np.outer(_0, _0)
_M1 = np.outer(_1, _1)

a = 0.6  # Amplitude for |0> in qubit0
b = 0.8  # Amplitude for |1> in qubit0
_psi = np.array([a, b], dtype=np.complex64)
_Mpsi = np.outer(_psi, _psi)

# Identity operator
_I = np.array([[1, 0],
               [0, 1]])

# Pauli-Z gate (phase flip)
_Z = np.array([[1, 0],
               [0, -1]])

# Pauli-X gate (bit flip)
_X = np.array([[0, 1],
               [1, 0]])

_X1_I2_I3 = np.kron(np.kron(np.kron(np.kron(_X, _I), _I), _I), _I)
_X1_X2_I3 = np.kron(np.kron(np.kron(np.kron(_X, _X), _I), _I), _I)



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

def three_qubits_rep_code(operator):

    state = np.kron(np.kron(np.kron(np.kron(_Mpsi, _M0), _M0), _M0), _M0)
    num_qubits = 5
    qsystem = QSystem(num_qubits, state=state)
    qubit0 = Qubit(qsystem, 0)
    qubit1 = Qubit(qsystem, 1)
    qubit2 = Qubit(qsystem, 2)
    a1 = Qubit(qsystem, 3)
    a2 = Qubit(qsystem, 4)

    #print(f"Initial quantum state (|psi>):\n", qsystem.state)

    CNOT(qubit0, qubit1)
    #print("Quantum state after applying CNOT gate:\n", qsystem.state)

    CNOT(qubit1, qubit2)
    #print("Quantum state after applying CNOT gate:\n", qsystem.state)


    E(qsystem, operator)

    H(a1)
    H(a2)

    # print("Quantum state:\n", syndrome_sys.state)
    CU(a1, qubit0, _Z)
    CU(a1, qubit1, _Z)

    CU(a2, qubit1, _Z)
    CU(a2, qubit2, _Z)

    H(a1)
    H(a2)

    measurement_result = a1.measure()
    #print(f"Measured qubit a1: {measurement_result}")
    measurement_result2 = a2.measure()
    #print(f"Measured qubit a2: {measurement_result2}")
    return measurement_result, measurement_result2



if __name__ == "__main__":
    for i in range(100):
        print(i)
        a1, a2 = three_qubits_rep_code(_X1_X2_I3)
        if a1!=0 or a2!=1:
            print("Error!")