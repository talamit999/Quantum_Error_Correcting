import numpy as np
from squanch import *
import qutip as qt

# Define basis states
_0 = np.array([1, 0], dtype=np.complex64)
_1 = np.array([0, 1], dtype=np.complex64)
_M0 = np.outer(_0, _0)
_M1 = np.outer(_1, _1)

a = 0.6  # Amplitude for |0> in qubit0
b = np.sqrt(1 - a ** 2)  # Amplitude for |1> in qubit0

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


def define_qsystem():
    """
    Defines the initial quantum system for the three-qubit repetition code.

    Returns:
        tuple: The initialized quantum system and its qubits.
    """
    state = np.kron(np.kron(np.kron(np.kron(_Mpsi, _M0), _M0), _M0), _M0)
    qsystem = QSystem(num_qubits=5, state=state)
    qubit0 = Qubit(qsystem, 0)
    qubit1 = Qubit(qsystem, 1)
    qubit2 = Qubit(qsystem, 2)
    a1 = Qubit(qsystem, 3)
    a2 = Qubit(qsystem, 4)
    return qsystem #, qubit0, qubit1, qubit2, a1, a2


def apply_error(qsystem, flipped_indices):
    """
    Applies an error operator to the quantum system based on the indices of qubits with bit-flip errors.

    Args:
        qsystem: The quantum system object with the attribute num_qubits.
        flipped_indices (list): A list of indices where the Pauli-X (bit-flip) error should be applied.
    """
    op = _I
    for i in range(qsystem.num_qubits):
        if i == 0:
            op = _X if i in flipped_indices else _I
        else:
            op = np.kron(op, _X if i in flipped_indices else _I)
    qsystem.apply(op)


def three_qubits_rep_code(qsystem, flipped_indices):
    qubit0, qubit1, qubit2, a1, a2 = qsystem.qubit(0), qsystem.qubit(1), qsystem.qubit(2), qsystem.qubit(3), qsystem.qubit(4)

    # Encoding
    CNOT(qubit0, qubit1)
    CNOT(qubit1, qubit2)

    # Error
    apply_error(qsystem, flipped_indices)

    # Decoding
    H(a1)
    H(a2)

    CU(a1, qubit0, _Z)
    CU(a1, qubit1, _Z)

    CU(a2, qubit1, _Z)
    CU(a2, qubit2, _Z)

    H(a1)
    H(a2)

    # Measurement
    measurement_result = a1.measure()
    print(f"Measured qubit a1: {measurement_result}")
    measurement_result2 = a2.measure()
    print(f"Measured qubit a2: {measurement_result2}")

def fix_error(qsystem):
    """
    Can only fix one qubit flip.

    Args:
        qsystem: Current QSystem after measuring ancillas.

    Returns:
        QSystem: The corrected quantum system.
    """
    CNOT(qsystem.qubit(0), qsystem.qubit(1))
    CNOT(qsystem.qubit(0), qsystem.qubit(2))
    TOFFOLI(qsystem.qubit(1), qsystem.qubit(2), qsystem.qubit(0))
    qsystem.measure_qubit(1)
    qsystem.measure_qubit(2)

    measure0 = gates.expand(_M0, 0, qsystem.num_qubits, "0" + str(0) + str(qsystem.num_qubits))
    prob0 = np.trace(np.dot(measure0, qsystem.state))
    print(f"a is {np.sqrt(np.abs(prob0)):.2f}")


if __name__ == "__main__":
    qsys = define_qsystem()
    three_qubits_rep_code(qsystem=qsys, flipped_indices=[2])
    fix_error(qsys)


