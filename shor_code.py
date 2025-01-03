from squanch import *
from scipy.stats import unitary_group
import copy
import numpy as np
import matplotlib.image as image
import matplotlib.pyplot as plt
import multiprocessing


class Alice(Agent):
    '''Alice sends an arbitrary Shor-encoded state to Bob'''

    def __init__(self, qstream, output):
        super().__init__(qstream, output, name="Alice")  # Call parent constructor to initialize name

    def shor_encode(self, qsys):
        # psi is state to send, q1...q8 are ancillas from top to bottom in diagram
        psi, q1, q2, q3, q4, q5, q6, q7, q8 = qsys.qubits
        # Gates are enumerated left to right, top to bottom from figure
        CNOT(psi, q3)
        CNOT(psi, q6)
        H(psi)
        H(q3)
        H(q6)
        CNOT(psi, q1)
        CNOT(psi, q2)
        CNOT(q3, q4)
        CNOT(q3, q5)
        CNOT(q6, q7)
        CNOT(q6, q8)
        return psi, q1, q2, q3, q4, q5, q6, q7, q8

    def run(self):
        for qsys in self.qstream:
            # Send the encoded qubits to Bob
            for qubit in self.shor_encode(qsys):
                self.qsend(bob, qubit)


class DumbAlice(Agent):
    '''DumbAlice sends a state to Bob but forgets to error-correct!'''

    def __init__(self, qstream, output):
        super().__init__(qstream, output, name="DumbAlice")  # Call parent constructor to initialize name

    def run(self):
        for qsys in self.qstream:
            for qubit in qsys.qubits:
                self.qsend(dumb_bob, qubit)


class Bob(Agent):
    '''Bob receives Alice's qubits and applies error correction'''

    def __init__(self, qstream, output):
        super().__init__(qstream, output, name="Bob")  # Call parent constructor to initialize name

    def shor_decode(self, psi, q1, q2, q3, q4, q5, q6, q7, q8):
        # same enumeration as Alice
        CNOT(psi, q1)
        CNOT(psi, q2)
        TOFFOLI(q2, q1, psi)
        CNOT(q3, q4)
        CNOT(q3, q5)
        TOFFOLI(q5, q4, q3)
        CNOT(q6, q7)
        CNOT(q6, q8)
        TOFFOLI(q7, q8, q6)  # Toffoli control qubit order doesn't matter
        H(psi)
        H(q3)
        H(q6)
        CNOT(psi, q3)
        CNOT(psi, q6)
        TOFFOLI(q6, q3, psi)
        return psi  # psi is now Alice's original state

    def run(self):
        measurement_results = []
        for _ in self.qstream:
            # Bob receives 9 qubits representing Alice's encoded state
            received = [self.qrecv(alice) for _ in range(9)]
            # Decode and measure the original state
            psi_true = self.shor_decode(*received)
            measurement_results.append(psi_true.measure())
        self.output["Bob"] = measurement_results  # Save results to shared output


class DumbBob(Agent):
    '''DumbBob receives qubits but doesn't correct errors'''

    def __init__(self, qstream, output):
        super().__init__(qstream, output, name="DumbBob")  # Call parent constructor to initialize name

    def run(self):
        measurement_results = []
        for _ in self.qstream:
            qubits = [self.qrecv(dumb_alice) for _ in range(9)]
            measurement_results.append(qubits[0].measure())  # Only measures the first qubit
        self.output["DumbBob"] = measurement_results  # Save results to shared output


class ShorError(QError):
    '''Applies a random unitary error to a single qubit out of 9'''

    def __init__(self, qchannel):
        QError.__init__(self, qchannel)
        self.count = 0
        self.error_applied = False

    def apply(self, qubit):
        if self.count == 0:
            self.error_applied = False
        self.count = (self.count + 1) % 9
        if not self.error_applied and qubit is not None:
            if np.random.rand() < 0.5:  # Apply error with 50% probability
                random_unitary = unitary_group.rvs(2)  # Random U(2) matrix
                qubit.apply(random_unitary)
                self.error_applied = True
        return qubit


class ShorQChannel(QChannel):
    '''Represents a quantum channel with a Shor error applied'''

    def __init__(self, from_agent, to_agent):
        QChannel.__init__(self, from_agent, to_agent)
        self.errors = [ShorError(self)]  # Register the error model


def to_bits(string):
    '''Convert a string to a list of bits'''
    result = []
    for c in string:
        bits = bin(ord(c))[2:]
        bits = '00000000'[len(bits):] + bits
        result.extend([int(b) for b in bits])
    return result


def from_bits(bits):
    '''Convert a list of bits to a string'''
    chars = []
    for b in range(int(len(bits) / 8)):
        byte = bits[b * 8:(b + 1) * 8]
        chars.append(chr(int(''.join([str(bit) for bit in byte]), 2)))
    return ''.join(chars)


if __name__ == '__main__':


    # Prepare the message
    msg = "Peter Shor once lived in Ruddock 238! But who was Airman?"
    bits = to_bits(msg)

    # Encode the message
    qstream = QStream(9, len(bits))
    for bit, qsystem in zip(bits, qstream):
        if bit == 1:
            X(qsystem.qubit(0))

    # Alice and Bob with error correction
    out = Agent.shared_output()
    alice = Alice(qstream, out)
    bob = Bob(qstream, out)
    alice.qconnect(bob, ShorQChannel)

    # Dumb agents without error correction
    qstream2 = copy.deepcopy(qstream)
    dumb_alice = DumbAlice(qstream2, out)
    dumb_bob = DumbBob(qstream2, out)
    dumb_alice.qconnect(dumb_bob, ShorQChannel)

    # Run the simulation
    #Simulation(dumb_alice, dumb_bob, alice, bob).run()

    # Print results
    print("DumbAlice sent:   {}".format(msg))

    if "DumbBob" in out and out["DumbBob"] is not None:
        print("DumbBob received: {}".format(from_bits(out["DumbBob"])))
    else:
        print("DumbBob output is missing or invalid.")

    if "Bob" in out and out["Bob"] is not None:
        print("Alice sent:       {}".format(msg))
        print("Bob received:     {}".format(from_bits(out["Bob"])))
    else:
        print("Bob output is missing or invalid.")
