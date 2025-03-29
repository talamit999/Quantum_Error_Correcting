# Imports
import numpy as np
import os
import pandas as pd
import time
from classiq import authenticate
authenticate(overwrite=True)
from classiq import *
import matplotlib.pyplot as plt
import scipy
from classiq import Output, create_model, power, prepare_amplitudes, synthesize, unitary
from classiq import write_qmod



# Tunable parameters
n = 1                                                  # System Dimension: [2^n, 2^n]
prob_indicator_bit_flip = np.arange(0.00, 0.12, 0.02)  # Range of prob values
n_total_shots = 1000000                                # Number of measurements
correction_pairs = [(0, 0), (1, 0), (0, 2**n - 1)]     # Sets for (correct_indicator, correct_res) values to check



# Consts (internal use - do not change)
correct_indicator = 0
correct_res = 0
QPE_SIZE = 2 * n
precision = 2 * n
n_shots = 1
error_num = 0
index_qubits_map = []



@qfunc
def load_b(
        amplitudes: CArray[CReal], state: Output[QArray[QBit]], bound: CReal
) -> None:
    prepare_amplitudes(amplitudes, bound, state)


@qfunc
def simple_eig_inv(phase: QNum, indicator: Output[QBit]):
    allocate(1, indicator)
    C = 1 / 2 ** phase.size
    indicator *= C / phase


@qfunc
def hhl(
        rhs_vector: CArray[CReal],
        bound: CReal,
        precision: CInt,
        hamiltonian_evolution_with_power: QCallable[CInt, QArray[QBit]],
        state: Output[QArray[QBit]],
        phase: Output[QNum],
        indicator: Output[QBit],
):

    # Allocate a quantum number for the phase with given precision
    allocate_num(precision, False, precision, phase)

    # Prepare initial state
    load_b(amplitudes=b.tolist(), state=state, bound=bound)

    # Perform quantum phase estimation and eigenvalue inversion within a quantum operation
    within_apply(
        lambda: qpe_flexible(
            unitary_with_power=lambda k: hamiltonian_evolution_with_power(k, state),
            phase=phase,
        ),
        lambda: simple_eig_inv(phase=phase, indicator=indicator),
    )


from classiq.execution import (
    ClassiqBackendPreferences,
    ClassiqSimulatorBackendNames,
    ExecutionPreferences,
)


# Set backend preferences with noise properties
backend_preferences = ClassiqBackendPreferences(
    backend_name=ClassiqSimulatorBackendNames.SIMULATOR_STATEVECTOR,
)

@qfunc
def unitary_with_power_logic(
        pw: CInt, matrix: CArray[CArray[CReal]], target: QArray[QBit]
) -> None:
    power(pw, lambda: unitary(elements=matrix, target=target))




@qfunc
def main(res: Output[QNum], phase: Output[QNum], indicator: Output[QBit],) -> None:
    # apply hhl
    hhl(rhs_vector=b.tolist(), bound=0, precision=QPE_SIZE,
        hamiltonian_evolution_with_power=lambda arg0, arg1: unitary_with_power_logic(
        matrix=scipy.linalg.expm(2 * np.pi * 1j * A).tolist(), pw=arg0, target=arg1),
        state=res, phase=phase, indicator=indicator,)

    apply_error(res=res, phase=phase, indicator=indicator,)

@qfunc
def apply_error(res: QArray[QBit], phase: QArray[QBit], indicator: QBit,):  # Insert error with/out correction
    global n
    global error_num
    global correct_res
    global correct_indicator

    aux_ind0 = QBit('aux_ind0')
    aux_ind1 = QBit('aux_ind1')
    aux_res0 = QBit('aux_res0')
    aux_res1 = QBit('aux_res1')
    aux_res2 = QBit('aux_res2')
    aux_res3 = QBit('aux_res3')
    aux_res4 = QBit('aux_res4')
    aux_res5 = QBit('aux_res5')
    allocate(1, aux_ind0)
    allocate(1, aux_ind1)
    allocate(1, aux_res0)
    allocate(1, aux_res1)
    allocate(1, aux_res2)
    allocate(1, aux_res3)
    allocate(1, aux_res4)
    allocate(1, aux_res5)

    # pre-correct indicator
    if correct_indicator:
        CX(indicator, aux_ind0)
        CX(indicator, aux_ind1)

    # pre-correct res
    if n >= 1 and correct_res & 1:
        #print("pre correct res0")
        CX(res[0], aux_res0)
        CX(res[0], aux_res1)
    if n >= 2 and correct_res & 2:
        #print("pre correct res1")
        CX(res[1], aux_res2)
        CX(res[1], aux_res3)
    if n >= 3 and correct_res & 4:
        CX(res[2], aux_res4)
        CX(res[2], aux_res5)

    k =  1 + 3 * n  # אורך המחרוזת הבינארית
    error_pattern = format(error_num, f'0{k}b')  # המרה למספר בינארי באורך k
    print("error_pattern: ", str(error_pattern))

    for i, bit in enumerate(error_pattern):
        if bit == '1':
            if i == 0:
                X(indicator)
            elif 1 <= i <= 2 * n:
                X(phase[i-1])
            elif 1 + 2 * n <= i < k:
                index = i - (1+2*n)
                X(res[index])

    # post-correct indicator
    ctrl_ind = QArray('ctrl_ind')
    if correct_indicator:
        CX(indicator, aux_ind0)
        CX(indicator, aux_ind1)
        bind([aux_ind0, aux_ind1], ctrl_ind)
        CCX(ctrl_ind, indicator)

    # post-correct res
    ctrl_res0 = QArray('ctrl_res0')
    ctrl_res1 = QArray('ctrl_res1')
    ctrl_res2 = QArray('ctrl_res2')
    if n >= 1 and correct_res & 1:
        CX(res[0], aux_res0)
        CX(res[0], aux_res1)
        bind([aux_res0, aux_res1], ctrl_res0)
        CCX(ctrl_res0, res[0])
    if n >= 2 and correct_res & 2:
        CX(res[1], aux_res2)
        CX(res[1], aux_res3)
        bind([aux_res2, aux_res3], ctrl_res1)
        CCX(ctrl_res1, res[1])
    if n >= 3 and correct_res & 4:
        CX(res[2], aux_res4)
        CX(res[2], aux_res5)
        bind([aux_res4, aux_res5], ctrl_res2)
        CCX(ctrl_res2, res[2])


def hhl_model(main, backend_preferences, n_shots):
    """
    Constructs an HHL quantum model with specified noise and execution preferences.

    Args:
        main: The main quantum circuit or model input.
        backend_preferences: Backend execution settings, including noise preferences.
        n_shots (int): Number of measurement shots for execution.

    Returns:
        A quantum model configured with the given execution preferences.
    """
    qmod_hhl = create_model(
        main,
        execution_preferences=ExecutionPreferences(
            num_shots=n_shots, backend_preferences=backend_preferences,
        ),
    )
    return qmod_hhl






def read_positions(circuit_hhl, res_hhl):
    """
    Extracts the positions of key qubits in the HHL circuit.

    Args:
        circuit_hhl: The HHL quantum circuit.
        res_hhl: The result object containing the physical qubit mapping.

    Returns:
        tuple:
            - target_pos (int): Position of the control qubit.
            - sol_pos (list): Positions of the solution qubits.
            - phase_pos (list): Positions of the phase register qubits, reordered for endianness.
    """
    # Position of the control qubit
    target_pos = res_hhl.physical_qubits_map["indicator"][0]

    # Positions of the solution qubits
    sol_pos = list(res_hhl.physical_qubits_map["res"])

    # Find the position of the "phase" register and adjust for endianness
    total_q = circuit_hhl.data.width  # Total number of qubits in the circuit
    phase_pos = [
        total_q - k - 1 for k in range(total_q) if k not in sol_pos + [target_pos]
    ]

    return target_pos, sol_pos, phase_pos





def prepare_system(A, b, condition_number_max_val=100000):
    """
    Prepares the system of equations Ax = b for solving using the HHL algorithm.

    This function checks matrix A properties, normalizes vector b, and ensures that the
    condition number is acceptable and eigenvalues are in a valid range.

    Args:
        A (numpy.ndarray): The system matrix A.
        b (numpy.ndarray): The right-hand side vector b.
        condition_number_max_val (int, optional): Maximum allowable condition number for A. Default is 100000.

    Raises:
        Exception: If A is not symmetric, if its determinant is zero, if the condition number is too high,
                    or if eigenvalues are out of the (0, 1) range.

    Returns:
        tuple:
            - A (numpy.ndarray): Possibly modified matrix A.
            - b (numpy.ndarray): Normalized vector b.
            - hamiltonian (numpy.ndarray): Hamiltonian representation of A.
    """
    # Normalize vector b
    # b = b / np.linalg.norm(b)

    # Check if A is symmetric
    if not np.allclose(A, A.T, rtol=1e-6, atol=1e-6):
        raise Exception("The matrix is not symmetric")

    # Check if det(A) is zero
    if np.linalg.det(A) == 0:
        raise Exception("The determinant of A is zero, indicating a singular matrix")

    # Compute eigenvalues
    w, v = np.linalg.eig(A)

    # Find the maximum and minimum nonzero eigenvalues
    lambda_max = np.max(w)
    lambda_min = np.min(w[w > 0])  # Avoid division by zero
    condition_num = lambda_max / lambda_min

    if condition_num > condition_number_max_val:
        raise Exception("Condition number is too big")

    # The solution of Ax=b is the same as the solution of (λ_max + 1) Ax = (λ_max + 1) b
    if lambda_max > 1:
        A = A / (lambda_max + 1)
        b = b / (lambda_max + 1)
        w, v = np.linalg.eig(A)

    # Normalize vector b
    b = b / np.linalg.norm(b)

    for lam in w:
        if lam < 0 or lam > 1:
            raise Exception("Eigenvalues are not in (0,1)")

    # Binary representation of eigenvalues (classically calculated)
    m = 32  # Precision of a binary representation, e.g., 32 binary digits
    sign = lambda num: "-" if num < 0 else ""  # Calculate sign of a number
    binary = lambda fraction: str(
        np.binary_repr(int(np.abs(fraction) * 2 ** m)).zfill(m)
    ).rstrip("0")  # Binary representation of a fraction

    print("\nEigenvalues:")
    for eig in sorted(w):
        print(f"{sign(eig)}0.{binary(eig.real)} =~ {eig.real}")

    # Convert matrix A to Hamiltonian representation
    hamiltonian = matrix_to_hamiltonian(A)

    return A, b, hamiltonian





def compute_normalized_state(output_map, hist_filename):
    """
    Computes the normalized state from the histogram data.

    Args:
        output_map (dict): A dictionary mapping "res", "phase", and "indicator" to qubit positions.
        hist_filename (str): Path to the CSV file containing the measurement histogram.

    Returns:
        numpy.ndarray: The normalized state vector, or None if the file is not found or invalid.
    """
    # Calculate the expected length of the measurement string
    expected_len = max(max(output_map["res"]), max(output_map["phase"]), max(output_map["indicator"])) + 1

    # Load the CSV file if it exists, otherwise return None
    if not os.path.exists(hist_filename):
        print("File not found:", hist_filename)
        return None

    df = pd.read_csv(hist_filename, dtype={"measurement": str, "counts": int})

    # Create a histogram dictionary where each measurement string is padded to the expected length
    hist = {}
    for _, row in df.iterrows():
        meas = str(row["measurement"])
        meas_padded = meas.zfill(expected_len)
        count = int(row["counts"])
        if meas_padded in hist:
            hist[meas_padded] += count
        else:
            hist[meas_padded] = count

    # Retrieve indices for res, phase, and indicator
    len_s = expected_len
    res_index = [len_s - 1 - i for i in output_map["res"]]
    phase_index = [len_s - 1 - i for i in output_map["phase"]]
    indicator_index = [len_s - 1 - i for i in output_map["indicator"]]

    # Filter valid measurements: check if the indicator bit is "1" and all phase bits are "0"
    valid_counts = {
        state: count for state, count in hist.items()
        if state[indicator_index[0]] == "1" and all(state[pi] == "0" for pi in phase_index)
    }

    # Sum the valid counts
    n_success = sum(valid_counts.values())

    # Initialize a dictionary to store counts for each possible value of res
    n_res = {i: 0 for i in range(2 ** len(output_map["res"]))}
    for state, count in valid_counts.items():
        # Convert the res bits to an integer value
        start = min(res_index)
        end = max(res_index) + 1
        res_value = int(state[start:end], 2)
        n_res[res_value] += count

    # Calculate probabilities and corresponding alpha values
    p_values = {k: v / n_success for k, v in n_res.items()}
    alpha_values = {k: np.sqrt(p) for k, p in p_values.items()}

    # Construct the normalized state vector
    x_estimated = np.array([alpha_values.get(i, 0) for i in range(2 ** len(output_map["res"]))])

    return x_estimated



def calc_statistics(A, b, x_estimated):
    """
    Calculates the fidelity and average error between the classical and estimated quantum solutions.

    Args:
        A (numpy.ndarray): The system matrix.
        b (numpy.ndarray): The right-hand side vector.
        x_estimated (numpy.ndarray): The estimated quantum solution.

    Returns:
        tuple: (fidelity, avg_err), where:
            - fidelity (float): Fidelity between classical and quantum solutions in percentage.
            - avg_err (float): Average relative error between solutions in percentage.
    """
    # Solve for the classical solution and normalize it
    sol_classical = np.linalg.solve(A, b).flatten()
    sol_classical_normalized = sol_classical / np.linalg.norm(sol_classical)
    print("Classical Solution normalized:         ", sol_classical_normalized)

    # Normalize and adjust the sign of the estimated quantum solution
    x_estimated *= np.sign(sol_classical_normalized)
    print("x_estimated normalized:                ", x_estimated)

    # Fidelity: calculate the overlap between the normalized classical and quantum solutions
    quantum_x_normalized = x_estimated / np.linalg.norm(x_estimated)
    fidelity = np.abs(np.dot(sol_classical_normalized, quantum_x_normalized)) ** 2
    fidelity = np.round(fidelity * 100, 3)
    print(f"Fidelity:                    {np.round(fidelity, 2)} %")

    # Average error: calculate the relative error between the two solutions
    relative_error = np.linalg.norm(sol_classical_normalized - quantum_x_normalized) / np.linalg.norm(
        sol_classical_normalized)
    avg_err = np.round(relative_error * 100, 3)
    print(f"Average Error:               {np.round(avg_err, 3)} %")

    return fidelity, avg_err



def solve_HHL(A, b, n_shots):
    """
    Solves the system of linear equations `Ax = b` using the HHL algorithm.

    This function initializes the HHL model, executes the quantum algorithm for solving `Ax = b`,
    and returns the result of the computation using classiq platform.

    Args:
        A (numpy.ndarray): The system matrix.
        b (numpy.ndarray): The right-hand side vector.
        n_shots (int): The number of shots to be used in the quantum computation.

    Returns:
        res_hhl_exact: The result of the HHL algorithm execution.
    """
    # Initialize the HHL model with the given backend preferences and number of shots
    qmod_hhl_exact = hhl_model(main, backend_preferences, n_shots)

    # Save the quantum model file
    write_qmod(qmod_hhl_exact, "hhl_exact", decimal_precision=20)

    # Synthesize the quantum program and generate the corresponding quantum circuit
    qprog_hhl_exact = synthesize(qmod_hhl_exact)
    circuit_hhl_exact = QuantumProgram.from_qprog(qprog_hhl_exact)

    # Execute the quantum program and get the result
    res_hhl_exact = execute(qprog_hhl_exact).result_value()

    return res_hhl_exact




def update_histogram(hist, hist_filename):
    """
    Updates the histogram by merging new data and saving it to a CSV file.

    Args:
        hist (dict): Dictionary of histogram data with measurements as keys and counts as values.
        hist_filename (str): Path to the CSV file for saving the updated histogram.

    Returns:
        None
    """
    os.makedirs("hist_results", exist_ok=True)

    if os.path.exists(hist_filename):
        df = pd.read_csv(hist_filename, dtype={'measurement': str})
    else:
        df = pd.DataFrame(columns=["measurement", "counts"])

    new_data = pd.DataFrame(hist.items(), columns=["measurement", "counts"])
    df = pd.concat([df, new_data]).groupby("measurement", as_index=False).sum()
    df.to_csv(hist_filename, index=False)





def count_bit_flips(n_shots, k, p):
    """
    Computes the number of measurements corresponding to each possible bit flip pattern.

    Parameters:
        n_shots (int): Number of measurements
        k (int): Length of the bit string
        p (float): Probability of a single bit flipping

    Returns:
        np.ndarray: An array of size 2^k containing the count for each error pattern
    """
    error_counts = np.zeros(2 ** k)

    for i in range(2 ** k):
        num_flipped_bits = bin(i).count('1')  # Count the number of flipped bits
        probability = (1 - p) ** (k - num_flipped_bits) * (p ** num_flipped_bits)
        error_counts[i] = n_shots * probability

    return error_counts





def simulate_quantum_noise(prob_indicator_bit_flip, hist_filename):
    """
    Simulates quantum noise by applying bit flips to the system and computes the fidelity and average error.

    Args:
        prob_indicator_bit_flip (float): Probability of a single bit flipping.
        hist_filename (str): File path to store histogram results.

    Returns:
        tuple: A tuple containing the fidelity, average error, and runtime of the simulation.
    """
    global error_num
    global index_qubits_map
    global n

    start_time = time.time()
    k = 1 + 3 * n  # Length of the bit string

    # Get measurement distribution based on bit flip probabilities
    error_distribution = count_bit_flips(n_total_shots, k, prob_indicator_bit_flip)

    # Simulate error patterns and update histogram
    for error_pattern in range(2 ** k):
        noisy_shots = round(error_distribution[error_pattern])
        if noisy_shots > 0:
            error_num = error_pattern  # Store error pattern in decimal
            n_shots = noisy_shots
            res_hhl_with_noise = solve_HHL(A, b, n_shots)
            update_histogram(res_hhl_with_noise.counts, hist_filename)
            index_qubits_map = res_hhl_with_noise.output_qubits_map

    # Analyze results
    run_time = time.time() - start_time
    run_time = np.round(run_time, 3)
    x_estimated = compute_normalized_state(index_qubits_map, hist_filename)
    fidelity, avg_err = calc_statistics(A, b, x_estimated)

    return fidelity, avg_err, run_time






def compare_correction_effect(i):
    """
    Compares the correction effect by simulating quantum noise and evaluating the performance.

    This function runs simulations for different probabilities of bit flips and correction pairs,
    calculates fidelity and average error, and saves the results to a CSV file.

    Args:
        i (int): An identifier for the current simulation run.

    Returns:
        None
    """
    global n
    global correct_indicator
    global correct_res
    global correction_pairs

    os.makedirs("compare_results", exist_ok=True)

    for p in prob_indicator_bit_flip:
        for correct_indicator, correct_res in correction_pairs:
            print("--------------------------------------------------------------------------------------------------")
            print(f"Correct ind: {correct_indicator}, Correct res: {correct_res}")
            hist_filename = f"hist_results/histogram_n={n}_p={p}_nshots={n_total_shots}_correct_ind={correct_indicator}_correct_res={correct_res}_i={i}.csv"

            fidelity, avg_err, run_time = simulate_quantum_noise(p, hist_filename)
            print(f"Prob: {p:.3f} | Avg Error: {avg_err:.2f}% | Fidelity: {fidelity:.2f}% | Run Time: {run_time:.2f} sec")

            # Save results to CSV file
            results_file = os.path.join(os.getcwd(), f"compare_results/n={n}_nshots={n_total_shots}_correct_ind={correct_indicator}_correct_res={correct_res}_i={i}.csv")
            df = pd.DataFrame({"p": [p], "avg_err": [avg_err], "fidelity": [fidelity], "run_time": [run_time]})
            df.to_csv(results_file, mode='a', header=not os.path.exists(results_file), index=False)




def plot_correction_effect(file1, label1, file2, label2, file3=None, label3=None, file4=None, label4=None, file5=None,
                           label5=None, plot_type="avg_err"):
    """
    Plots the effect of error correction based on the data from multiple CSV files.

    This function loads data from up to five CSV files and plots the specified `plot_type` (either "avg_err" or "fidelity")
    against the bit flip probability (p) for each file.

    Args:
        file1 (str): Path to the first CSV file.
        label1 (str): Label for the first data series.
        ...
    Returns:
        None
    """

    # Load data files
    data_files = [(file1, label1), (file2, label2), (file3, label3), (file4, label4), (file5, label5)]
    dataframes = []

    for file, label in data_files:
        if file is not None and os.path.exists(file):
            df = pd.read_csv(file)
            dataframes.append((df, label))
        elif file is not None:
            print(f"File not found: {file}")

    # Plot
    plt.figure(figsize=(8, 6))

    # Plot each dataset
    markers = ["o", "s", "^", "v", "p"]
    colors = ["b", "g", "r", "orange", "m"]

    for i, (df, label) in enumerate(dataframes):
        plt.plot(df["p"], df[plot_type], label=label, marker=markers[i], color=colors[i])

    # Set labels and title based on the plot type
    plt.xlabel("Bit Flip Probability")
    if plot_type == "avg_err":
        plt.ylabel("Average Error [%]")
        plt.title("Effect of Error Correction on Average Error (nshots=10e6)")
    elif plot_type == "fidelity":
        plt.ylabel("Fidelity [%]")
        plt.title("Effect of Error Correction on Fidelity (nshots=10e6)")

    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()




def generate_random_vector(size):
    """
    Generates a random vector of the specified size.

    Args:
        size (int): Size of the vector.

    Returns:
        numpy.ndarray: A random vector of shape (size, 1).
    """
    return np.random.rand(size, 1)




def generate_positive_definite_matrix(size):
    """
    Generates a random positive definite matrix of the specified size.

    Args:
        size (int): Size of the matrix.

    Returns:
        numpy.ndarray: A positive definite matrix of shape (size, size).
    """
    M = np.random.randn(size, size)
    A = np.dot(M.T, M)
    A += np.eye(size) * 0.1
    return A





if __name__ == "__main__":
    '''
    Main loop for running quantum system tests. It performs N_samples successful iterations 
    where a positive definite matrix A and vector b are generated, the system is prepared, 
    and correction effects are compared.
    '''
    N_samples = 100

    #counters
    successful_runs = 0
    total_attempts = 0

    #collect results until done
    while successful_runs < N_samples:
        try:
            A = generate_positive_definite_matrix(2**n)
            b = generate_random_vector(2**n)

            A, b, hamiltonian = prepare_system(A, b, condition_number_max_val=5)
            amplitudes = b.tolist()

            compare_correction_effect(successful_runs)
            successful_runs += 1  # Increment only on successful run

        except Exception as e:
            print(f"Attempt {total_attempts + 1} failed: {e}")

        total_attempts += 1


    '''
    file1 = "compare_results/n=2_nshots=1000000_correct_ind=0_correct_res=0.csv"
    file2 = "compare_results/n=2_nshots=1000000_correct_ind=1_correct_res=0.csv"
    file3 = "compare_results/n=2_nshots=1000000_correct_ind=0_correct_res=3.csv"
    file4 = "compare_results/n=2_nshots=1000000_correct_ind=0_correct_res=1.csv"
    file5 = "compare_results/n=2_nshots=1000000_correct_ind=1_correct_res=3.csv"

    label1 = "No Correction"
    label2 = "Correct Indicator"
    label3 = "Correct Res (fully)"
    label4 = "Correct Res (partially)"
    label5 = "Correct Indicator and Res"
    plot_correction_effect(file1, label1, file2, label2, file3, label3, file4, label4, file5, label5)
    '''

