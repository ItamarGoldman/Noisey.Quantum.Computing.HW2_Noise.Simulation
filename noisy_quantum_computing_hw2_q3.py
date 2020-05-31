# -*- coding: utf-8 -*-


import qiskit
import numpy as np
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors.standard_errors import depolarizing_error, thermal_relaxation_error
from qiskit.ignis.characterization.coherence import t2star_circuits

import qiskit.ignis.verification.randomized_benchmarking as rb

from qiskit import QuantumRegister, QuantumCircuit
from qiskit import Aer
import qiskit.ignis.verification.tomography as tomo
from qiskit.quantum_info import state_fidelity


from qiskit import QuantumCircuit, execute
from qiskit import Aer, IBMQ

# Choose a real device to simulate from IBMQ provider

#IBMQ.save_account('')
provider = IBMQ.load_account()

#provider = IBMQ.enable_account('//PUT API KEY HERE//')


backend_sim = Aer.get_backend('qasm_simulator')

backend_real = provider.get_backend('ibmq_ourense')

basis_gates = ['u1','u2','u3','cx']


# Start Circuit 1

# Generate 3-qubit GHZ state
num_qubits = 3
q5 = QuantumRegister(3)
circuit = QuantumCircuit(q5)

# Start of Deutsch-Josza Algorithm - mod 2 =0 

circuit.h(0)
circuit.h(1)
circuit.x(2)
circuit.h(2)
circuit.barrier()
circuit.cx(0,2)
circuit.cx(1,2)
circuit.x(2)
circuit.barrier()
circuit.h(0)
circuit.h(1)
circuit.h(2)

# measurement
# circuit.measure(range(num_qubits),range(num_qubits))

# End of Deutsch-Josza Algorithm - mod 2 =0

# Start Noise Configuration
noise_model = NoiseModel()
p1Q = 0.002
p2Q = 0.01
noise_model.add_all_qubit_quantum_error(depolarizing_error(p1Q, 1), 'u2')
noise_model.add_all_qubit_quantum_error(depolarizing_error(2*p1Q, 1), 'u3')
noise_model.add_all_qubit_quantum_error(depolarizing_error(p2Q, 2), 'cx')
# End Noise configuration

# Start of Simulation with noise (thermal_relaxation_error)

# Start Noise Configuration

noise_model2 = NoiseModel()

#Add T1/T2 noise to the simulation
t1 = 100.
t2 = 80.
gate1Q = 0.1
gate2Q = 0.5

noise_model2.add_all_qubit_quantum_error(thermal_relaxation_error(t1,t2,gate1Q), 'u2')
noise_model2.add_all_qubit_quantum_error(thermal_relaxation_error(t1,t2,2*gate1Q), 'u3')
noise_model2.add_all_qubit_quantum_error(
    thermal_relaxation_error(t1,t2,gate2Q).tensor(thermal_relaxation_error(t1,t2,gate2Q)), 'cx')

# End Noise configuration

# Start state tomography - without noise

qst_deutsch = tomo.state_tomography_circuits(circuit, q5)
job = qiskit.execute(qst_deutsch, backend_sim, shots=5000)
results = job.result()
print("results:")
print(results.get_counts(0))
statefit = tomo.StateTomographyFitter(results, qst_deutsch)
p, M, weights = statefit._fitter_data(True, 0.5)
M_dg = np.conj(M).T
linear_inversion_matrix = np.linalg.inv(M_dg @ M) @ M_dg
rho_deutsch = linear_inversion_matrix @ p
rho_deutsch = np.reshape(rho_deutsch, (2**num_qubits, 2**num_qubits))

job = qiskit.execute(circuit, Aer.get_backend('statevector_simulator'))
psi_deutsch_clear = job.result().get_statevector(circuit)
print("Deutsch state vector (without noise): ")
print(psi_deutsch_clear)

F_deutsch = state_fidelity(psi_deutsch_clear, rho_deutsch, validate=False)
print('Fit Fidelity linear inversion =', F_deutsch)

rho_mle_deutsch = statefit.fit(method='lstsq')
F_deutsch = state_fidelity(psi_deutsch_clear, rho_mle_deutsch, validate=False)
print('Fit Fidelity MLE fit =', F_deutsch)

# End state tomography - without noise

# Start state tomography - with noise (t2)
print("With noise:")
qst_deutsch = tomo.state_tomography_circuits(circuit, q5)
job = qiskit.execute(qst_deutsch, backend_sim, noise_model=noise_model2, shots=5000)
results = job.result()
print("results:")
print(results.get_counts(0))
statefit = tomo.StateTomographyFitter(results, qst_deutsch)
p, M, weights = statefit._fitter_data(True, 0.5)
M_dg = np.conj(M).T
linear_inversion_matrix = np.linalg.inv(M_dg @ M) @ M_dg
rho_deutsch = linear_inversion_matrix @ p
rho_deutsch = np.reshape(rho_deutsch, (2**num_qubits, 2**num_qubits))

job = qiskit.execute(circuit, Aer.get_backend('statevector_simulator'))
psi_deutsch = job.result().get_statevector(circuit)
print("Deutsch state vector: ")
print(psi_deutsch)

F_deutsch = state_fidelity(psi_deutsch, rho_deutsch, validate=False)
print('Fit Fidelity linear inversion =', F_deutsch)

rho_mle_deutsch = statefit.fit(method='lstsq')
F_deutsch = state_fidelity(psi_deutsch, rho_mle_deutsch, validate=False)
print('Fit Fidelity MLE fit =', F_deutsch)

# End state tomography - with noise (t2)


# Start state tomography - real hardware (t2)
print("real hardware:")
qst_deutsch = tomo.state_tomography_circuits(circuit, q5)
job = qiskit.execute(qst_deutsch, backend_real, shots=5000)
results = job.result()
print("results:")
print(results.get_counts(0))
statefit = tomo.StateTomographyFitter(results, qst_deutsch)
p, M, weights = statefit._fitter_data(True, 0.5)
M_dg = np.conj(M).T
linear_inversion_matrix = np.linalg.inv(M_dg @ M) @ M_dg
rho_deutsch = linear_inversion_matrix @ p
rho_deutsch = np.reshape(rho_deutsch, (2**num_qubits, 2**num_qubits))

job = qiskit.execute(circuit, Aer.get_backend('statevector_simulator'))
psi_deutsch = job.result().get_statevector(circuit)
print("Deutsch state vector: ")
print(psi_deutsch)

F_deutsch = state_fidelity(psi_deutsch, rho_deutsch, validate=False)
print('Fit Fidelity linear inversion =', F_deutsch)

rho_mle_deutsch = statefit.fit(method='lstsq')
F_deutsch = state_fidelity(psi_deutsch, rho_mle_deutsch, validate=False)
print('Fit Fidelity MLE fit =', F_deutsch)

# End state tomography - real hardware (t2)

# End circuit 1




# Start Circuit 2

# Start of Simon Algorithem s='11'

s = '1'

# Creating registers
# qubits and classical bits for querying the oracle and finding the hidden period s
n = 2*len(str(s))
qn = QuantumRegister(n)
simonCircuit = QuantumCircuit(qn)
barriers = True

# Apply Hadamard gates before querying the oracle
simonCircuit.h(range(n))    

# Apply barrier 
if barriers:
    simonCircuit.barrier()

# Apply the query function
## 2-qubit oracle for s = 1
simonCircuit.cx(0, len(str(s)) + 0)
# simonCircuit.cx(0, len(str(s)) + 1)
# simonCircuit.cx(1, len(str(s)) + 0)
# simonCircuit.cx(1, len(str(s)) + 1)  

# Apply barrier 
if barriers:
    simonCircuit.barrier()

# Apply Hadamard gates to the input register
simonCircuit.h(range(len(str(s))))

# Measure ancilla qubits

# simonCircuit.measure(range(n),range(n))
#simonCircuit.measure_all()

# End of Simon Algorithem s='11'


# Start state tomography - without noise
print("Simon algorithm state tomography:")
qst_simon = tomo.state_tomography_circuits(simonCircuit, qn)
job = qiskit.execute(qst_simon, backend_sim, shots=5000)
results2 = job.result()
print("results:")
print(results2.get_counts(0))
statefit = tomo.StateTomographyFitter(results2, qst_simon)
p, M, weights = statefit._fitter_data(True, 0.5)
M_dg = np.conj(M).T
linear_inversion_matrix = np.linalg.inv(M_dg @ M) @ M_dg
rho_simon = linear_inversion_matrix @ p
rho_simon = np.reshape(rho_simon, (2**n, 2**n))

job = qiskit.execute(simonCircuit, Aer.get_backend('statevector_simulator'))
psi_simon = job.result().get_statevector(simonCircuit)
print("Simon state vector (without noise): ")
print(psi_simon)

F_simon = state_fidelity(psi_simon, rho_simon, validate=False)
print('Fit Fidelity linear inversion =', F_simon)

rho_mle_simon = statefit.fit(method='lstsq')
F_simon = state_fidelity(psi_simon, rho_mle_simon, validate=False)
print('Fit Fidelity MLE fit =', F_simon)

# End state tomography - without noise

# Start state tomography - with noise (t2)
print("With noise:")
qst_simon = tomo.state_tomography_circuits(simonCircuit, qn)
job = qiskit.execute(qst_simon, backend_sim, noise_model=noise_model2, shots=5000)
results2 = job.result()
print("results:")
print(results2.get_counts(0))
statefit = tomo.StateTomographyFitter(results2, qst_simon)
p, M, weights = statefit._fitter_data(True, 0.5)
M_dg = np.conj(M).T
linear_inversion_matrix = np.linalg.inv(M_dg @ M) @ M_dg
rho_simon = linear_inversion_matrix @ p
rho_simon = np.reshape(rho_simon, (2**n, 2**n))

job = qiskit.execute(simonCircuit, Aer.get_backend('statevector_simulator'))
psi_simon = job.result().get_statevector(simonCircuit)
print("Simon state vector (with noise): ")
print(psi_simon)

F_simon = state_fidelity(psi_simon, rho_simon, validate=False)
print('Fit Fidelity linear inversion =', F_simon)

rho_mle_simon = statefit.fit(method='lstsq')
F_simon = state_fidelity(psi_simon, rho_mle_simon, validate=False)
print('Fit Fidelity MLE fit =', F_simon)

# End state tomography - with noise (t2)


# Start state tomography - real hardware
print("real hardware:")
qst_simon = tomo.state_tomography_circuits(simonCircuit, qn)
job = qiskit.execute(qst_simon, backend_real, basis_gates=['u1','u2','u3','cx'], shots=4096)
results2 = job.result()
print("results:")
print(results2.get_counts(0))
statefit = tomo.StateTomographyFitter(results2, qst_simon)
p, M, weights = statefit._fitter_data(True, 0.5)
M_dg = np.conj(M).T
linear_inversion_matrix = np.linalg.inv(M_dg @ M) @ M_dg
rho_simon = linear_inversion_matrix @ p
rho_simon = np.reshape(rho_simon, (2**n, 2**n))

job = qiskit.execute(simonCircuit, Aer.get_backend('statevector_simulator'))
psi_simon = job.result().get_statevector(simonCircuit)
print("Simon state vector (real hardware): ")
print(psi_simon)

F_simon = state_fidelity(psi_simon, rho_simon, validate=False)
print('Fit Fidelity linear inversion =', F_simon)

rho_mle_simon = statefit.fit(method='lstsq')
F_simon = state_fidelity(psi_simon, rho_mle_simon, validate=False)
print('Fit Fidelity MLE fit =', F_simon)

# End state tomography - real hardware

# End Circuit 2

print(job.error_message())

