# -*- coding: utf-8 -*-

import qiskit
import numpy as np
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors.standard_errors import depolarizing_error, thermal_relaxation_error
from qiskit.ignis.characterization.coherence import t2star_circuits

import qiskit.ignis.verification.randomized_benchmarking as rb
import qiskit.ignis.verification.tomography as tomo


from qiskit import QuantumCircuit, execute
from qiskit import Aer, IBMQ

# Choose a real device to simulate from IBMQ provider

#IBMQ.save_account('')
provider = IBMQ.load_account()

#provider = IBMQ.enable_account('//PUT API KEY HERE//')


backend_sim = Aer.get_backend('qasm_simulator')

backend_real = provider.get_backend('ibmq_vigo')
coupling_map = backend_real.configuration().coupling_map
noise_model_extracted = NoiseModel.from_backend(backend_real)
print(noise_model_extracted)

basis_gates = ['u1','u2','u3','cx']


# Start Circuit 1

# Generate 3-qubit GHZ state
num_qubits = 5
circuit = QuantumCircuit(5, 5)

# Start of Deutsch-Josza Algorithm - mod 2 =0 

circuit.h(0)
circuit.h(1)
circuit.h(2)
circuit.h(3)
circuit.x(4)
circuit.h(4)
circuit.barrier()
circuit.cx(0,4)
circuit.cx(1,4)
circuit.x(2)
circuit.cx(2,4)
circuit.cx(3,4)
circuit.barrier()
circuit.h(0)
circuit.h(1)
circuit.h(2)
circuit.h(3)
circuit.h(4)

# measurement
circuit.measure(range(num_qubits),range(num_qubits))

# End of Deutsch-Josza Algorithm - mod 2 =0

# Start of Simulation without noise
job = execute(circuit, backend_sim,
              basis_gates=basis_gates, shots=4096)
result1 = job.result()

print("Deutsch-Josza Algorithm - simulation without noise")
print(result1.get_counts(0))

# End of simoulation without noise


# Start of Simulation with noise (depolarizing_error)

# Start Noise Configuration
noise_model = NoiseModel()
p1Q = 0.002
p2Q = 0.01
noise_model.add_all_qubit_quantum_error(depolarizing_error(p1Q, 1), 'u2')
noise_model.add_all_qubit_quantum_error(depolarizing_error(2*p1Q, 1), 'u3')
noise_model.add_all_qubit_quantum_error(depolarizing_error(p2Q, 2), 'cx')
# End Noise configuration


job = execute(circuit, backend_sim,
              noise_model=noise_model,
              basis_gates=['u1','u2','u3','cx'], shots=4096)
result2 = job.result()

print("Deutsch-Josza Algorithm - simulation with noise (depolarizing_error)")
print(result2.get_counts(0))

# End of simoulation with noise (depolarizing_error)


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


job = execute(circuit, backend_sim,
              noise_model=noise_model2,
              basis_gates=['u1','u2','u3','cx'], shots=4096)
result2 = job.result()

print("Deutsch-Josza Algorithm - simulation with noise (thermal_relaxation_error)")
print(result2.get_counts(0))

# End of Simulation with noise (thermal_relaxation_error)

# Start of simulation with noise extracted from real hardware

job = execute(circuit, backend_sim,
              noise_model=noise_model_extracted,
              basis_gates=['u1','u2','u3','cx'], shots=4096)
result_extracted1 = job.result()

print("Deutsch-Josza Algorithm - simulation with noise (extracted from real hardware)")
print(result_extracted1.get_counts(0))

# End of simulation with noise extracted from real hardware

# # Start of Run on Real Hardware


job = execute(circuit, backend_real,
              basis_gates=['u1','u2','u3','cx'], shots=4096)
result_real1 = job.result()
print("Deutsch-Josza Algorithm - Real Hardware")
print(result_real1.get_counts(0))

# # End of Run on Real Hardware



# End circuit 1




# Start Circuit 2

# Start of Simon Algorithem s='11'

s = '11'

# Creating registers
# qubits and classical bits for querying the oracle and finding the hidden period s
n = 2*len(str(s))
simonCircuit = QuantumCircuit(n,n)
barriers = True

# Apply Hadamard gates before querying the oracle
simonCircuit.h(range(n))    

# Apply barrier 
if barriers:
    simonCircuit.barrier()

# Apply the query function
## 2-qubit oracle for s = 11
simonCircuit.cx(0, len(str(s)) + 0)
simonCircuit.cx(0, len(str(s)) + 1)
simonCircuit.cx(1, len(str(s)) + 0)
simonCircuit.cx(1, len(str(s)) + 1)  

# Apply barrier 
if barriers:
    simonCircuit.barrier()

# Apply Hadamard gates to the input register
simonCircuit.h(range(len(str(s))))

# Measure ancilla qubits

simonCircuit.measure(range(n),range(n))
#simonCircuit.measure_all()

# End of Simon Algorithem s='11'


# Start of Simulation without noise

job = execute(simonCircuit, backend_sim,
              basis_gates=basis_gates, shots=4096)
result3 = job.result()
print("Simon Algorithem - simulation without noise")
print(result3.get_counts(0))

# End of simoulation without noise


# Start of Simulation with noise (depolarizing_error)

job = execute(simonCircuit, backend_sim,
              noise_model=noise_model,
              basis_gates=['u1','u2','u3','cx'], shots=4096)
result4 = job.result()
print("Simon Algorithem - simulation with noise (depolarizing_error)")
print(result4.get_counts(0))

# End of simoulation with noise (depolarizing_error)


# Start of Simulation with noise (thermal_relaxation_error)

job = execute(simonCircuit, backend_sim,
              noise_model=noise_model2,
              basis_gates=['u1','u2','u3','cx'], shots=4096)
result4 = job.result()

print("Simon Algorithem - simulation with noise (thermal_relaxation_error)")
print(result4.get_counts(0))

# End of simoulation with noise (thermal_relaxation_error)

# Start of simulation with noise extracted from real hardware

job = execute(circuit, backend_sim,
              noise_model=noise_model_extracted,
              basis_gates=['u1','u2','u3','cx'], shots=4096)
result_extracted2 = job.result()

print("Simon Algorithm - simulation with noise (extracted from real hardware)")
print(result_extracted2.get_counts(0))

# End of simulation with noise extracted from real hardware


# # Start of Run on Real Hardware


job = execute(simonCircuit, backend_real,
              basis_gates=['u1','u2','u3','cx'], shots=4096)
result_real2 = job.result()
print("Simon Algorithem - Real Hardware")
print(result_real2.get_counts(0))

# # End of Run on Real Hardware


# End Circuit 2


# Question 2
print("Question 2:")


# T2* Charactirization circuit
SEED = 0
num_of_gates = np.append(
    (np.linspace(10, 150, 10)).astype(int),
    (np.linspace(160, 450, 5)).astype(int))
gate_time = 0.1
qubits = [0,1]
circs_osc, xdata, omega = t2star_circuits(num_of_gates, gate_time,
                                              qubits, 5)
shots = 2048

# Start of T2* characteristic whithout noise - simulation

# T2_cha_result = qiskit.execute(
#     circs_osc, backend,
#     shots=shots,
#     seed_simulator=SEED,
#     optimization_level=0).result()


# print("T2* characteristic - simulation without noise - NOT GOOD NEED TO CHANGE TO GRAPH")
# print(T2_cha_result.get_counts(0))

# End of T2* characteristic whithout noise  - simulation


# Start of T2* characteristic whith thermal noise simulation

# T2_cha_result_thermal = qiskit.execute(
#     circs_osc, backend,
#     shots=shots,
#     seed_simulator=SEED,
#     noise_model=noise_model2,
#     optimization_level=0).result()

# print("T2* characteristic - simulation with noise - NOT GOOD NEED TO CHANGE TO GRAPH")
# print(T2_cha_result_thermal.get_counts(0))

# End of T2* characteristic whith thermal noise simulation


# # Start of Run on Real Hardware


# job_t2star_real = execute(circs_osc, backend_real,
#               basis_gates=['u1','u2','u3','cx'], shots=shots,
#               optimization_level=0))
# result_real3 = job_t2star_real.result()
#print("T2* characterization - Real Hardware - NOT GOOD NEED TO CHANGE TO GRAPH")
# print(result_real3.get_counts(0))

# # End of Run on Real Hardware



# End of T2* characteristic whith thermal noise




# noise_model = NoiseModel()
# p1Q = 0.002
# p2Q = 0.01
# noise_model.add_all_qubit_quantum_error(depolarizing_error(p1Q, 1), 'u2')
# noise_model.add_all_qubit_quantum_error(depolarizing_error(2*p1Q, 1), 'u3')
# noise_model.add_all_qubit_quantum_error(depolarizing_error(p2Q, 2), 'cx')

# noise_model2 = NoiseModel()

# #Add T1/T2 noise to the simulation
# t1 = 100.
# t2 = 80.
# gate1Q = 0.1
# gate2Q = 0.5
# noise_model2.add_all_qubit_quantum_error(thermal_relaxation_error(t1,t2,gate1Q), 'u2')
# noise_model2.add_all_qubit_quantum_error(thermal_relaxation_error(t1,t2,2*gate1Q), 'u3')
# noise_model2.add_all_qubit_quantum_error(
#     thermal_relaxation_error(t1,t2,gate2Q).tensor(thermal_relaxation_error(t1,t2,gate2Q)), 'cx')

# rb_opts = {}
# rb_opts['length_vector'] = nCliffs
# rb_opts['nseeds'] = nseeds
# rb_opts['rb_pattern'] = rb_pattern
# rb_opts['length_multiplier'] = length_multiplier
# rb_circs, xdata = rb.randomized_benchmarking_seq(**rb_opts)

