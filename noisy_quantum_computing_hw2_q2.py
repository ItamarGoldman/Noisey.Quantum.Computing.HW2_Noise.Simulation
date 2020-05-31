# -*- coding: utf-8 -*-

import qiskit
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors.standard_errors import depolarizing_error, thermal_relaxation_error

import qiskit.ignis.verification.randomized_benchmarking as rb
import qiskit.ignis.verification.tomography as tomo

from qiskit import QuantumCircuit, execute
from qiskit import Aer, IBMQ

# Choose a real device to simulate from IBMQ provider
# IBMQ.save_account('//PUT API KEY HERE//')
provider = IBMQ.load_account()
backend = provider.get_backend('ibmq_vigo')
coupling_map = backend.configuration().coupling_map

# Generate an Aer noise model for device
noise_model = NoiseModel.from_backend(backend)
basis_gates = noise_model.basis_gates

# Generate 3-qubit GHZ state
num_qubits = 3
circ = QuantumCircuit(3, 3)
circ.h(0)
circ.cx(0, 1)
circ.cx(1, 2)
circ.measure([0, 1, 2], [0, 1 ,2])

# Perform noisy simulation
backend = Aer.get_backend('qasm_simulator')
job = execute(circ, backend,
              coupling_map=coupling_map,
              noise_model=noise_model,
              basis_gates=basis_gates)
result = job.result()

print(result.get_counts(0))



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

