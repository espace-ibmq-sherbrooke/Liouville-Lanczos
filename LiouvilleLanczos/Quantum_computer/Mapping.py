# The following code is based on mapomatic usage exemple, with minimal modifications,
# as such we make no claim of ownership. This file is licensed under the Apache 
# License, Version 2.0. You may obtain a copy of this license 
# at http://www.apache.org/licenses/LICENSE-2.0.

from mapomatic import deflate_circuit, evaluate_layouts, matching_layouts
from qiskit import transpile,QuantumCircuit,ClassicalRegister
import numpy as np


"""

Fiddling to allow sub-circuit optimization. Mappomatic reorder the qubits in a way that 
doesn't tell us the correspondance with the orginial qubits. Tracks qubit permutation
with by adding measurements to the end of the supplied circuit. This create a labeling 
correspondance between the classical and quantum bits, and the classical bits labeling is 
unchanged by the optimization process.

Use find_best_layout to obtain a backend optimized layout for your circuit.

"""

def find_best_layout(circuit:QuantumCircuit,backend,num_tries,level=3,seed=132423,initial_layout = None):
    """
    The optimization process is stochastic, num_tries is the number of trials to perform.
    Supply the circuit to optimize, a backend and a number of trials.
    """
    nqbit = circuit.num_qubits
    circuit = circuit.copy()
    permutation_tracking(circuit)
    circuits_ts = transpile(
        [circuit] * num_tries, backend, optimization_level=level, seed_transpiler=seed,initial_layout=initial_layout
    )
    egate = detect_entangling_gate(backend)
    cx_counts = [circuits_ts[idx].count_ops()[egate] for idx in range(num_tries)]
    best_idx = np.argmin(cx_counts)
    best_circuit = circuits_ts[best_idx]
    permutation = mapping_to_permutation(extract_physical_mapping(best_circuit,nqbit))
    best_circuit.data = best_circuit.data[:-(nqbit+1)] #cleanup the permutation tracking elements
    deflated_circuit = deflate_circuit(best_circuit)
    layouts = matching_layouts(deflated_circuit, backend)
    scored_layouts = evaluate_layouts(
        deflated_circuit, layouts, backend
    )  # cost_function = cost_func
    return apply_permutation(scored_layouts,permutation,deflated_circuit)

def detect_entangling_gate(backend):
    pot_entaglers=["ecr","cz"] #Add new hardware entangling gate as hardware evolve
    egate='cx' #default value, most backends uses this for now.
    for entangler in pot_entaglers:
        if entangler in backend.configuration.basis_gates:
            egate = entangler
    return egate


def apply_permutation(scored_layouts,permutation,deflated_circuit):
    #apply the permutation
    scored_layout = np.zeros(len(scored_layouts[0][0]))
    scored_layout[permutation] = scored_layouts[0][0]
    ncbit = deflated_circuit.num_clbits 
    nqbit = deflated_circuit.num_qubits 
    rdef_circ = QuantumCircuit(nqbit,ncbit)
    rdef_circ.compose(deflated_circuit,permutation,inplace=True)
    return list(scored_layout),rdef_circ

def permutation_tracking(circuit:QuantumCircuit):
    nqbit = circuit.num_qubits
    circuit.barrier(range(nqbit))
    Creg = ClassicalRegister(nqbit,"perm")
    circuit.add_register(Creg)
    for i in range(nqbit):
        circuit.measure(i,Creg[i])

def extract_physical_mapping(circuit:QuantumCircuit,nqbits):
    mapping = {}
    for M in circuit.data[-nqbits:]:
           qubit = M.qubits[0]
           cbit = M.clbits[0]
           cbit = circuit.find_bit(cbit).registers
           for r in cbit:
               if r[0].name == "perm":
                    qbit = circuit.find_bit(qubit).registers
                    mapping[qbit[0][1]] = r[1]
                    break
    return mapping

def mapping_to_permutation(mapping:dict):
    p = []
    for k in sorted(mapping.keys()):
        p.append(mapping[k])
    return p


def optimize_init_layout(circuit,backend,num_tries,level=3,seed=1123123,initial_layout = None):
    return find_best_layout(circuit,backend,num_tries,level,seed,initial_layout=initial_layout)
