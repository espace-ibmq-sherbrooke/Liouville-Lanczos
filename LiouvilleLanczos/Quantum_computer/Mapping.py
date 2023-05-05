
from mapomatic import deflate_circuit, evaluate_layouts, matching_layouts
from qiskit import transpile,QuantumCircuit,ClassicalRegister
import numpy as np


"""
Fiddling to allow sub-circuit optimization. Might break with qiskit updates, i'm afraid
i'm relying on implementation details.

"""

def find_best_layout(circuit:QuantumCircuit,backend,num_tries,level=3,seed=132423,initial_layout = None):
    nqbit = circuit.num_qubits
    circuit = circuit.copy()
    permutation_tracking(circuit)
    circuits_ts = transpile(
        [circuit] * num_tries, backend, optimization_level=level, seed_transpiler=seed,initial_layout=initial_layout
    )
    if 'ecr' in backend.configuration().basis_gates:
        egate = 'ecr'
    else:
        egate = 'cx'
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
                    # print(qbit[0][0].name) I'm a bit worried that in some situtation it wont fetch the right qubit register.
                    # Currently, a transpiled circuit only has one quantum register "q"
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
