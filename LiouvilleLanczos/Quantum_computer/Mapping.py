
from mapomatic import deflate_circuit, evaluate_layouts, matching_layouts
from qiskit import transpile,QuantumCircuit
import numpy as np


def find_best_layout(circuit:QuantumCircuit,backend,num_tries,level=3,seed=132423,initial_layout = None):
    nqbit = circuit.num_qubits
    circuits_ts = transpile(
        [circuit] * num_tries, backend, optimization_level=level, seed_transpiler=seed,initial_layout=initial_layout
    )
    cx_counts = [circuits_ts[idx].count_ops()["cx"] for idx in range(num_tries)]
    best_idx = np.argmin(cx_counts)
    best_circuit = circuits_ts[best_idx]
    deflated_circuit = deflate_circuit(best_circuit)
    layouts = matching_layouts(deflated_circuit, backend)
    scored_layouts = evaluate_layouts(
        deflated_circuit, layouts, backend
    )  # cost_function = cost_func
    return scored_layouts[0],best_circuit,deflated_circuit

def filter_map(transpiled_circuit,original_circuit):
    """get the mapping of the virtual qubit in the transpiled circuit. 
    For all register present in the original circuit"""
    layout_object = transpiled_circuit._layout
    layout = layout_object.initial_layout
    p_bit_map = layout.get_physical_bits()
    out = {}
    for key in p_bit_map:
        if p_bit_map[key] in original_circuit.qubits:
            out[key] = p_bit_map[key]
    return out

def apply_scored_layout(base_layout_dict,scored_layout):
    """
    Scored layout contain a physical qubit to physical qubit map, in an implicit manner.
    it's actually a list of the physical qubits destination. The index of the list
    refer to the physical qubit, in order, in the circuit supplied to mapomatic.
    """
    layout = []
    for key,val in base_layout_dict.items():
        layout.append((key,val))
    layout.sort(key = lambda x:x[0])
    out = {}
    for i,p in enumerate(layout):
        out[scored_layout[0][i]] = p[1]
    print(layout)
    return out

def optimize_init_layout(circuit,backend,num_tries,level=3,seed=1123123,initial_layout = None):
    scored_layout,best_circuit,deflated_circuit = find_best_layout(circuit,backend,num_tries,level,seed,initial_layout=initial_layout)
    filtered_map = filter_map(best_circuit,circuit)
    scored_layout = apply_scored_layout(filtered_map,scored_layout)
    return scored_layout,deflated_circuit,best_circuit