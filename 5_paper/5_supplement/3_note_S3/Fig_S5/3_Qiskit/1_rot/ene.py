from pathlib import Path
import argparse
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import XGate, SwapGate


parser = argparse.ArgumentParser()
parser.add_argument("map_file", type=str)
parser.add_argument("occ_file", type=str)
parser.add_argument("out_file", type=str)
parser.add_argument("--norm_atol", type=float, default=1e-8)
parser.add_argument("--map_norm_file", type=str, default=None)
parser.add_argument("--occ_norm_file", type=str, default=None)
args = parser.parse_args()


TYPE_SIZE = 4
GRID_SIZE = 32
DATA_SIZE = TYPE_SIZE * GRID_SIZE * GRID_SIZE * GRID_SIZE

DATA_QUBITS = 17
ANCILLA_QUBIT = 17

Q_RX90 = 18
Q_RY180 = 19
Q_RY90 = 20
Q_RZ180 = 21
Q_RZ90 = 22

Q_TX_P1 = 23
Q_TX_P2 = 24
Q_TX_M4 = 25

TOTAL_QUBITS = 26
ROT_CONTROL_SIZE = 32
TX_CONTROL_SIZE = 8
CONTROL_SIZE = ROT_CONTROL_SIZE * TX_CONTROL_SIZE

NX = list(range(0, 5))
NY = list(range(5, 10))
NZ = list(range(10, 15))

DATA_BLOCK_SIZE = 2 ** DATA_QUBITS
CONTROL_BLOCK_SIZE = 2 ** (DATA_QUBITS + 1)

MAP_FILE = Path(args.map_file)
OCC_FILE = Path(args.occ_file)
OUT_FILE = Path(args.out_file)
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

MAP_NORM_FILE = Path(args.map_norm_file) if args.map_norm_file is not None else MAP_FILE.parent / "len"
OCC_NORM_FILE = Path(args.occ_norm_file) if args.occ_norm_file is not None else OCC_FILE.parent / "len"


def read_vector_binary_np(path, expected_size):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"file not found: {path}")

    arr = np.fromfile(path, dtype=np.float64)

    if arr.size != expected_size:
        raise ValueError(
            f"wrong number of values in {path}: "
            f"got {arr.size}, expected {expected_size}"
        )

    if not np.all(np.isfinite(arr)):
        raise ValueError(f"NaN or Inf found in {path}")

    return arr


def read_norm_table(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"norm file not found: {path}")

    table = {}

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.split()

            if len(parts) < 2:
                continue

            table[parts[0]] = float(parts[1])

    return table


def find_norm_value(norm_table, data_file):
    data_file = Path(data_file)

    candidates = [
        data_file.name,
        data_file.stem,
    ]

    for key in candidates:
        if key in norm_table:
            return norm_table[key], key

    raise KeyError(
        f"cannot find norm value for {data_file}\n"
        f"tried keys: {candidates}"
    )


def load_vector(path):
    arr = read_vector_binary_np(path, DATA_SIZE)
    norm = np.linalg.norm(arr)

    if not np.isclose(norm, 1.0, atol=args.norm_atol):
        raise ValueError(
            f"vector norm is not 1: {path}, norm = {norm:.16e}"
        )

    return arr, norm


def mcx_states(qc, controls, states, target):
    flipped = []

    for q, s in zip(controls, states):
        if s == 0:
            qc.x(q)
            flipped.append(q)

    if len(controls) == 0:
        qc.x(target)
    elif len(controls) == 1:
        qc.cx(controls[0], target)
    else:
        qc.mcx(controls, target)

    for q in reversed(flipped):
        qc.x(q)


def controlled_x(qc, control, control_state, target, extra_controls=None, extra_states=None):
    if extra_controls is None:
        extra_controls = []

    if extra_states is None:
        extra_states = []

    controls = [ANCILLA_QUBIT, control] + list(extra_controls)
    states = [1, control_state] + list(extra_states)

    mcx_states(qc, controls, states, target)


def controlled_swap(qc, control, qa, qb):
    qc.append(
        SwapGate().control(2),
        [ANCILLA_QUBIT, control, qa, qb],
    )


def controlled_swap_registers(qc, control, reg_a, reg_b):
    for qa, qb in zip(reg_a, reg_b):
        controlled_swap(qc, control, qa, qb)


def controlled_flip_register(qc, control, reg):
    for q in reg:
        qc.append(
            XGate().control(2),
            [ANCILLA_QUBIT, control, q],
        )


def apply_rotation_circuit(qc):
    controlled_swap_registers(qc, Q_RZ90, NX, NY)
    controlled_flip_register(qc, Q_RZ90, NY)

    controlled_flip_register(qc, Q_RZ180, NX)
    controlled_flip_register(qc, Q_RZ180, NY)

    controlled_swap_registers(qc, Q_RY90, NX, NZ)
    controlled_flip_register(qc, Q_RY90, NX)

    controlled_flip_register(qc, Q_RY180, NX)
    controlled_flip_register(qc, Q_RY180, NZ)

    controlled_swap_registers(qc, Q_RX90, NY, NZ)
    controlled_flip_register(qc, Q_RX90, NZ)


def add_p1(qc, ctrl, reg):
    controlled_x(qc, ctrl, 1, reg[0])
    controlled_x(qc, ctrl, 1, reg[1], [reg[0]], [0])
    controlled_x(qc, ctrl, 1, reg[2], [reg[0], reg[1]], [0, 0])
    controlled_x(qc, ctrl, 1, reg[3], [reg[0], reg[1], reg[2]], [0, 0, 0])
    controlled_x(qc, ctrl, 1, reg[4], [reg[0], reg[1], reg[2], reg[3]], [0, 0, 0, 0])


def add_p2(qc, ctrl, reg):
    controlled_x(qc, ctrl, 1, reg[1])
    controlled_x(qc, ctrl, 1, reg[2], [reg[1]], [0])
    controlled_x(qc, ctrl, 1, reg[3], [reg[1], reg[2]], [0, 0])
    controlled_x(qc, ctrl, 1, reg[4], [reg[1], reg[2], reg[3]], [0, 0, 0])


def add_m4_open(qc, ctrl, reg):
    controlled_x(qc, ctrl, 0, reg[2])
    controlled_x(qc, ctrl, 0, reg[3], [reg[2]], [1])
    controlled_x(qc, ctrl, 0, reg[4], [reg[2], reg[3]], [1, 1])


def apply_translation_x_circuit(qc):
    add_m4_open(qc, Q_TX_M4, NX)
    add_p2(qc, Q_TX_P2, NX)
    add_p1(qc, Q_TX_P1, NX)


def shift_x_from_id(shift_id):
    tx_p1 = shift_id & 1
    tx_p2 = (shift_id >> 1) & 1
    tx_m4 = (shift_id >> 2) & 1

    return (0 if tx_m4 == 1 else -4) + 2 * tx_p2 + tx_p1


map_norm_table = read_norm_table(MAP_NORM_FILE)
occ_norm_table = read_norm_table(OCC_NORM_FILE)

map_name = MAP_FILE.stem
occ_name = OCC_FILE.stem

receptor_vec, receptor_vector_norm = load_vector(MAP_FILE)
ligand_vec, ligand_vector_norm = load_vector(OCC_FILE)

receptor_norm_value, receptor_norm_key = find_norm_value(map_norm_table, MAP_FILE)
ligand_norm_value, ligand_norm_key = find_norm_value(occ_norm_table, OCC_FILE)

scale_factor = receptor_norm_value * ligand_norm_value

state_size = 2 ** TOTAL_QUBITS
init_state = np.zeros(state_size, dtype=np.complex128)

init_state[0:DATA_SIZE] = receptor_vec / np.sqrt(2.0)
init_state[DATA_BLOCK_SIZE:DATA_BLOCK_SIZE + DATA_SIZE] = ligand_vec / np.sqrt(2.0)

init_norm = np.linalg.norm(init_state)

if not np.isclose(init_norm, 1.0, atol=args.norm_atol):
    raise ValueError(f"initial state norm is not 1: norm = {init_norm:.16e}")

qc = QuantumCircuit(TOTAL_QUBITS)

for q in [Q_RX90, Q_RY180, Q_RY90, Q_RZ180, Q_RZ90]:
    qc.h(q)

for q in [Q_TX_P1, Q_TX_P2, Q_TX_M4]:
    qc.h(q)

apply_rotation_circuit(qc)
apply_translation_x_circuit(qc)

qc.h(ANCILLA_QUBIT)

sv = Statevector(init_state)
sv_after = sv.evolve(qc)
sv_data = sv_after.data

rows = []

for shift_id in range(8):
    shift_x = shift_x_from_id(shift_id)

    for rot_index in range(32):
        control_id = rot_index + ROT_CONTROL_SIZE * shift_id
        block_start = control_id * CONTROL_BLOCK_SIZE

        ancilla0_start = block_start
        ancilla1_start = block_start + DATA_BLOCK_SIZE

        amp0 = sv_data[ancilla0_start:ancilla0_start + DATA_SIZE]
        amp1 = sv_data[ancilla1_start:ancilla1_start + DATA_SIZE]

        p0 = float(np.vdot(amp0, amp0).real)
        p1 = float(np.vdot(amp1, amp1).real)

        dot = CONTROL_SIZE * (p0 - p1)
        energy = dot * scale_factor

        rows.append(
            [
                map_name,
                occ_name,
                0,
                0,
                shift_x,
                rot_index,
                energy,
            ]
        )

data = np.array(rows, dtype=object)

fmt = [
    "%6s",
    "%4s",
    "%2.0f",
    "%2.0f",
    "%2.0f",
    "%2.0f",
    "%17.10e",
]

np.savetxt(OUT_FILE, data, fmt=fmt, delimiter=" ")

energies = np.array([float(row[6]) for row in rows], dtype=np.float64)

print(f"wrote {OUT_FILE}")
print(f"map file                 = {MAP_FILE}")
print(f"occ file                 = {OCC_FILE}")
print(f"map norm file            = {MAP_NORM_FILE}")
print(f"occ norm file            = {OCC_NORM_FILE}")
print(f"map name                 = {map_name}")
print(f"occ name                 = {occ_name}")
print(f"q0-q16                   = data qubits")
print(f"q17                      = ancilla")
print(f"q18                      = rx90")
print(f"q19                      = ry180")
print(f"q20                      = ry90")
print(f"q21                      = rz180")
print(f"q22                      = rz90")
print(f"q23                      = tx +1")
print(f"q24                      = tx +2")
print(f"q25                      = tx -4 open-control")
print(f"total qubits             = {TOTAL_QUBITS}")
print(f"state size               = {state_size}")
print(f"control size             = {CONTROL_SIZE}")
print(f"initial state norm       = {init_norm:.16e}")
print(f"scale factor             = {scale_factor:.16e}")
print(f"min energy               = {float(np.min(energies)):.16e}")
print(f"max energy               = {float(np.max(energies)):.16e}")
print(f"receptor norm key        = {receptor_norm_key}")
print(f"receptor norm value      = {receptor_norm_value:.16e}")
print(f"receptor vector norm     = {receptor_vector_norm:.16e}")
print(f"ligand norm key          = {ligand_norm_key}")
print(f"ligand norm value        = {ligand_norm_value:.16e}")
print(f"ligand vector norm       = {ligand_vector_norm:.16e}")

