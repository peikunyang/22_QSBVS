from pathlib import Path
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("pytorch_file", type=str)
parser.add_argument("qiskit_file", type=str)
parser.add_argument("out_csv", type=str)
args = parser.parse_args()

PYTORCH_FILE = Path(args.pytorch_file)
QISKIT_FILE = Path(args.qiskit_file)
OUT_CSV = Path(args.out_csv)
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)


def make_key(map_name, occ_name, shift_z, shift_y, shift_x, rot_index):
    return (
        map_name,
        occ_name,
        int(float(shift_z)),
        int(float(shift_y)),
        int(float(shift_x)),
        int(float(rot_index)),
    )


def load_pytorch_energy(path):
    arr = np.loadtxt(path, dtype=str, usecols=range(9), ndmin=2)

    data = {}

    for row in arr:
        key = make_key(
            row[0],
            row[1],
            row[2],
            row[3],
            row[4],
            row[5],
        )

        energy = float(row[8])

        if key in data:
            raise ValueError(f"duplicated key in PyTorch file: {key}")

        data[key] = energy

    return data


def load_qiskit_energy(path):
    arr = np.loadtxt(path, dtype=str, usecols=range(7), ndmin=2)

    data = {}

    for row in arr:
        key = make_key(
            row[0],
            row[1],
            row[2],
            row[3],
            row[4],
            row[5],
        )

        energy = float(row[6])

        if key in data:
            raise ValueError(f"duplicated key in Qiskit file: {key}")

        data[key] = energy

    return data


pytorch_data = load_pytorch_energy(PYTORCH_FILE)
qiskit_data = load_qiskit_energy(QISKIT_FILE)

rows = []
missing_count = 0

for key in sorted(qiskit_data):
    if key not in pytorch_data:
        missing_count += 1
        continue

    qiskit_energy = qiskit_data[key]
    pytorch_energy = pytorch_data[key]

    rows.append(
        [
            qiskit_energy,
            pytorch_energy,
        ]
    )

if len(rows) == 0:
    raise ValueError("no matched rows found")

data = np.array(rows, dtype=np.float64)

np.savetxt(
    OUT_CSV,
    data,
    delimiter=",",
    header="qiskit_energy,pytorch_energy",
    comments="",
    fmt="%.16e",
)

print(f"wrote {OUT_CSV}")
print(f"pytorch file   = {PYTORCH_FILE}")
print(f"qiskit file    = {QISKIT_FILE}")
print(f"matched rows   = {len(rows)}")
print(f"missing rows   = {missing_count}")
print("x column       = qiskit_energy")
print("y column       = pytorch_energy")

