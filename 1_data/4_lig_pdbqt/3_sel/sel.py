import os
import sys


def main():
    if len(sys.argv) != 5:
        print(f"Usage: python {sys.argv[0]} input_file output_dir N_grid N_type")
        sys.exit(1)

    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    N_grid = int(sys.argv[3])
    N_type = int(sys.argv[4])

    output_file = os.path.join(output_dir, f"sel_grid{N_grid}_type{N_type}")

    os.makedirs(output_dir, exist_ok=True)

    selected = []

    with open(input_file, "r", encoding="utf-8") as fin:
        for line in fin:
            parts = line.split()

            if len(parts) < 4:
                continue

            pdbid = parts[0]
            diameter = float(parts[1])
            ngrid = int(parts[2])
            num_atom_types = int(parts[3])

            if ngrid <= N_grid and num_atom_types <= N_type:
                selected.append((num_atom_types, diameter, line))

    selected.sort(key=lambda x: (x[0], x[1]), reverse=True)

    with open(output_file, "w", encoding="utf-8") as fout:
        for _, _, line in selected:
            fout.write(line)

    print(f"Done: {output_file}")


main()

