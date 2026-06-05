import argparse
from pathlib import Path

GRID = 0.375
GRID_MIN = -16.0 * GRID
N = 32


def parse_pdbqt_atoms(pdbqt_file):
    atoms = []

    with open(pdbqt_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                q = float(line[70:76])
                atoms.append((x, y, z, q))

    if not atoms:
        raise ValueError(f"no ATOM/HETATM found: {pdbqt_file}")

    return atoms


def add_if_valid(grid, ix, iy, iz, value):
    if 0 <= ix < N and 0 <= iy < N and 0 <= iz < N:
        grid[iz][iy][ix] += value
        return True
    return False


def build_charge_grid(atoms):
    grid = [[[0.0 for _ in range(N)] for _ in range(N)] for _ in range(N)]

    kept = 0
    skipped = 0

    for x, y, z, q in atoms:
        tx = (x - GRID_MIN) / GRID
        ty = (y - GRID_MIN) / GRID
        tz = (z - GRID_MIN) / GRID

        ix0 = int(tx)
        iy0 = int(ty)
        iz0 = int(tz)

        fx = tx - ix0
        fy = ty - iy0
        fz = tz - iz0

        ix1 = ix0 + 1
        iy1 = iy0 + 1
        iz1 = iz0 + 1

        weights = [
            (ix0, iy0, iz0, (1.0 - fx) * (1.0 - fy) * (1.0 - fz)),
            (ix1, iy0, iz0, fx * (1.0 - fy) * (1.0 - fz)),
            (ix0, iy1, iz0, (1.0 - fx) * fy * (1.0 - fz)),
            (ix1, iy1, iz0, fx * fy * (1.0 - fz)),
            (ix0, iy0, iz1, (1.0 - fx) * (1.0 - fy) * fz),
            (ix1, iy0, iz1, fx * (1.0 - fy) * fz),
            (ix0, iy1, iz1, (1.0 - fx) * fy * fz),
            (ix1, iy1, iz1, fx * fy * fz),
        ]

        touched = False

        for ix, iy, iz, w in weights:
            touched |= add_if_valid(grid, ix, iy, iz, q * w)

        if touched:
            kept += 1
        else:
            skipped += 1

    return grid, kept, skipped


def flatten_grid(grid):
    values = []

    for z in range(N):
        for y in range(N):
            for x in range(N):
                values.append(grid[z][y][x])

    return values


def get_pdbid_from_lig_name(stem):
    if not stem.endswith("_lig"):
        raise ValueError(f"unexpected filename: {stem}")

    return stem[:-len("_lig")]


def process_one_file(infile, output_root):
    atoms = parse_pdbqt_atoms(infile)
    grid, kept, skipped = build_charge_grid(atoms)
    values = flatten_grid(grid)

    pdbid = get_pdbid_from_lig_name(infile.stem)

    outdir = output_root / pdbid
    outdir.mkdir(parents=True, exist_ok=True)

    outfile = outdir / "e"

    with open(outfile, "w", encoding="utf-8") as f:
        for i in range(0, len(values), 32):
            chunk = values[i:i + 32]
            f.write(" ".join(f"{v:9.6f}" for v in chunk) + "\n")

    return kept, skipped, outfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_root", required=True, help="input root directory")
    parser.add_argument("-o", "--output_root", required=True, help="output root directory")
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()

    output_root.mkdir(parents=True, exist_ok=True)

    files = sorted(input_root.rglob("*_lig.pdbqt"))

    if not files:
        print("No matching *_lig.pdbqt files found.")
        return

    for infile in files:
        try:
            kept, skipped, outfile = process_one_file(infile, output_root)
            rel = infile.relative_to(input_root)
            print(f"[OK] {rel} -> {outfile}  kept={kept} skipped={skipped}")

        except Exception as e:
            print(f"[FAIL] {infile}  {e}")

    print("Done")


if __name__ == "__main__":
    main()

