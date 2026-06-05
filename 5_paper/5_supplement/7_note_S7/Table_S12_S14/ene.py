from pathlib import Path
import argparse

PDBIDS = [
    "1ecq",
    "1h0r",
    "1we2",
    "2aa9",
    "2ael",
    "2bjm",
    "2gtv",
    "2os9",
    "2q9y",
    "2ybs",
    "3b9s",
    "4gui",
    "4lm3",
    "4pnc",
    "6cjw",
    "6o4x",
]

parser = argparse.ArgumentParser()
parser.add_argument("input_dir", type=str)
parser.add_argument("output_file", type=str)
parser.add_argument("--pattern", type=str, default="*.ene")
parser.add_argument("--start-col", type=int, default=8)
args = parser.parse_args()

INPUT_DIR = Path(args.input_dir)
OUTPUT_FILE = Path(args.output_file)
START_COL = args.start_col

if not INPUT_DIR.is_dir():
    raise NotADirectoryError(f"input directory not found: {INPUT_DIR}")

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


def is_zero(x):
    try:
        return float(x) == 0.0
    except ValueError:
        return False


def find_map_dirs(input_dir):
    map_dirs = [
        p for p in input_dir.iterdir()
        if p.is_dir() and p.name.startswith("map_")
    ]

    def sort_key(p):
        try:
            return int(p.name.split("_", 1)[1])
        except Exception:
            return p.name

    map_dirs = sorted(map_dirs, key=sort_key)

    if len(map_dirs) == 0:
        raise FileNotFoundError(f"no map_* folders found in {input_dir}")

    return map_dirs


def extract_values_from_file(ene_file, pdbid):
    target_map = f"{pdbid}_0"
    target_occ = pdbid

    with open(ene_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.split()

            if len(parts) <= START_COL:
                continue

            map_pdbid = parts[0]
            occ_pdbid = parts[1]

            if map_pdbid != target_map:
                continue

            if occ_pdbid != target_occ:
                continue

            if not (
                is_zero(parts[2])
                and is_zero(parts[3])
                and is_zero(parts[4])
                and is_zero(parts[5])
            ):
                continue

            return parts[START_COL:]

    return None


def extract_values_from_map_dir(map_dir, pdbid):
    direct_file = map_dir / f"{pdbid}.ene"

    if direct_file.is_file():
        return extract_values_from_file(direct_file, pdbid)

    for ene_file in sorted(map_dir.glob(args.pattern)):
        values = extract_values_from_file(ene_file, pdbid)

        if values is not None:
            return values

    return None


map_dirs = find_map_dirs(INPUT_DIR)

results = {}

for map_dir in map_dirs:
    map_results = {}

    for pdbid in PDBIDS:
        values = extract_values_from_map_dir(map_dir, pdbid)
        map_results[pdbid] = values

    results[map_dir.name] = map_results


MAP_COL_W = 8
OCC_COL_W = 6
VAL_COL_W = 18

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    for map_i, map_dir in enumerate(map_dirs):
        if map_i > 0:
            out.write("\n")

        map_name = map_dir.name
        map_results = results[map_name]

        max_cols = 0
        for pdbid in PDBIDS:
            values = map_results[pdbid]
            if values is not None:
                max_cols = max(max_cols, len(values))

        out.write(f"{map_name}\n")
        out.write(f"{'map':<{MAP_COL_W}s} {'occ':<{OCC_COL_W}s}")

        for j in range(max_cols):
            col_name = f"col{START_COL + j:02d}"
            out.write(f" {col_name:>{VAL_COL_W}s}")

        out.write("\n")

        for pdbid in PDBIDS:
            map_pdbid = f"{pdbid}_0"
            occ_pdbid = pdbid
            values = map_results[pdbid]

            out.write(f"{map_pdbid:<{MAP_COL_W}s} {occ_pdbid:<{OCC_COL_W}s}")

            if values is None:
                for _ in range(max_cols):
                    out.write(f" {'nan':>{VAL_COL_W}s}")
            else:
                for j in range(max_cols):
                    if j < len(values):
                        out.write(f" {values[j]:>{VAL_COL_W}s}")
                    else:
                        out.write(f" {'nan':>{VAL_COL_W}s}")

            out.write("\n")


print(f"input directory = {INPUT_DIR}")
print(f"map folders     = {', '.join([p.name for p in map_dirs])}")
print(f"pdbids          = {len(PDBIDS)}")
print(f"start column    = {START_COL}")
print(f"output file     = {OUTPUT_FILE}")

