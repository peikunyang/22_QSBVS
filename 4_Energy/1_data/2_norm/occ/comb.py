import argparse
import numpy as np
from pathlib import Path

N = 32
MAP_SIZE = N * N * N
N_MAPS = 4

MAP_ORDER = [
    "e",
    "A",
    "C",
    "HD",
    "N",
    "NA",
    "OA",
]


def read_map_values(map_file):
    values = []

    with open(map_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            for p in line.split():
                values.append(float(p))

    arr = np.asarray(values, dtype=np.float64)

    if arr.size != MAP_SIZE:
        raise ValueError(
            f"wrong number of values: {map_file} "
            f"got {arr.size}, expected {MAP_SIZE}"
        )

    return arr


def build_merged_vector(pdbid_dir):
    arrays = []

    for name in MAP_ORDER:
        map_file = pdbid_dir / name

        if map_file.is_file():
            arrays.append(read_map_values(map_file))

            if len(arrays) == N_MAPS:
                break

    while len(arrays) < N_MAPS:
        arrays.append(np.zeros(MAP_SIZE, dtype=np.float64))

    merged = np.concatenate(arrays)

    if merged.size != N_MAPS * MAP_SIZE:
        raise ValueError(
            f"merged length error: {pdbid_dir} "
            f"got {merged.size}, expected {N_MAPS * MAP_SIZE}"
        )

    norm_before = np.linalg.norm(merged)

    if norm_before > 0:
        merged_norm = merged / norm_before
    else:
        merged_norm = merged.copy()

    return merged_norm, norm_before


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    length_file = output_dir / "len"

    pdbid_dirs = sorted([p for p in input_dir.iterdir() if p.is_dir()])

    if not pdbid_dirs:
        raise ValueError(f"no PDBID folders found in: {input_dir}")

    ok_count = 0
    fail_count = 0

    with open(length_file, "w", encoding="utf-8") as length_out:
        for pdbid_dir in pdbid_dirs:
            output_file = output_dir / f"{pdbid_dir.name}.bin"

            try:
                merged_norm, norm_before = build_merged_vector(pdbid_dir)

                merged_norm.astype(np.float64).tofile(output_file)

                length_out.write(
                    f"{pdbid_dir.name} {norm_before:.12e}\n"
                )

                ok_count += 1

            except Exception as e:
                print(f"[FAIL] {pdbid_dir.name}: {e}")
                fail_count += 1

    print(f"[done] output_dir: {output_dir}")
    print(f"[done] length_file: {length_file}")
    print(f"[done] OK: {ok_count}, FAIL: {fail_count}")


if __name__ == "__main__":
    main()

