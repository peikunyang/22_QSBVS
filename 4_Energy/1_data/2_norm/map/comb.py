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


def build_merged_vector(folder):
    arrays = []

    for name in MAP_ORDER:
        map_file = folder / name

        if map_file.is_file():
            arrays.append(read_map_values(map_file))

            if len(arrays) == N_MAPS:
                break

    while len(arrays) < N_MAPS:
        arrays.append(np.zeros(MAP_SIZE, dtype=np.float64))

    merged = np.concatenate(arrays)

    if merged.size != N_MAPS * MAP_SIZE:
        raise ValueError(
            f"merged length error: {folder} "
            f"got {merged.size}, expected {N_MAPS * MAP_SIZE}"
        )

    norm_before = np.linalg.norm(merged)

    if norm_before > 0:
        merged_norm = merged / norm_before
    else:
        merged_norm = merged.copy()

    return merged_norm, norm_before


def process_one_map_dir(map_dir, output_map_dir):
    output_map_dir.mkdir(parents=True, exist_ok=True)

    subdirs = sorted([p for p in map_dir.iterdir() if p.is_dir()])

    if not subdirs:
        raise ValueError(f"no subfolders found in: {map_dir}")

    length_file = output_map_dir / "len"

    ok_count = 0
    fail_count = 0

    with open(length_file, "w", encoding="utf-8") as length_out:
        for folder in subdirs:
            output_file = output_map_dir / f"{folder.name}.bin"

            try:
                merged_norm, norm_before = build_merged_vector(folder)

                merged_norm.astype(np.float64).tofile(output_file)

                length_out.write(
                    f"{folder.name} {norm_before:.12e}\n"
                )

                ok_count += 1

            except Exception as e:
                print(f"[FAIL] {map_dir.name}/{folder.name}: {e}")
                fail_count += 1

    return ok_count, fail_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_root")
    parser.add_argument("output_root")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    if not input_root.is_dir():
        raise FileNotFoundError(f"input_root not found: {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)

    map_dirs = sorted([
        p for p in input_root.iterdir()
        if p.is_dir() and p.name.startswith("map_")
    ])

    if not map_dirs:
        raise ValueError(f"no map_* folders found in: {input_root}")

    total_ok = 0
    total_fail = 0

    for map_dir in map_dirs:
        output_map_dir = output_root / map_dir.name

        ok_count, fail_count = process_one_map_dir(
            map_dir,
            output_map_dir
        )

        total_ok += ok_count
        total_fail += fail_count

        print(
            f"[done] {map_dir.name}: "
            f"OK {ok_count}, FAIL {fail_count}"
        )

    print(f"[done] output_root: {output_root}")
    print(f"[done] total OK: {total_ok}, total FAIL: {total_fail}")


if __name__ == "__main__":
    main()

