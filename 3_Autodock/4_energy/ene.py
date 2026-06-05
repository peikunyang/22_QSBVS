import argparse
from pathlib import Path

EXPECTED_N = 32 * 32 * 32
ELEC_TYPE = "e"


def read_values_exact(file_path, expected_n=EXPECTED_N):
    values = []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            for p in line.strip().split():
                try:
                    values.append(float(p))
                except ValueError:
                    pass

    if len(values) != expected_n:
        raise ValueError(f"{file_path} has {len(values)} values, expected {expected_n}")

    return values


def dot_product(a, b):
    return sum(x * y for x, y in zip(a, b))


def read_one_type_if_exists(folder, name):
    file_path = folder / name

    if not file_path.is_file():
        return None

    return read_values_exact(file_path)


def get_pdbids_from_occ(occ_root):
    return sorted([p.name for p in occ_root.iterdir() if p.is_dir()])


def get_ligand_types(occ_pdb_dir):
    return sorted([p.name for p in occ_pdb_dir.iterdir() if p.is_file()])


def get_map_dirs(map_root):
    return sorted(
        [p for p in map_root.iterdir() if p.is_dir() and p.name.startswith("map_")],
        key=lambda x: int(x.name.split("_")[1])
    )


def calc_energy_one_pair(map_pose_dir, occ_pdb_dir, ligand_types):
    electrostatic = 0.0
    vdw = 0.0
    used_types = []
    missing_types = []

    for t in ligand_types:
        occ_values = read_one_type_if_exists(occ_pdb_dir, t)

        if occ_values is None:
            continue

        map_values = read_one_type_if_exists(map_pose_dir, t)

        if map_values is None:
            missing_types.append(t)
            continue

        value = dot_product(map_values, occ_values)

        if t == ELEC_TYPE:
            electrostatic += value
        else:
            vdw += value

        used_types.append(t)

    total = electrostatic + vdw

    return electrostatic, vdw, total, used_types, missing_types


def process_one_map_dir(map_dir, occ_root):
    lines = []
    all_ligand_types = set()

    for pdbid in get_pdbids_from_occ(occ_root):
        occ_pdb_dir = occ_root / pdbid
        ligand_types = get_ligand_types(occ_pdb_dir)
        all_ligand_types.update(ligand_types)

        if not ligand_types:
            print(f"[skip] {map_dir.name} {pdbid}: no ligand type files")
            continue

        map_pose_0 = map_dir / f"{pdbid}_0"
        map_pose_1 = map_dir / f"{pdbid}_1"

        if not map_pose_0.is_dir():
            print(f"[skip] {map_dir.name} {pdbid}: missing {pdbid}_0")
            continue

        if not map_pose_1.is_dir():
            print(f"[skip] {map_dir.name} {pdbid}: missing {pdbid}_1")
            continue

        try:
            elec0, vdw0, total0, used0, miss0 = calc_energy_one_pair(
                map_pose_0,
                occ_pdb_dir,
                ligand_types,
            )

            elec1, vdw1, total1, used1, miss1 = calc_energy_one_pair(
                map_pose_1,
                occ_pdb_dir,
                ligand_types,
            )

            used_types = sorted(set(used0 + used1))

            line = (
                f"{pdbid:<6}"
                f"{elec0:11.3f}"
                f"{vdw0:11.3f}"
                f"{total0:11.3f}"
                f"{elec1:11.3f}"
                f"{vdw1:11.3f}"
                f"{total1:11.3f}  "
                f"{','.join(used_types)}\n"
            )

            lines.append(line)

            if miss0:
                print(f"[warn] {map_dir.name} {pdbid}_0 missing map types: {','.join(miss0)}")

            if miss1:
                print(f"[warn] {map_dir.name} {pdbid}_1 missing map types: {','.join(miss1)}")

        except Exception as e:
            print(f"[fail] {map_dir.name} {pdbid}: {e}")

    return lines, all_ligand_types


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map_root", required=True)
    parser.add_argument("--occ_root", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    map_root = Path(args.map_root)
    occ_root = Path(args.occ_root)
    out_dir = Path(args.out_dir)

    if not map_root.is_dir():
        raise FileNotFoundError(f"map_root not found: {map_root}")

    if not occ_root.is_dir():
        raise FileNotFoundError(f"occ_root not found: {occ_root}")

    out_dir.mkdir(parents=True, exist_ok=True)

    map_dirs = get_map_dirs(map_root)

    if not map_dirs:
        raise RuntimeError(f"no map_* folders found under {map_root}")

    header = (
        f"{'pdbid':<6}"
        f"{'elec_0':>11}"
        f"{'vdw_0':>11}"
        f"{'total_0':>11}"
        f"{'elec_1':>11}"
        f"{'vdw_1':>11}"
        f"{'total_1':>11}  "
        f"used_types\n"
    )

    all_types_global = set()

    for map_dir in map_dirs:
        lines, all_types = process_one_map_dir(
            map_dir,
            occ_root,
        )

        all_types_global.update(all_types)

        out_file = out_dir / map_dir.name

        with open(out_file, "w", encoding="utf-8") as f:
            f.write(header)
            f.writelines(lines)

        print(f"[OK] {map_dir.name} -> {out_file}  rows={len(lines)}")

    type_file = out_dir / "type"

    with open(type_file, "w", encoding="utf-8") as f:
        for t in sorted(all_types_global):
            f.write(t + "\n")

    print()
    print(f"ligand types saved: {type_file}")
    print("Done")


if __name__ == "__main__":
    main()

