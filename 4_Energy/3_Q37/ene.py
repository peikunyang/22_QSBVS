from pathlib import Path
import argparse
import numpy as np
import torch

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
parser.add_argument("map_dir", type=str)
parser.add_argument("occ_dir", type=str)
parser.add_argument("out_dir", type=str)
parser.add_argument("--norm_atol", type=float, default=1e-8)
parser.add_argument("--rot_batch", type=int, default=2)
parser.add_argument("--shot_n_start", type=int, default=None)
parser.add_argument("--shot_n_end", type=int, default=None)
parser.add_argument("--repeat", type=int, default=1)
args = parser.parse_args()

if args.repeat <= 0:
    raise ValueError("--repeat must be > 0")

if (args.shot_n_start is None) != (args.shot_n_end is None):
    raise ValueError("--shot_n_start and --shot_n_end must be used together")

DO_SHOT = args.shot_n_start is not None

if DO_SHOT:
    if args.shot_n_start < 0:
        raise ValueError("--shot_n_start must be >= 0")

    if args.shot_n_end < args.shot_n_start:
        raise ValueError("--shot_n_end must be >= --shot_n_start")

    if args.shot_n_end > 18:
        raise ValueError("current binomial simulation supports up to about 10^18 shots")

    SHOT_N_LIST = list(range(args.shot_n_start, args.shot_n_end + 1))
    SHOT_NUMBERS = [10 ** n for n in SHOT_N_LIST]
else:
    SHOT_N_LIST = []
    SHOT_NUMBERS = []

MAP_ROOT_DIR = Path(args.map_dir)
OCC_DIR = Path(args.occ_dir)
OUT_DIR = Path(args.out_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

OCC_NORM_FILE = OCC_DIR / "len"

TYPE_SIZE = 4
GRID_SIZE = 32
DATA_SIZE = TYPE_SIZE * GRID_SIZE * GRID_SIZE * GRID_SIZE

BASE_CONTROL_SIZE = 2 ** 14
OCC_SELECT_SIZE = 16
MAP_SELECT_SIZE = 2
CONTROL_SIZE = BASE_CONTROL_SIZE * OCC_SELECT_SIZE * MAP_SELECT_SIZE

device = torch.device("cuda")
dtype = torch.float64

rng = np.random.default_rng()


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


def load_vector_to_gpu(path):
    arr64 = read_vector_binary_np(path, DATA_SIZE)
    norm64 = np.linalg.norm(arr64)

    if not np.isclose(norm64, 1.0, atol=args.norm_atol):
        raise ValueError(
            f"vector norm is not 1: {path}, norm = {norm64:.16e}"
        )

    tensor = torch.from_numpy(arr64).to(device=device, dtype=dtype)
    tensor = tensor.reshape(TYPE_SIZE, GRID_SIZE, GRID_SIZE, GRID_SIZE)

    return tensor, norm64


def rz90(v):
    return torch.rot90(v, k=1, dims=(2, 3))


def rz180(v):
    return torch.rot90(v, k=2, dims=(2, 3))


def ry90(v):
    return torch.rot90(v, k=-1, dims=(1, 3))


def ry180(v):
    return torch.rot90(v, k=2, dims=(1, 3))


def rx90(v):
    return torch.rot90(v, k=1, dims=(1, 2))


def apply_rotation(v, rot_index):
    out = v

    if (rot_index >> 4) & 1:
        out = rz90(out)

    if (rot_index >> 3) & 1:
        out = rz180(out)

    if (rot_index >> 2) & 1:
        out = ry90(out)

    if (rot_index >> 1) & 1:
        out = ry180(out)

    if rot_index & 1:
        out = rx90(out)

    return out.contiguous()


def build_shift_arrays():
    shift_bits = np.arange(512, dtype=np.int64)

    q23 = (shift_bits >> 0) & 1
    q24 = (shift_bits >> 1) & 1
    q25 = (shift_bits >> 2) & 1

    q26 = (shift_bits >> 3) & 1
    q27 = (shift_bits >> 4) & 1
    q28 = (shift_bits >> 5) & 1

    q29 = (shift_bits >> 6) & 1
    q30 = (shift_bits >> 7) & 1
    q31 = (shift_bits >> 8) & 1

    shift_x = np.where(q25 == 1, 0, -4) + 2 * q24 + q23
    shift_y = np.where(q28 == 1, 0, -4) + 2 * q27 + q26
    shift_z = np.where(q31 == 1, 0, -4) + 2 * q30 + q29

    return shift_z, shift_y, shift_x


def build_base_output_columns():
    base = np.arange(BASE_CONTROL_SIZE, dtype=np.int64)

    rot_index = base & 31
    shift_bits = base >> 5

    shift_z_512, shift_y_512, shift_x_512 = build_shift_arrays()

    shift_z = shift_z_512[shift_bits]
    shift_y = shift_y_512[shift_bits]
    shift_x = shift_x_512[shift_bits]

    return rot_index, shift_z, shift_y, shift_x


def build_gpu_shift_indices():
    shift_z_np, shift_y_np, shift_x_np = build_shift_arrays()
    n = GRID_SIZE

    z = torch.arange(n, device=device, dtype=torch.long).view(1, n, 1, 1)
    y = torch.arange(n, device=device, dtype=torch.long).view(1, 1, n, 1)
    x = torch.arange(n, device=device, dtype=torch.long).view(1, 1, 1, n)

    shift_z = torch.tensor(shift_z_np, device=device, dtype=torch.long).view(512, 1, 1, 1)
    shift_y = torch.tensor(shift_y_np, device=device, dtype=torch.long).view(512, 1, 1, 1)
    shift_x = torch.tensor(shift_x_np, device=device, dtype=torch.long).view(512, 1, 1, 1)

    idx_z = (z - shift_z) % n
    idx_y = (y - shift_y) % n
    idx_x = (x - shift_x) % n

    return idx_z, idx_y, idx_x


def compute_base_dots_gpu(receptor_map, ligand_occ, idx_z, idx_y, idx_x):
    rotated = torch.stack(
        [apply_rotation(ligand_occ, r) for r in range(32)],
        dim=0,
    )

    dots_base = torch.empty(BASE_CONTROL_SIZE, device=device, dtype=dtype)
    shift_ids = torch.arange(512, device=device, dtype=torch.long)

    for start in range(0, 32, args.rot_batch):
        end = min(start + args.rot_batch, 32)

        v = rotated[start:end]
        shifted = v[:, :, idx_z, idx_y, idx_x]

        dots = torch.einsum(
            "btszyx,tzyx->bs",
            shifted,
            receptor_map,
        )

        rot_ids = torch.arange(start, end, device=device, dtype=torch.long).view(-1, 1)
        base_ids = rot_ids + 32 * shift_ids.view(1, -1)

        dots_base[base_ids.reshape(-1)] = dots.reshape(-1)

        del shifted, dots

    return dots_base.cpu().numpy().astype(np.float64, copy=False)


def simulate_one_shot_number(p0, p1, scale_factor, shots):
    samples = np.empty((args.repeat, BASE_CONTROL_SIZE), dtype=np.float64)

    pair_prob = p0 + p1

    p0_ratio = np.divide(
        p0,
        pair_prob,
        out=np.zeros_like(p0, dtype=np.float64),
        where=pair_prob > 0.0,
    )

    for r in range(args.repeat):
        pair_counts = rng.binomial(shots, pair_prob)
        count0 = rng.binomial(pair_counts, p0_ratio)
        count1 = pair_counts - count0

        p0_sample = count0.astype(np.float64) / float(shots)
        p1_sample = count1.astype(np.float64) / float(shots)

        dots_sample = CONTROL_SIZE * (p0_sample - p1_sample)
        samples[r] = dots_sample * scale_factor

    shot_mean = np.mean(samples, axis=0)

    if args.repeat > 1:
        shot_std = np.std(samples, axis=0, ddof=1)
    else:
        shot_std = np.zeros(BASE_CONTROL_SIZE, dtype=np.float64)

    return shot_mean, shot_std


def simulate_shot_range_mean_std(p0, p1, scale_factor):
    results = []

    for shots in SHOT_NUMBERS:
        shot_mean, shot_std = simulate_one_shot_number(
            p0,
            p1,
            scale_factor,
            shots,
        )
        results.append((shot_mean, shot_std))

    return results


def find_map_dirs(map_root_dir):
    map_root_dir = Path(map_root_dir)

    if not map_root_dir.is_dir():
        raise NotADirectoryError(f"map dir not found: {map_root_dir}")

    dirs = [
        p for p in map_root_dir.iterdir()
        if p.is_dir() and p.name.startswith("map_")
    ]

    def sort_key(p):
        try:
            return int(p.name.split("_", 1)[1])
        except Exception:
            return p.name

    dirs = sorted(dirs, key=sort_key)

    if len(dirs) == 0:
        return [map_root_dir], False

    return dirs, True


if not OCC_DIR.is_dir():
    raise NotADirectoryError(f"occ dir not found: {OCC_DIR}")

MAP_DIRS, HAS_MAP_SUBDIRS = find_map_dirs(MAP_ROOT_DIR)

occ_norm_table = read_norm_table(OCC_NORM_FILE)

ligand_occs = []
ligand_occ_norm_values = []
ligand_occ_norm_keys = []
ligand_vector_norms = []

for pdbid in PDBIDS:
    path = OCC_DIR / f"{pdbid}.bin"
    tensor, vec_norm = load_vector_to_gpu(path)
    norm_value, norm_key = find_norm_value(occ_norm_table, path)

    ligand_occs.append(tensor)
    ligand_vector_norms.append(vec_norm)
    ligand_occ_norm_values.append(norm_value)
    ligand_occ_norm_keys.append(norm_key)

rot_index_col, shift_z_col, shift_y_col, shift_x_col = build_base_output_columns()
idx_z, idx_y, idx_x = build_gpu_shift_indices()

fmt = [
    "%6s",
    "% 4s",
    "%2.0f",
    "%2.0f",
    "%2.0f",
    "%2.0f",
    "%17.10e",
    "%17.10e",
    "%17.10e",
]

if DO_SHOT:
    for _ in SHOT_NUMBERS:
        fmt.extend(
            [
                "%17.10e",
                "%17.10e",
            ]
        )

global_prob_sum = 0.0
global_inner_min = np.inf
global_inner_max = -np.inf
global_inner_scaled_min = np.inf
global_inner_scaled_max = -np.inf

with torch.no_grad():
    for MAP_DIR in MAP_DIRS:
        MAP_NORM_FILE = MAP_DIR / "len"
        map_norm_table = read_norm_table(MAP_NORM_FILE)

        if HAS_MAP_SUBDIRS:
            CURRENT_OUT_DIR = OUT_DIR / MAP_DIR.name
        else:
            CURRENT_OUT_DIR = OUT_DIR

        CURRENT_OUT_DIR.mkdir(parents=True, exist_ok=True)

        map_prob_sum = 0.0
        map_inner_min = np.inf
        map_inner_max = -np.inf
        map_inner_scaled_min = np.inf
        map_inner_scaled_max = -np.inf

        for receptor_pdbid in PDBIDS:
            receptor_maps = []
            receptor_map_norm_values = []
            receptor_map_norm_keys = []
            receptor_vector_norms = []

            receptor_map_files = [
                MAP_DIR / f"{receptor_pdbid}_0.bin",
                MAP_DIR / f"{receptor_pdbid}_1.bin",
            ]

            for path in receptor_map_files:
                tensor, vec_norm = load_vector_to_gpu(path)
                norm_value, norm_key = find_norm_value(map_norm_table, path)

                receptor_maps.append(tensor)
                receptor_vector_norms.append(vec_norm)
                receptor_map_norm_values.append(norm_value)
                receptor_map_norm_keys.append(norm_key)

            result_file = CURRENT_OUT_DIR / f"{receptor_pdbid}.ene"

            file_prob_sum = 0.0
            file_inner_min = np.inf
            file_inner_max = -np.inf
            file_inner_scaled_min = np.inf
            file_inner_scaled_max = -np.inf

            with open(result_file, "w", encoding="utf-8") as out:
                for map_index in range(MAP_SELECT_SIZE):
                    receptor_map = receptor_maps[map_index]
                    map_name = f"{receptor_pdbid}_{map_index}"

                    for occ_index in range(OCC_SELECT_SIZE):
                        ligand_occ = ligand_occs[occ_index]
                        occ_name = PDBIDS[occ_index]

                        dots = compute_base_dots_gpu(
                            receptor_map,
                            ligand_occ,
                            idx_z,
                            idx_y,
                            idx_x,
                        )

                        scale_factor = (
                            receptor_map_norm_values[map_index]
                            * ligand_occ_norm_values[occ_index]
                        )

                        dots_scaled = dots * scale_factor

                        p0 = (1.0 + dots) / (2.0 * CONTROL_SIZE)
                        p1 = (1.0 - dots) / (2.0 * CONTROL_SIZE)

                        map_col = np.full(BASE_CONTROL_SIZE, map_name, dtype=object)
                        occ_col = np.full(BASE_CONTROL_SIZE, occ_name, dtype=object)

                        n_extra_cols = 2 * len(SHOT_NUMBERS) if DO_SHOT else 0
                        data = np.empty((BASE_CONTROL_SIZE, 9 + n_extra_cols), dtype=object)

                        data[:, 0] = map_col
                        data[:, 1] = occ_col
                        data[:, 2] = shift_z_col
                        data[:, 3] = shift_y_col
                        data[:, 4] = shift_x_col
                        data[:, 5] = rot_index_col
                        data[:, 6] = p0
                        data[:, 7] = p1
                        data[:, 8] = dots_scaled

                        if DO_SHOT:
                            shot_results = simulate_shot_range_mean_std(
                                p0,
                                p1,
                                scale_factor,
                            )

                            col = 9
                            for shot_mean, shot_std in shot_results:
                                data[:, col] = shot_mean
                                data[:, col + 1] = shot_std
                                col += 2

                            del shot_results

                        np.savetxt(out, data, fmt=fmt, delimiter=" ")

                        file_prob_sum += float(np.sum(p0 + p1))
                        file_inner_min = min(file_inner_min, float(np.min(dots)))
                        file_inner_max = max(file_inner_max, float(np.max(dots)))
                        file_inner_scaled_min = min(file_inner_scaled_min, float(np.min(dots_scaled)))
                        file_inner_scaled_max = max(file_inner_scaled_max, float(np.max(dots_scaled)))

                        del dots, dots_scaled, p0, p1, data, map_col, occ_col

            map_prob_sum += file_prob_sum
            map_inner_min = min(map_inner_min, file_inner_min)
            map_inner_max = max(map_inner_max, file_inner_max)
            map_inner_scaled_min = min(map_inner_scaled_min, file_inner_scaled_min)
            map_inner_scaled_max = max(map_inner_scaled_max, file_inner_scaled_max)

            global_prob_sum += file_prob_sum
            global_inner_min = min(global_inner_min, file_inner_min)
            global_inner_max = max(global_inner_max, file_inner_max)
            global_inner_scaled_min = min(global_inner_scaled_min, file_inner_scaled_min)
            global_inner_scaled_max = max(global_inner_scaled_max, file_inner_scaled_max)

            print(f"wrote {result_file}")
            print(f"  map folder           = {MAP_DIR.name}")
            print(f"  receptor pdbid       = {receptor_pdbid}")
            print(f"  total probability    = {file_prob_sum:.16e}")
            print(f"  min normalized inner = {file_inner_min:.16e}")
            print(f"  max normalized inner = {file_inner_max:.16e}")
            print(f"  min scaled inner     = {file_inner_scaled_min:.16e}")
            print(f"  max scaled inner     = {file_inner_scaled_max:.16e}")

            del receptor_maps
            del receptor_map_norm_values
            del receptor_map_norm_keys
            del receptor_vector_norms
            torch.cuda.empty_cache()

        print(f"finished map folder    = {MAP_DIR.name}")
        print(f"  map probability sum  = {map_prob_sum:.16e}")
        print(f"  map min inner        = {map_inner_min:.16e}")
        print(f"  map max inner        = {map_inner_max:.16e}")
        print(f"  map min scaled       = {map_inner_scaled_min:.16e}")
        print(f"  map max scaled       = {map_inner_scaled_max:.16e}")

print(f"device                 = {device}")
print(f"dtype                  = {dtype}")
print(f"map root dir           = {MAP_ROOT_DIR}")
print(f"occ dir                = {OCC_DIR}")
print(f"occ norm file          = {OCC_NORM_FILE}")
print(f"output directory       = {OUT_DIR}")
print(f"type size              = {TYPE_SIZE}")
print(f"grid size              = {GRID_SIZE}")
print(f"data size              = {DATA_SIZE}")
print(f"base control size      = {BASE_CONTROL_SIZE}")
print(f"occ select size        = {OCC_SELECT_SIZE}")
print(f"map select size        = {MAP_SELECT_SIZE}")
print(f"total control size     = {CONTROL_SIZE}")
print(f"rot batch              = {args.rot_batch}")
print(f"repeat                 = {args.repeat}")
print(f"shot simulation        = {DO_SHOT}")

if DO_SHOT:
    print(f"shot n range           = {args.shot_n_start} ... {args.shot_n_end}")
    print(f"shot numbers           = {', '.join([str(s) for s in SHOT_NUMBERS])}")

print(f"map folders            = {', '.join([p.name for p in MAP_DIRS])}")
print(f"global probability sum = {global_prob_sum:.16e}")
print(f"global min inner       = {global_inner_min:.16e}")
print(f"global max inner       = {global_inner_max:.16e}")
print(f"global min scaled      = {global_inner_scaled_min:.16e}")
print(f"global max scaled      = {global_inner_scaled_max:.16e}")

print("ligand occ files:")
for i, pdbid in enumerate(PDBIDS):
    print(
        f"  occ_index {i:2d}: {pdbid}.bin "
        f"norm_key={ligand_occ_norm_keys[i]} "
        f"norm={ligand_occ_norm_values[i]:.16e} "
        f"vector_norm={ligand_vector_norms[i]:.16e}"
    )

