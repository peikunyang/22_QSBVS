import argparse
import math
from pathlib import Path
import re
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau


PDBIDS = [
    "1ecq", "1h0r", "1we2", "2aa9",
    "2ael", "2bjm", "2gtv", "2os9",
    "2q9y", "2ybs", "3b9s", "4gui",
    "4lm3", "4pnc", "6cjw", "6o4x",
]

EXPECTED_COLS = 33
REF_COL = 9

TARGET_COLS = [9, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
CORR_LABELS = ["pro", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15", "s16", "s17"]

SORT_COLS = [8, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
RANK_LABELS = ["pro", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15", "s16", "s17"]

TARGET_ROW_COLS = [2, 3, 4, 5]
TARGET_ROW_VALUES = [0, 0, 0, 0]

PERCENT_LEVELS = [
    ("1%", 0.01),
    ("10%", 0.10),
]


def natural_key(path):
    parts = re.split(r"(\d+)", path.name)
    return [int(x) if x.isdigit() else x for x in parts]


def get_map_dirs(root):
    out = []

    for p in root.iterdir():
        if not p.is_dir():
            continue

        if not p.name.startswith("map_"):
            continue

        try:
            idx = int(p.name.split("_", 1)[1])
        except ValueError:
            continue

        out.append((idx, p.name))

    out.sort(key=lambda x: x[0])
    return [name for _, name in out]


def read_rows(path):
    rows = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue

            parts = line.split()

            if len(parts) != EXPECTED_COLS:
                print("ERROR")
                print(f"file      : {path}")
                print(f"line      : {line_no}")
                print(f"n_columns : {len(parts)}")
                print(f"expected  : {EXPECTED_COLS}")
                print(f"content   : {line.rstrip()}")
                raise SystemExit(1)

            rows.append((line_no, parts))

    if len(rows) == 0:
        print("ERROR")
        print(f"file   : {path}")
        print("reason : no valid data rows")
        raise SystemExit(1)

    return rows


def find_input_file(root, map_name, pdbid):
    folder = root / map_name

    candidates = [
        folder / f"{pdbid}.ene",
        folder / pdbid,
        folder / f"{pdbid}.txt",
    ]

    for path in candidates:
        if path.exists():
            return path

    matches = sorted(folder.glob(f"{pdbid}*"), key=natural_key)

    if matches:
        return matches[0]

    return None


def safe_corr(x, y):
    if len(x) < 2:
        return np.nan, np.nan, np.nan

    if np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan, np.nan, np.nan

    pearson = pearsonr(x, y)[0]
    spearman = spearmanr(x, y)[0]
    kendall = kendalltau(x, y)[0]

    return pearson, spearman, kendall


def safe_nrmse(ref, target):
    if len(ref) == 0:
        return np.nan

    denominator = np.sqrt(np.mean(ref ** 2))

    if denominator == 0:
        return np.nan

    rmse = np.sqrt(np.mean((target - ref) ** 2))
    return rmse / denominator


def compute_original_metrics(rows, include_nrmse):
    ref_idx = REF_COL - 1
    target_idx = [c - 1 for c in TARGET_COLS]

    data = []

    for _, parts in rows:
        try:
            ref_value = float(parts[ref_idx])
            target_values = [float(parts[idx]) for idx in target_idx]
        except ValueError:
            continue

        data.append([ref_value] + target_values)

    if len(data) == 0:
        return []

    data = np.array(data, dtype=float)
    baseline = data[:, 0]

    results = []

    for i, target_col in enumerate(TARGET_COLS, start=1):
        target = data[:, i]
        pearson, spearman, kendall = safe_corr(baseline, target)

        row = {
            "target_col": target_col,
            "n": len(data),
            "pearson": pearson,
            "spearman": spearman,
            "kendall": kendall,
        }

        if include_nrmse:
            row["nrmse"] = safe_nrmse(baseline, target)

        results.append(row)

    return results


def compute_fixed_map0_metrics(ref_rows, target_rows):
    ref_idx = REF_COL - 1
    target_idx = [c - 1 for c in TARGET_COLS]

    if len(ref_rows) != len(target_rows):
        print("ERROR")
        print(f"reference rows : {len(ref_rows)}")
        print(f"target rows    : {len(target_rows)}")
        print("reason         : row numbers are different")
        raise SystemExit(1)

    baseline = []
    targets = {col: [] for col in TARGET_COLS}

    for (_, ref_parts), (_, target_parts) in zip(ref_rows, target_rows):
        try:
            ref_value = float(ref_parts[ref_idx])

            for col, idx in zip(TARGET_COLS, target_idx):
                targets[col].append(float(target_parts[idx]))

            baseline.append(ref_value)

        except ValueError:
            continue

    baseline = np.array(baseline, dtype=float)

    results = []

    for target_col in TARGET_COLS:
        target = np.array(targets[target_col], dtype=float)
        pearson, spearman, kendall = safe_corr(baseline, target)

        results.append(
            {
                "target_col": target_col,
                "n": len(baseline),
                "pearson": pearson,
                "spearman": spearman,
                "kendall": kendall,
            }
        )

    return results


def to_int_token(x):
    try:
        v = float(x)
    except ValueError:
        return None

    if not v.is_integer():
        return None

    return int(v)


def is_target_row(parts, pdbid):
    if parts[0] != f"{pdbid}_0":
        return False

    if parts[1] != pdbid:
        return False

    values = [to_int_token(parts[i]) for i in TARGET_ROW_COLS]

    return values == TARGET_ROW_VALUES


def find_target_line_no(rows, pdbid):
    for line_no, parts in rows:
        if is_target_row(parts, pdbid):
            return line_no

    return None


def rank_target_and_total(rows, pdbid, sort_col):
    target_line_no = find_target_line_no(rows, pdbid)

    sortable = []

    for line_no, parts in rows:
        try:
            value = float(parts[sort_col])
        except ValueError:
            continue

        sortable.append((value, line_no))

    total = len(sortable)

    if target_line_no is None:
        return "NA", total

    sortable.sort(key=lambda x: (x[0], x[1]))

    for rank, (_, line_no) in enumerate(sortable, start=1):
        if line_no == target_line_no:
            return str(rank), total

    return "NA", total


def threshold_from_total(total, fraction):
    if total <= 0:
        return 0

    return max(1, math.ceil(total * fraction))


def count_percent(block_infos, fraction):
    values = []

    for col_idx in range(len(RANK_LABELS)):
        n = 0

        for info in block_infos:
            rank = info["ranks"][col_idx]
            total = info["totals"][col_idx]

            try:
                rank = int(rank)
            except ValueError:
                continue

            threshold = threshold_from_total(total, fraction)

            if threshold > 0 and rank <= threshold:
                n += 1

        percent = n / 16.0 * 100.0
        values.append(f"{percent:5.1f}")

    return values


def get_section_threshold(block_infos, fraction):
    thresholds = []

    for info in block_infos:
        for total in info["totals"]:
            threshold = threshold_from_total(total, fraction)

            if threshold > 0:
                thresholds.append(threshold)

    if not thresholds:
        return "0"

    if min(thresholds) == max(thresholds):
        return str(thresholds[0])

    return f"{min(thresholds)}-{max(thresholds)}"


def format_table(table):
    widths = []

    for col_idx in range(len(table[0])):
        width = max(len(row[col_idx]) for row in table)
        widths.append(width)

    lines = []

    for row in table:
        parts = []

        for col_idx, value in enumerate(row):
            if col_idx == 0:
                parts.append(f"{value:<{widths[col_idx]}}")
            else:
                parts.append(f"{value:>{widths[col_idx]}}")

        lines.append("  ".join(parts))

    return "\n".join(lines)


def label_from_target_col(target_col):
    return CORR_LABELS[TARGET_COLS.index(target_col)]


def format_matrix_section(title, map_names, summaries, key):
    table = [["target_col"] + map_names]

    for target_col in TARGET_COLS:
        row = [label_from_target_col(target_col)]

        for map_name in map_names:
            value = summaries[map_name][target_col][key]
            row.append(f"{value:.8f}")

        table.append(row)

    return title + "\n" + format_table(table)


def format_top_section(all_blocks):
    sections = []
    map_names = [x[0] for x in all_blocks]

    for label, fraction in PERCENT_LEVELS:
        all_infos = []

        for _, block_infos in all_blocks:
            all_infos.extend(block_infos)

        threshold_text = get_section_threshold(all_infos, fraction)

        values_by_map = {}

        for map_name, block_infos in all_blocks:
            values_by_map[map_name] = count_percent(block_infos, fraction)

        table = [["score"] + map_names]

        for score_idx, score_label in enumerate(RANK_LABELS):
            row = [score_label]

            for map_name in map_names:
                row.append(values_by_map[map_name][score_idx])

            table.append(row)

        sections.append(f"{label} {threshold_text}\n" + format_table(table))

    return "\n\n".join(sections)


def compute_summary(metric_rows_by_map, include_nrmse):
    summaries = {}

    for map_name, metric_rows in metric_rows_by_map.items():
        values = {
            col: {
                "pearson": [],
                "spearman": [],
                "kendall": [],
                "nrmse": [],
            }
            for col in TARGET_COLS
        }

        for row in metric_rows:
            col = row["target_col"]
            values[col]["pearson"].append(row["pearson"])
            values[col]["spearman"].append(row["spearman"])
            values[col]["kendall"].append(row["kendall"])

            if include_nrmse:
                values[col]["nrmse"].append(row["nrmse"])

        summaries[map_name] = {}

        for col in TARGET_COLS:
            summaries[map_name][col] = {
                "pearson_mean": float(np.mean(values[col]["pearson"])),
                "spearman_mean": float(np.mean(values[col]["spearman"])),
                "kendall_mean": float(np.mean(values[col]["kendall"])),
            }

            if include_nrmse:
                summaries[map_name][col]["nrmse_mean"] = float(np.mean(values[col]["nrmse"]))

    return summaries


def build_original_summary_text(map_names, summaries, all_blocks):
    sections = [
        "1_original_summary",
        "",
        format_matrix_section("nRMSE_mean", map_names, summaries, "nrmse_mean"),
        "",
        format_matrix_section("pearson_mean", map_names, summaries, "pearson_mean"),
        "",
        format_matrix_section("spearman_mean", map_names, summaries, "spearman_mean"),
        "",
        format_matrix_section("kendall_mean", map_names, summaries, "kendall_mean"),
        "",
        "Top summary",
        format_top_section(all_blocks),
    ]

    return "\n".join(sections) + "\n"


def build_fixed_summary_text(map_names, summaries, all_blocks):
    sections = [
        "2_fixed_map0_summary",
        "",
        format_matrix_section("pearson_mean", map_names, summaries, "pearson_mean"),
        "",
        format_matrix_section("spearman_mean", map_names, summaries, "spearman_mean"),
        "",
        format_matrix_section("kendall_mean", map_names, summaries, "kendall_mean"),
        "",
        "Top summary",
        format_top_section(all_blocks),
    ]

    return "\n".join(sections) + "\n"


def write_detail_header(f):
    f.write(
        f"{'map':<8s}"
        f"{'pdbid':<8s}"
        f"{'file':<20s}"
        f"{'target_col':>12s}"
        f"{'n':>10s}"
        f"{'pearson':>16s}"
        f"{'spearman':>16s}"
        f"{'kendall':>16s}"
        "\n"
    )


def write_detail_row(f, r):
    f.write(
        f"{r['map']:<8s}"
        f"{r['pdbid']:<8s}"
        f"{r['file']:<20s}"
        f"{r['target_col']:>12d}"
        f"{r['n']:>10d}"
        f"{r['pearson']:>16.8f}"
        f"{r['spearman']:>16.8f}"
        f"{r['kendall']:>16.8f}"
        "\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str)
    parser.add_argument("output_dir", type=str)
    args = parser.parse_args()

    root = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not root.is_dir():
        print("ERROR")
        print(f"input_dir does not exist: {root}")
        raise SystemExit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    map_names = get_map_dirs(root)

    if "map_0" not in map_names:
        print("ERROR")
        print("map_0 does not exist")
        raise SystemExit(1)

    print(f"Input root : {root}", flush=True)
    print(f"Output dir : {output_dir}", flush=True)
    print(f"Maps       : {len(map_names)}", flush=True)
    print("", flush=True)

    original_metric_rows_by_map = {map_name: [] for map_name in map_names}
    fixed_metric_rows_by_map = {map_name: [] for map_name in map_names}

    original_blocks = []
    fixed_blocks = []

    output_3 = output_dir / "3_fixed_map0_detail"

    with open(output_3, "w", encoding="utf-8") as detail_f:
        write_detail_header(detail_f)

        for map_idx, map_name in enumerate(map_names, start=1):
            print(f"[MAP {map_idx}/{len(map_names)}] {map_name}", flush=True)

            original_block_infos = []
            fixed_block_infos = []

            for pdb_idx, pdbid in enumerate(PDBIDS, start=1):
                print(f"  [PDB {pdb_idx}/{len(PDBIDS)}] {pdbid}", flush=True)

                file_path = find_input_file(root, map_name, pdbid)
                ref_file_path = find_input_file(root, "map_0", pdbid)

                original_ranks = ["NA"] * len(SORT_COLS)
                original_totals = [0] * len(SORT_COLS)
                fixed_ranks = ["NA"] * len(SORT_COLS)
                fixed_totals = [0] * len(SORT_COLS)

                if file_path is None:
                    print("    target file: missing", flush=True)
                else:
                    print(f"    target file: {file_path.name}", flush=True)

                    rows = read_rows(file_path)

                    original_ranks = []
                    original_totals = []
                    fixed_ranks = []
                    fixed_totals = []

                    for sort_col in SORT_COLS:
                        rank, total = rank_target_and_total(rows, pdbid, sort_col)
                        original_ranks.append(rank)
                        original_totals.append(total)
                        fixed_ranks.append(rank)
                        fixed_totals.append(total)

                    original_metrics = compute_original_metrics(rows, include_nrmse=True)

                    for m in original_metrics:
                        original_metric_rows_by_map[map_name].append(m)

                    if ref_file_path is None:
                        print("ERROR")
                        print(f"reference file missing: map_0/{pdbid}")
                        raise SystemExit(1)

                    ref_rows = read_rows(ref_file_path)

                    fixed_metrics = compute_fixed_map0_metrics(ref_rows, rows)

                    for m in fixed_metrics:
                        fixed_metric_rows_by_map[map_name].append(m)

                        write_detail_row(
                            detail_f,
                            {
                                "map": map_name,
                                "pdbid": pdbid,
                                "file": file_path.name,
                                "target_col": m["target_col"],
                                "n": m["n"],
                                "pearson": m["pearson"],
                                "spearman": m["spearman"],
                                "kendall": m["kendall"],
                            },
                        )

                    del rows
                    del ref_rows

                original_block_infos.append(
                    {
                        "pdbid": pdbid,
                        "map": map_name,
                        "ranks": original_ranks,
                        "totals": original_totals,
                    }
                )

                fixed_block_infos.append(
                    {
                        "pdbid": pdbid,
                        "map": map_name,
                        "ranks": fixed_ranks,
                        "totals": fixed_totals,
                    }
                )

            original_blocks.append((map_name, original_block_infos))
            fixed_blocks.append((map_name, fixed_block_infos))

            print(
                f"  finished {map_name}: "
                f"original_rows={len(original_metric_rows_by_map[map_name])}, "
                f"fixed_rows={len(fixed_metric_rows_by_map[map_name])}",
                flush=True,
            )
            print("", flush=True)

    print("Computing summaries...", flush=True)

    original_summaries = compute_summary(original_metric_rows_by_map, include_nrmse=True)
    fixed_summaries = compute_summary(fixed_metric_rows_by_map, include_nrmse=False)

    output_1 = output_dir / "1_original_summary"
    output_2 = output_dir / "2_fixed_map0_summary"

    with open(output_1, "w", encoding="utf-8") as f:
        f.write(build_original_summary_text(map_names, original_summaries, original_blocks))

    with open(output_2, "w", encoding="utf-8") as f:
        f.write(build_fixed_summary_text(map_names, fixed_summaries, fixed_blocks))

    print("Done.", flush=True)
    print(f"Output written to: {output_1}", flush=True)
    print(f"Output written to: {output_2}", flush=True)
    print(f"Output written to: {output_3}", flush=True)


main()

