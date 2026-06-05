import os
import sys
import pandas as pd

def read_map_file(filepath, colname):
    df = pd.read_csv(filepath, delim_whitespace=True)
    if "pdbid" not in df.columns or "total_0" not in df.columns:
        raise ValueError(f"{filepath} 缺少 pdbid 或 total_0 欄位")
    df = df[["pdbid", "total_0"]].copy()
    df = df.rename(columns={"total_0": colname})
    return df

def main():
    if len(sys.argv) != 3:
        print("Usage: python make_map_table.py INPUT_FOLDER OUTPUT_FILE")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_file = sys.argv[2]

    merged = None

    for i in range(10):
        fname = f"map_{i}"
        fpath = os.path.join(input_folder, fname)

        if not os.path.exists(fpath):
            raise FileNotFoundError(f"找不到檔案: {fpath}")

        df = read_map_file(fpath, fname)

        if merged is None:
            merged = df
        else:
            merged = pd.merge(merged, df, on="pdbid", how="outer")

    merged = merged.sort_values("pdbid").reset_index(drop=True)

    merged.insert(0, "map", merged["pdbid"].astype(str) + "_0")
    merged.insert(1, "occ", merged["pdbid"])

    merged = merged.drop(columns=["pdbid"])

    merged.to_csv(output_file, sep="\t", index=False, float_format="%.3f")

if __name__ == "__main__":
    main()

