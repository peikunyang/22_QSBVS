import pandas as pd

autodock_file = "../1_data/1_Autodock_ene"
q32_file = "../1_data/2_Q32_ene"
q37_file = "../1_data/3_Q37_ene"

output_file = "Figs6"

map_cols = [f"map_{i}" for i in range(10)]

df_auto = pd.read_csv(autodock_file, sep=r"\s+")
df_q32 = pd.read_csv(q32_file, sep=r"\s+")
df_q37 = pd.read_csv(q37_file, sep=r"\s+")

rows = []

for col in map_cols:
    for i in range(len(df_auto)):
        rows.append([
            df_auto.loc[i, col],
            df_q32.loc[i, col],
            df_q37.loc[i, col]
        ])

out = pd.DataFrame(rows, columns=["Autodock", "Q32", "Q37"])
out.to_csv(output_file, sep="\t", index=False, float_format="%.3f")

print(f"Saved to {output_file}")

