import pandas as pd

df = pd.read_csv("resultados3.csv") 

outliers = [
    "11090_OpenABC_leaf_level_verilog_gf12_bp_quad_bsg_array_concentrate_static_1b_128.v",
    "12036_OpenABC_leaf_level_verilog_nangate45_bp_quad_bsg_mesh_stitch_width_p130_x_max_p2_y_max_p1.v",
    "12037_OpenABC_leaf_level_verilog_nangate45_bp_quad_bsg_mesh_stitch_width_p130_x_max_p2_y_max_p2.v"
]

df = df[df["size"] > 512]
# df = df[~df["file"].isin(outliers)]

mn = df["memory-non-vectorized"].mean()
mv = df["memory-vectorized"].mean()

tn = df["time-non-vectorized"].mean()
tv = df["time-vectorized"].mean()

pd.set_option('display.max_colwidth', None)

# Se quiser mostrar todas as linhas
pd.set_option('display.max_rows', None)

# print(tv / tn)
# print(mv / mn)

print(df["file"])





