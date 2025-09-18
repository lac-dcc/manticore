import pandas as pd
import os
from tqdm import tqdm
import subprocess

def get_vector_size(design):
    result = subprocess.run(['bash', './get-vector-size.sh', design], stdout=subprocess.PIPE, text=True)
    output = result.stdout.strip()

    return int(output)
    


def save_sizes():
    df = pd.read_csv("resultados2.csv") 
    print(df.columns)

    sizes = []

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processando"):
        val = get_vector_size(row["file"])
        sizes.append(val)

    df["size"] = sizes

    df.to_csv("resultados3.csv")

save_sizes()
