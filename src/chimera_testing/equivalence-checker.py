import os
import subprocess
import pandas as pd
from tqdm import tqdm

def check_equivalence(design):
    tcl_code = f"""
check_sec -compile_context spec
analyze -sv /home/ullas/manticore/manticore/designs/non-vectorized/{design}
elaborate        

check_sec -compile_context imp
analyze -sv /home/ullas/manticore/manticore/designs/real-vectorized/{design}
elaborate

clock -none
reset -none
check_sec -setup
check_sec -generate
    """

    tcl_code += """
\nif {[catch {check_return {check_sec -prove} proven} err]} {
    exit 1
} else {
    exit 0
}"""

    with open("equivalence.tcl", "w") as tcl:
        tcl.write(tcl_code)

    result = subprocess.run(['bash', './equivalence-executor.sh'], stdout=subprocess.PIPE, text=True)
    output = result.stdout

    return "0" in output




def run_equivalence_check():
    df = pd.read_csv("resultados.csv") 
    
    equivalents = []

    for index, row, in tqdm(df.iterrows(), total=len(df), desc="Processando"):
        equivalents.append(check_equivalence(row["file"]))

    df["equivalent"] = equivalents

    df.to_csv("resultados2.csv")



run_equivalence_check()



