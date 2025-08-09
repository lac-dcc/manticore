import os
import subprocess
import pandas as pd
from tqdm import tqdm

def designs_list():
    designs = []
    path = "/home/ullas/manticore/manticore/scripts/vectorizable-designs/vectorized/"

    for file in os.listdir(path):
        designs.append(file)

    return designs

def run_analysis(file, vectorized):
    path = "../../scripts/vectorizable-designs/vectorized/" if vectorized else  "../../scripts/vectorizable-designs/default/"
    path += file

    tcl_file_content = f'analyze -sv {path}\nelaborate\nexit 0'
    with open("jasper-commands.tcl", "w") as tcl:
        tcl.write(tcl_file_content)

    result = subprocess.run(['bash', './jasper_executor.sh'], stdout=subprocess.PIPE, text=True)
    output = result.stdout
    values = output.split(' ')
    values[1] = values[1][:-1]

    return (int(values[0]), int(values[1])) 



def get_data_from_design(design):
    time_default, mem_default = run_analysis(design, False)
    time_vectorized, mem_vectorized = run_analysis(design, True)
    
    return [time_default, mem_default, time_vectorized, mem_vectorized]


def get_data():
    designs = designs_list()


    results = []

    for design in tqdm(designs, desc="Analise dos Designs"):
        data = get_data_from_design(design)
        results.append(data)


    df = pd.DataFrame(results, columns=["time-non-vectorized", "memory-non-vectorized", "time-vectorized", "memory-vectorized"])
    df.to_csv("resultados.csv")


get_data()


