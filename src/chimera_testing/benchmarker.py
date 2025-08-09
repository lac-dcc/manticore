import subprocess
import os

def save_tcl_file(design_path):
    tcl_file_content = f'analyze -sv v1-{i}.v\nelaborate\nexit 0'
    with open("jasper-commands.tcl", "w") as tcl:
        tcl.write(tcl_file_content)

def get_data_from_jasper():
    data_vectorization = []
    data_non_vectorization = []

    for i in range(1,23):
        tcl_file_content = f'analyze -sv v1-{i}.v\nelaborate\nexit 0'
        with open("jasper-commands.tcl", "w") as tcl:
            tcl.write(tcl_file_content)

        result = subprocess.run(['bash', './jasper_executor.sh'], stdout=subprocess.PIPE, text=True)
        output = result.stdout
        values = output.split(' ')
        values[1] = values[1][:-1]

        data_vectorization.append((int(values[0]), int(values[1])))
        print(i)

    for i in range(1,23):
        tcl_file_content = f'analyze -sv v2-{i}.v\nelaborate\nexit 0'
        with open("jasper-commands.tcl", "w") as tcl:
            tcl.write(tcl_file_content)

        result = subprocess.run(['bash', './jasper_executor.sh'], stdout=subprocess.PIPE, text=True)
        output = result.stdout
        values = output.split(' ')
        values[1] = values[1][:-1]

        data_non_vectorization.append((int(values[0]), int(values[1])))
        print(i)

        
    print(data_vectorization)
    print(data_non_vectorization)

get_data_from_jasper()



