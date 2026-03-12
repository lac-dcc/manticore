import os
import re
import pandas as pd
import subprocess
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

input_files_path = "./input_verilog"
temporary_files_path = "./tmp_mlir"
output_files_path = "./generated_verilog"
circt_bin_path = os.path.expanduser("~/circt/build/bin")
circt_plugin_path = ".././build"

os.makedirs(temporary_files_path, exist_ok=True)
os.makedirs(output_files_path, exist_ok=True)
os.makedirs(input_files_path, exist_ok=True)

def process_file(file_name):
    
    to_hw = ["circt-verilog", "--ir-hw", f"{input_files_path}/{file_name}.v",
             "-o", f"{temporary_files_path}/{file_name}.hw.mlir"]

    run_pass = [f"{circt_bin_path}/circt-opt", f"{temporary_files_path}/{file_name}.hw.mlir" ,
        f"--load-pass-plugin={circt_plugin_path}/SliceExtractorPass.so" ,
        "--pass-pipeline=builtin.module(slice-extractor)",
        "-o", f"{temporary_files_path}/{file_name}.outlined.mlir"]

    try:
        hw_result = subprocess.run(to_hw, capture_output=True, timeout=60)
        if hw_result.returncode != 0:
            return None  
    except subprocess.TimeoutExpired:
        return None  

    try:
        did_it_work = subprocess.run(run_pass, capture_output=True, text=True, timeout=60)
        benchmarking_string = did_it_work.stderr
    except subprocess.TimeoutExpired:
        if os.path.exists(f"{temporary_files_path}/{file_name}.hw.mlir"):
            os.remove(f"{temporary_files_path}/{file_name}.hw.mlir")
        return None

    if did_it_work.returncode != 0 or "NewModules=" not in benchmarking_string:
        if os.path.exists(f"{temporary_files_path}/{file_name}.hw.mlir"):
            os.remove(f"{temporary_files_path}/{file_name}.hw.mlir")
        return None

    try:
        new_modules = int(re.search(r"NewModules=(\d+)", benchmarking_string).group(1))
        replaced_instances = int(re.search(r"ReplacedInstances=(\d+)", benchmarking_string).group(1))
        ops_saved = int(re.search(r"OpsSaved=(\d+)", benchmarking_string).group(1))
        max_slice_size = int(re.search(r"MaxSliceSize=(\d+)", benchmarking_string).group(1))
        max_slice_inputs = int(re.search(r"MaxSliceInputs=(\d+)", benchmarking_string).group(1))
    except (AttributeError, ValueError):
        return None  

    for sufixo in [".hw.mlir", ".outlined.mlir", ".cleaned.mlir"]:
        tmp_file = f"{temporary_files_path}/{file_name}{sufixo}"
        if os.path.exists(tmp_file):
            os.remove(tmp_file)

    return {
        "Arquivo": file_name,
        "NewModules": new_modules,
        "ReplacedInstances": replaced_instances,
        "OpsSaved": ops_saved,
        "MaxSliceSize": max_slice_size,
        "MaxSliceInputs": max_slice_inputs
    }


if __name__ == "__main__":
    
    dir_files = [os.path.splitext(f)[0] for f in os.listdir(input_files_path) if f.endswith(".v")]
    
    final_results = []

    print(f"Starting processing of {len(dir_files)} files using 16 cores...")

    with ProcessPoolExecutor(max_workers=16) as executor:
        
        futures = {executor.submit(process_file, fname): fname for fname in dir_files}
        
        for future in tqdm(as_completed(futures), total=len(dir_files), desc="Processando", unit="arq"):
            try:
                result = future.result()

                if result is not None:
                    final_results.append(result)
            except Exception:
                continue

    results_df = pd.DataFrame(final_results)
    results_df.to_csv("./Results_by_file.csv", index=False)
    
    print("\n" + "="*45)
    print("BENCHMARK FINISHED")
    print(f"Files processes succesfuly: {len(final_results)}")
    print("="*45)

