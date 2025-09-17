import subprocess
import os
import random

TCL_FILE_VECTORIZED = "vectorized.tcl"
TCL_FILE_NON_VECTORIZED = "non-vectorized.tcl"

class DesignBenchmarker:

    def __init__(self, design, sample_size):
        self.design = design
        self.sample_size = sample_size
        
        with open("tcl-scripts-jg/jg-analyzis.tcl", encoding="utf-8") as f:
            tcl_template = f.read()

        self.designs_directory_path = "../../designs/"

        tcl_vectorized = tcl_template.replace("file.v", f"{self.designs_directory_path}/real-vectorized/{self.design}")
        tcl_non_vectorized = tcl_template.replace("file.v", f"{self.designs_directory_path}/non-vectorized/{self.design}")
        
        with open(TCL_FILE_VECTORIZED, "w", encoding="utf-8") as out1:
            out1.write(tcl_vectorized)

        with open(TCL_FILE_NON_VECTORIZED, "w", encoding="utf-8") as out2:
            out2.write(tcl_non_vectorized)

        

    def run_analysis(self, vectorized):
        tcl_file = TCL_FILE_VECTORIZED if vectorized else TCL_FILE_NON_VECTORIZED 
        
        result = subprocess.run(['bash', './bash-scripts/exec-analysis.sh', tcl_file], stdout=subprocess.PIPE, text=True)
        output = result.stdout.split("\n")
        output.pop()

        statistics = [float(x) for x in output]
        
        return statistics 

        

    def benchmark(self):
        permutation = [False] * self.sample_size + [True] * self.sample_size
        random.shuffle(permutation)

        vectorized_statistics = []
        non_vectorized_statistics = []

        for flag in permutation:
            statistics = self.run_analysis(flag)

            if flag: vectorized_statistics.append(statistics)
            else: non_vectorized_statistics.append(statistics)

        return (vectorized_statistics, non_vectorized_statistics)




