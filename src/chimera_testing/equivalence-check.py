import os
import subprocess
import random
from tqdm import tqdm
import pandas as pd

class EquivalenceChecker:
    def __init__(self):
        with open("tcl-scripts-jg/equivalence-check.tcl", encoding="utf-8") as f:
            self.tcl_template = f.read()

        self.designs_directory_path = "../../designs/"
        self.result_df = pd.DataFrame(columns=["design", "equivalent"])

    def vectorized_and_non_vectorized_equivalent(self, design):
        new_tcl = self.tcl_template.replace("first.v", f"{self.designs_directory_path}/real-vectorized/{design}")
        new_tcl = new_tcl.replace("second.v", f"{self.designs_directory_path}/non-vectorized/{design}")

        tcl_file_name = "equivalence-check.tcl"

        with open(tcl_file_name, "w", encoding="utf-8") as f:
            f.write(new_tcl)


        result = subprocess.run(['bash', './bash-scripts/equivalence-check.sh', tcl_file_name], stdout=subprocess.PIPE, text=True)
        output = result.stdout.split("\n")

        return output[0] == "True"

    def check_equivalence(self):
        design_list = [f for f in os.listdir("../../designs/real-vectorized") 
                if os.path.isfile(os.path.join("../../designs/real-vectorized/", f))]

        for design in tqdm(design_list, desc="Teste de Equivalencia"):
            row = [design, self.vectorized_and_non_vectorized_equivalent(design)]
            self.result_df.loc[len(self.result_df)] = row

        self.result_df.to_csv("equivalence.csv", index=True)

        subprocess.run(['bash', 'rm *.tcl'], stdout=subprocess.PIPE, text=True)


e = EquivalenceChecker()
e.check_equivalence()





