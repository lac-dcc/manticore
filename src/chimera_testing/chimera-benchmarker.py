from design_benchmarker import DesignBenchmarker
import os
from tqdm import tqdm
import pandas as pd


class ChimeraBenchmarker:
    def __init__(self, sample_size):
        self.sample_size = sample_size
        self.statistics_df = pd.DataFrame(columns=["design", "vectorized", "analyze-time", "elaborate-time", "memory-usage"])

    def update_statistics(self, design, vectorized_statistics, non_vectorized_statistics):
        prefix_vectorized = [design, True]
        prefix_non_vectorized = [design, False]

        for statistic in vectorized_statistics:
            concated = prefix_vectorized + statistic
            self.statistics_df.loc[len(self.statistics_df)] = concated

        for statistic in non_vectorized_statistics:
            concated = prefix_non_vectorized + statistic
            self.statistics_df.loc[len(self.statistics_df)] = concated

        self.statistics_df.to_csv("output.csv", index=True)
    
    def benchmark(self):
        design_list = [f for f in os.listdir("../../designs/real-vectorized") 
                if os.path.isfile(os.path.join("../../designs/real-vectorized/", f))]

        for design in tqdm(design_list, desc="Processamento dos designs"):
            design_benchmarker = DesignBenchmarker(design, self.sample_size)
            vectorized_statistics, non_vectorized_statistics = design_benchmarker.benchmark() 
            self.update_statistics(design, vectorized_statistics, non_vectorized_statistics)



aux = ChimeraBenchmarker(2)
aux.benchmark()





