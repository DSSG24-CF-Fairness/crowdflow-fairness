import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import multiprocessing

sys.path.append(os.path.abspath('../preprocessing'))
from data_integration_for_DG_and_NLG import *

if __name__ == '__main__':
    datasets = ["NY"]
    # Create a pool of processes, one for each dataset
    processes = []
    for dataset in datasets:
        p = multiprocessing.Process(target = run_data_integration_form_DG_and_NLG, args=(dataset,))
        processes.append(p)
        p.start()  # Start each process

    # Wait for all processes to complete
    for p in processes:
        p.join()