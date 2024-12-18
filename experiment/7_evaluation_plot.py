import argparse
import os
import random
import sys
import time

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import concurrent.futures
from pathlib import Path
import csv


sys.path.append(os.path.abspath('../evaluation'))
from eval_plot import *


steepness_factors = ['steep5', 'steep20']
location_names = ['NY', 'WA']
performance_types = ['CPC', 'overestimation', 'underestimation']
metric_types = ['kl_divergence']

for steepness_factor in steepness_factors:
    for location_name in location_names:
        for performance_type in performance_types:
            for metric_type in metric_types:
                plot_unfairness_vs_performance(steepness_factor, location_name, performance_type, metric_type)
                time.sleep(1)