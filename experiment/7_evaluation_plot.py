import argparse
import os
import random
import sys
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import concurrent.futures
from pathlib import Path
import csv


sys.path.append(os.path.abspath('../evaluation'))
from eval_plot import *


location_name = 'NY'
performance_type = 'CPC'
metric_type = 'kl_divergence'
plot_unfairness_vs_performance(location_name, performance_type, metric_type)