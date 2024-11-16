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



location_name = 'WA'
accuracy_type = 'CPC'
metric_type = 'kl_divergence'
plot_fairness_vs_accuracy(location_name, accuracy_type, metric_type)