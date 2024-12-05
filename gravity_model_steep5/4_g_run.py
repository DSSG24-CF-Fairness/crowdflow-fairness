import os
import sys


sys.path.append(os.path.abspath('../gravity_model'))
from gravity import *

folder_names = ['WA']


for folder_name in folder_names:
    # Run model for all train data
    test_flow_path = f"../processed_data/{folder_name}/test/test_flow.csv"

    for dirpath, dirname, filenames in os.walk(f'../processed_data/{folder_name}/train'):
        for idx, filename in enumerate(filenames):
            if 'svi_2_descending_biased_flow_6_steep_5.csv' in filename:
                file_path = os.path.join(dirpath, filename)
                print("Running model:", file_path)

                tessellation_train = gpd.read_file(f"../processed_data/{folder_name}/train/train_tessellation.geojson")
                tessellation_test = gpd.read_file(f"../processed_data/{folder_name}/test/test_tessellation.geojson")

                grav_Model(tessellation_train, tessellation_test, file_path, test_flow_path, "gravity_singly_constrained", 'flows', folder_name = folder_name)
