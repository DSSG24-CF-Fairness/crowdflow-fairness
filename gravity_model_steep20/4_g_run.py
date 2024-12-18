import os
import sys

root_dir_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
print("project root path", root_dir_path)
sys.path.insert(0, os.path.abspath(root_dir_path))
print("using overflow fixed statsmodels library")
sys.path.append(os.path.abspath('../gravity_model_steep20'))
from gravity import *

folder_names = ['WA_steep20', 'NY_steep20']


for folder_name in folder_names:
    # Run model for all train data
    test_flow_path = f"../processed_data/{folder_name}/test/test_flow.csv"

    for dirpath, dirname, filenames in os.walk(f'../processed_data/{folder_name}/train'):
        for idx, filename in enumerate(filenames):
            if 'flow' in filename:
                # try:
                file_path = os.path.join(dirpath, filename)
                print("Running model:", file_path)

                tessellation_train = gpd.read_file(f"../processed_data/{folder_name}/train/train_tessellation.geojson")
                tessellation_test = gpd.read_file(f"../processed_data/{folder_name}/test/test_tessellation.geojson")

                grav_Model(tessellation_train, tessellation_test, file_path, test_flow_path, "gravity_singly_constrained", 'flows', folder_name = folder_name)
                # except Exception as e:
                #     print("Error Occurred for file:", file_path)