import subprocess

python_path = r"C:\Users\fy916\anaconda3\envs\Gravity_OCT31\python.exe"
script_path = r"C:\Users\fy916\Documents\Projects\crowdflow-fairness\deepgravity_new_ny_features\main.py"

common_args = [
    "--oa-id-column", "GEOID",
    "--flow-origin-column", "geoid_o",
    "--flow-destination-column", "geoid_d",
    "--flow-flows-column", "pop_flows",
    "--epochs", "50",
    "--device", "gpu",
    "--mode", "train"
]

locations = ["new_york"]
model_types = ["DG"]

for model_type in model_types:
    for location in locations:
        for i in range(0, 21):
            dataset_name = f"{location}{i}"
            command = [python_path, script_path, "--dataset", dataset_name, "--model-type", model_type] + common_args

            try:
                print(f"Running for dataset: {dataset_name}")
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error occurred for {dataset_name}: {e}")