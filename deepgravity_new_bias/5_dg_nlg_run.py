import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

python_path = r"C:\Users\kiki\AppData\Local\anaconda3\envs\pycharm_crowdflow_39\python.exe"
script_path = r"C:\Users\kiki\Documents\GitHub\Python\crowdflow-fairness\deepgravity_new_bias\main.py"

common_args = [
    "--oa-id-column", "GEOID",
    "--flow-origin-column", "geoid_o",
    "--flow-destination-column", "geoid_d",
    "--flow-flows-column", "pop_flows",
    "--epochs", "20",
    "--device", "gpu",
    "--mode", "train"
]

locations = ["new_york_new", "washington_new"]
model_types = ["DG", "NLG"]

def run_command(dataset_name, model_type):
    command = [python_path, script_path, "--dataset", dataset_name, "--model-type", model_type] + common_args
    try:
        print(f"Running for dataset: {dataset_name}")
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred for {dataset_name}: {e}")

def main():
    with ProcessPoolExecutor(max_workers=10) as executor:
        futures = []
        for model_type in model_types:
            for location in locations:
                for i in range(21):
                    dataset_name = f"{location}{i}"
                    futures.append(executor.submit(run_command, dataset_name, model_type))

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Exception encountered: {e}")

if __name__ == "__main__":
    main()
