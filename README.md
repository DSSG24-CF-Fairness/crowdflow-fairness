# crowdflow-fairness
#### This repository contains the code and instructions for implementing the experiments described in the research paper *Planning for Equity: A Framework for Measuring and Benchmarking Fairness of Generative Crowd-Flow Models*.

---
# Install 
To set up the environment, install the dependencies by using: 
```bash
pip install -r requirements.txt
```
---


# Provided Libraries
- **DeepGravity:**
  - `deepgravity_steep5`: DeepGravity with steepness parameter set to 5.
  - `deepgravity_steep20`: DeepGravity with steepness parameter set to 20.

- **Gravity Model:**
  - `gravity_model_steep5`: Gravity model with steepness parameter set to 5.
  - `gravity_model_steep20`: Gravity model with steepness parameter set to 20.

- **Statsmodels Library:** This is used by the gravity models. Minor fixes have been applied to prevent training overflow issues in rare cases.

---

# Workflow

### Step 1: Data Preparation
$LOCATION can either be NY or WA. 
1. Convert raw data located in `data/$LOCATION/raw/` using the script `data/$LOCATION/raw/0_raw_to_original.ipynb`.
This will generate the following files in folder `data/$LOCATION/`:
- `boundary.geojson`
- `tessellation.geojson`
- `flow.csv`

2. Add external data to the `data/$LOCATION/` folder:
- `demographics.csv`: Obtained from the Census Bureau.
- `features.csv`: Extracted from OpenStreetMaps.

---

### Step 2: Data Preprocessing
1. Run the scripts `experiment/1_train_test.py` and `experiment/2_sampling.py`.
These scripts will process the raw data and generate processed data in `processed_data/` folder. 
For example:
- `processed_data/NY_steep5/`
- `processed_data/NY_steep20/`
- `processed_data/WA_steep5/`
- `processed_data/WA_steep20/`

---


### Step 3: Data Integration
Run the script `experiment/3_data_integration.py` to integrate data into the `processed_data` folder. This script generates datasets such as:
- `new_york0` to `new_york24`
- `washington0` to `washington24`

Each folder contains datasets sampled with different fairness sampling strategies. Below is a detailed explanation of dataset naming conventions:

| Dataset Name         | Sampling Method                                | Random Seed |
|----------------------|-----------------------------------------------|-------------|
| new_york0 / washington0 | Unbiased                                      | NA          |
| new_york1 / washington1 | Ascending Demographic Disparity Sampling      | 1           |
| new_york2 / washington2 | Ascending Demographic Disparity Sampling      | 2           |
| new_york3 / washington3 | Ascending Demographic Disparity Sampling      | 3           |
| new_york4 / washington4 | Ascending Demographic Disparity Sampling      | 4           |
| new_york5 / washington5 | Ascending Demographic Disparity Sampling      | 5           |
| new_york6 / washington6 | Ascending Demographic Disparity No Sampling   | NA          |
| new_york7 / washington7 | Descending Demographic Disparity Sampling     | 1           |
| new_york8 / washington8 | Descending Demographic Disparity Sampling     | 2           |
| new_york9 / washington9 | Descending Demographic Disparity Sampling     | 3           |
| new_york10 / washington10 | Descending Demographic Disparity Sampling   | 4           |
| new_york11 / washington11 | Descending Demographic Disparity Sampling   | 5           |
| new_york12 / washington12 | Descending Demographic Disparity No Sampling | NA          |
| new_york13 / washington13 | Ascending Disparity Sampling                 | 1           |
| new_york14 / washington14 | Ascending Disparity Sampling                 | 2           |
| new_york15 / washington15 | Ascending Disparity Sampling                 | 3           |
| new_york16 / washington16 | Ascending Disparity Sampling                 | 4           |
| new_york17 / washington17 | Ascending Disparity Sampling                 | 5           |
| new_york18 / washington18 | Ascending Disparity No Sampling              | NA          |
| new_york19 / washington19 | Descending Disparity Sampling                | 1           |
| new_york20 / washington20 | Descending Disparity Sampling                | 2           |
| new_york21 / washington21 | Descending Disparity Sampling                | 3           |
| new_york22 / washington22 | Descending Disparity Sampling                | 4           |
| new_york23 / washington23 | Descending Disparity Sampling                | 5           |
| new_york24 / washington24 | Descending Disparity No Sampling             | NA          |



2. Manually copy the results to the respective `deepgravity_steep{X}/data` folder.

---


### Step 4: Model Execution
- For **Gravity Models**, run the script `4_g_run.py` in the `gravity_model_steep{X}` folder.
- For **DeepGravity Models**, run the script `5_run.py` in the `deepgravity_steep{X}` folder.

---

### Step 5: Evaluation
1. Run `experiment/6_evaluation.py` to evaluate model performance.
2. Generate evaluation plots using `experiment/7_evaluation_plot.py`.
---

### Final Outputs:
The final results and evaluation plots will help benchmark the fairness of the generative crowd-flow models. Use the generated results to analyze disparities and evaluate model effectiveness under different sampling strategies.


