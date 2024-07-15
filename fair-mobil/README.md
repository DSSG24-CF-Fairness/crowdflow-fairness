# fmb - fair-mobility-models

< ... >

## Installation

```
python3.10 -m pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple fmb==0.0.4
```

## Usage

Two Options:

### In the CLI:

```cli
fmb
```

### In the Code:

```
# look below !
```

## FairMob Class Methods

The FairMob class comprises several methods to assist in fairness analysis:

1. KLDiva Method
   Purpose: Calculate the Kullback-Leibler (KL) and Jensen-Shannon (JS) divergences between two distributions.

   Parameters:
   arr1: First array of data.
   arr2: Second array of data.

   Usage:

   ```
   fair_mob = FairMob()
   js_divergence, kl_divergence = fair_mob.KLDiva(arr1, arr2)
   ```

2. equalArray Method
   Purpose: Upsample the smaller array to match the length of the larger array.

   Parameters:
   arr1: First array of data.
   arr2: Second array of data.

   Usage:

   ```
   equal_arr1, equal_arr2 = fair_mob.equalArray(arr1, arr2)
   ```

3. runFairness Method
   Purpose: Analyze the fairness in data concerning the specified parameters.

   Parameters:
   Key DataFrame parameters: df (main data), measurementdf (measurement data), percentile (percentile for analysis).
   Optional parameters for customizing column names in the dataframes and additional behavior.

   Usage:

   ```
   fair_mob.runFairness(df, measurementdf, 10)
   ```

4. directional_Fairness Method

   Same as runFairness but specifically for directional analysis.

5. bidirectional_fairness Method

   Same as runFairness but supports bidirectional analysis between origins and destinations.

6. drawPlot Method

   Purpose: Use Matplotlib to draw custom plots (not usually called directly by the user).

   Parameters:
   data: Data to plot.

Example Usage

```
from fmb import FairMob
import pandas as pd
import numpy as np

# Sample data

data1 = {'origin': [161, 220], 'destination': [172, 223], 'flow': [35, 29], 'modelFlow': [30, 25]}
data2 = {'FIPS': [161, 172, 220, 223], 'RPL_THEMES': [1.2, 0.9, 1.7, 1.1]}
df = pd.DataFrame(data1)
measure_df = pd.DataFrame(data2)

# Instantiating and using FairMob

fair_mob = FairMob()
fair_mob.runFairness(df, measure_df, 10, realflow='flow', modelFlow='modelFlow', measurementOrigin='FIPS', measurement='RPL_THEMES')
```

This code initializes a FairMob object, then uses it to run a fairness analysis on mobility data, comparing real and modeled flows at specific origin and destination pairs, using resilience data from an external measurement DataFrame.
