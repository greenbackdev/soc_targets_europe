"""
Gets SOC reference values from the
Data-driven reciprocal modelling output.
"""

import os
import pandas as pd
pd.options.mode.chained_assignment = 'raise'

output_folder = 'output'
os.makedirs(output_folder, exist_ok=True)

data_path = os.path.join(
    'data',
    'data_driven_reciprocal_modelling',
    'cropToGrass_pred_content_no_CN_no_pH.csv'
)

# Load data
data = pd.read_csv(data_path, low_memory=False)

# Treat data
data.drop("lat", axis=1, inplace=True)
data.drop("lon", axis=1, inplace=True)
data.rename({"obs": "toc", "pred": "vref",
            "Point_ID": "POINT_ID"}, axis=1, inplace=True)
data.loc[:, "toc"] = data["toc"] * 10    # convert percentage to mass
data.loc[:, "vref"] = data["vref"] * 10    # convert percentage to mass

# Save results
data.to_csv(
    os.path.join(
        output_folder,
        'data_driven_reciprocal_modelling.csv'
    ),
    index=False
)
