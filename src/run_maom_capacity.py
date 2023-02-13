"""
Gets SOC reference values from the
MaOM capacity output.

- `moc_t` = mineral-associated soil organic carbon stocks for topsoil (0-30cm) [kgC/m2]
- `moc_s` = mineral-associated soil organic carbon stocks for subsoil (30-100cm) [kgC/m2]
- `mocmax_t` = mineralogical carbon capacity for topsoil (0-30cm) [kgC/m2]
- `mocmax_s` = mineralogical carbon capacity for subsoil (30-100cm) [kgC/m2]
"""

import xarray as xr
from shapely.geometry import Point
import geopandas as gpd
import os
import pandas as pd
pd.options.mode.chained_assignment = 'raise'


output_folder = 'output'
os.makedirs(output_folder, exist_ok=True)

data_folder = os.path.join(
    'data',
    'maom_capacity'
)


def load_netCDF_data(path):
    df = xr.open_dataset(path)
    df = df.to_dataframe()
    df = df.reset_index()
    geom = [Point(x, y) for x, y in zip(df['lon'], df['lat'])]
    gdf = gpd.GeoDataFrame(df, geometry=geom)
    return gdf

# Load MOC and MOCmax data


MOC = load_netCDF_data(os.path.join(data_folder, "global_MOC.nc"))
MOCmax = load_netCDF_data(os.path.join(data_folder, "global_MOCmax.nc"))
MOC = MOC.set_crs("EPSG:4326")
MOCmax = MOCmax.set_crs("EPSG:4326")

# Load output from natural references per pedoclimate
# to get selected LUCAS cropland sites

data = pd.read_csv(
    os.path.join(
        'output',
        'natural_references_per_pedoclimate.csv'
    ),
    low_memory=False
)
data.drop("vref", axis=1, inplace=True)

data.loc[:, 'geometry'] = gpd.GeoSeries.from_wkt(data['geometry'])
data = gpd.GeoDataFrame(data, geometry='geometry')
data = data.set_crs("EPSG:4326")


# Extract MOC and MOCmax data at selected LUCAS cropland sites

def get_nearest_data(df, other_df, id_column):
    nearest = other_df.sindex.nearest(df.geometry)
    nearest = pd.DataFrame(
        nearest[1], index=nearest[0], columns=["other_index"])
    nearest[id_column] = df[id_column]
    return nearest


nearest_moc_data = get_nearest_data(data, MOC, 'POINT_ID')

nearest_moc_data = get_nearest_data(data, MOC, 'POINT_ID')
data = data.merge(nearest_moc_data, on='POINT_ID', how='left').join(
    MOC.drop(["geometry", "lat", "lon", "moc_s"], axis=1),
    on="other_index"
).drop("other_index", axis=1)

nearest_mocmax_data = get_nearest_data(data, MOCmax, 'POINT_ID')
data = data.merge(nearest_moc_data, on='POINT_ID', how='left').join(
    MOCmax.drop(["geometry", "lat", "lon", "mocmax_s"], axis=1),
    on="other_index"
).drop("other_index", axis=1)

# convert to Mg/ha
data.loc[:, "moc_t"] = 10 * data["moc_t"]
data.loc[:, "mocmax_t"] = 10 * data["mocmax_t"]

data.loc[:, "deltamoc"] = data.mocmax_t - data.moc_t

# Rename to follow nomenclature of other methods
# (Note: we name TOC what is actually MOC)

data.rename({'mocmax_t': 'vref_stock',
             'deltamoc': 'deltastock'},
            axis=1, inplace=True)

data = data[['POINT_ID', 'vref_stock', 'deltastock']]

# Save results
data.to_csv(
    os.path.join(
        output_folder,
        'maom_capacity.csv'
    ),
    index=False
)
