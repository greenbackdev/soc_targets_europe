"""
Calculates SOC reference values for each pedoclimatic cluster
(output of run_pedoclimatic_clustering.py) using
natural references (grasslands and woodlands). Selects only data that did not
change land cover between 2009 and 2015.
"""

from models.natural_references_per_pedoclimate import CarbonReferenceValuesComputer
from maps_tools import add_scalebar, add_north_arrow

from matplotlib.colors import BoundaryNorm
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import scipy.stats as st
import geopandas as gpd
import os
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = 'raise'

# plt.rcParams.update({'font.size': 22})
plt.rcParams.update({
    "figure.facecolor":  (1.0, 1.0, 1.0, 1.0),  # white
    "axes.facecolor":    (1.0, 1.0, 1.0, 1.0),  # white
    "savefig.facecolor": (1.0, 1.0, 1.0, 0.0),  # transparent
})

output_folder = 'output'
os.makedirs(output_folder, exist_ok=True)
figures_folder = os.path.join(output_folder, 'figures')
os.makedirs(figures_folder, exist_ok=True)
tables_folder = os.path.join(output_folder, 'tables')
os.makedirs(tables_folder, exist_ok=True)

data_path = os.path.join('output', 'pedoclimatic_clustering.csv')

# Load data
data = pd.read_csv(data_path, low_memory=False)

data.loc[:, 'geometry'] = gpd.GeoSeries.from_wkt(data['geometry'])
data = gpd.GeoDataFrame(data, geometry='geometry')
data = data.set_crs("EPSG:4326")

# Select LUCAS sites that did not change land cover between 2009 and 2015
data = data[data.LC1_2009.str[0] == data.LC1_2015.str[0]]

# Supplementary table 1
data_cluster103 = data[(data.Cluster_climate == 10) & (data.Cluster_soil == 3)]
data_cluster103.groupby('LC0_Desc_2015').toc.describe().to_csv(
    os.path.join(tables_folder, 'supplementary_table1.csv')
)

# Supplementary figure 6

europe = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
europe = europe[europe.continent == 'Europe']

fig, axs = plt.subplots(1, 3, figsize=(15, 10))
vmin = 0
vmax = 120
color_resolution = 10
cmap = 'rainbow'
bounds = np.arange(vmin, vmax + color_resolution, color_resolution)
norm = BoundaryNorm(boundaries=bounds, ncolors=256, extend='neither')

cc = 2
cs = 2
data[(data.Cluster_climate == cc) & (data.Cluster_soil == cs) & (
    data.LC0_Desc_2015 == "Cropland")].plot("toc", norm=norm, cmap=cmap, ax=axs[0], markersize=5)
axs[0].set_title("Croplands")
data[(data.Cluster_climate == cc) & (data.Cluster_soil == cs) & (
    data.LC0_Desc_2015 == "Grassland")].plot("toc", norm=norm, cmap=cmap, ax=axs[1], markersize=5)
axs[1].set_title("Grasslands")
data[(data.Cluster_climate == cc) & (data.Cluster_soil == cs) & (
    data.LC0_Desc_2015 == "Woodland")].plot("toc", norm=norm, cmap=cmap, ax=axs[2], markersize=3)
axs[2].set_title("Woodlands")

for ax in axs:
    add_scalebar(ax, location='upper left')
    add_north_arrow(ax, x=0.9, headlength=15)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)

xlim = (min([ax.get_xlim()[0] for ax in axs]),
        max([ax.get_xlim()[1] for ax in axs]))
ylim = (min([ax.get_ylim()[0] for ax in axs]),
        max([ax.get_ylim()[1] for ax in axs]))

for i in range(3):
    ax = axs[i]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    europe[europe.name != "Russia"].geometry.boundary.plot(
        ax=ax, linewidth=1, facecolor='gray', alpha=0.1, color='black')

# Add colorbar
cax = fig.add_axes([0.98, 0.28, 0.02, 0.45])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm._A = []
cbar = fig.colorbar(sm, cax=cax)
cbar.ax.yaxis.set_ticks(bounds)
cbar.ax.yaxis.set_ticks_position('left')
cbar.ax.yaxis.set_label_position('left')
cbar.ax.set_ylabel("SOC [gC/kg]", fontsize=16)

plt.suptitle(f"Cluster {cc}-{cs}", y=0.95, fontsize=28)
plt.savefig(os.path.join(figures_folder, 'supplementaryfigure6.png'),
            format='png', dpi=600)
plt.close()


def bootstrap_reference_values(n_vmax, data, bootstrap_size, n_runs, references=["Grassland", "Woodland"], verbose=False):
    reference_values_batches = []

    for i in range(n_runs):
        reference_values, cluster_references_size = CarbonReferenceValuesComputer(
            n_vmax=n_vmax,
            data=data,
            references=references,
            verbose=verbose,
            bootstrap=True,
            bootstrap_size=bootstrap_size,
            bootstrap_random_state=i
        ).run()
        reference_values_batches.append(reference_values)

    reference_values_batches = pd.DataFrame(reference_values_batches)

    return reference_values_batches, cluster_references_size


def confidence_interval(x, confidence=0.9):
    ci = st.norm.interval(
        confidence=confidence,
        loc=np.mean(x),
        scale=st.sem(x)
    )
    return ci


def lower_limit_confidence_interval(x, confidence=0.9):
    return confidence_interval(x, confidence)[0]


def upper_limit_confidence_interval(x, confidence=0.9):
    return confidence_interval(x, confidence)[1]


reference_values_batches = {}
reference_values = {}
reference_values_upper = {}
reference_values_lower = {}
cluster_references_size = {}

for n_max in [40, 45, 50, 55]:
    reference_values_batches[n_max], cluster_references_size[n_max] = bootstrap_reference_values(
        n_vmax=n_max, data=data, bootstrap_size=0.85, n_runs=100, verbose=False)
    reference_values[n_max] = reference_values_batches[n_max].mean().to_dict()
    reference_values_upper[n_max] = reference_values_batches[n_max].apply(
        upper_limit_confidence_interval).to_dict()
    reference_values_lower[n_max] = reference_values_batches[n_max].apply(
        lower_limit_confidence_interval).to_dict()

# Supplementary Figure 3

fig, ax = plt.subplots(figsize=(15, 5))

colors = ['red', 'gold', 'dodgerblue', 'lightskyblue']
shifts = np.arange(-0.3, 0.5, 0.2)

for n_max, color, shift in zip([40, 45, 50, 55], colors, shifts):

    vmax_values = [reference_values[n_max][c] for c in reference_values[n_max]]
    vmax_uppervalues = [reference_values_upper[n_max][c] -
                        reference_values[n_max][c] for c in reference_values[n_max]]
    vmax_lowervalues = [reference_values[n_max][c] -
                        reference_values_lower[n_max][c] for c in reference_values[n_max]]

    ax.errorbar(x=range(len(vmax_values)) + shift, y=vmax_values,
                yerr=[vmax_lowervalues, vmax_uppervalues], fmt='o', c=color, label=n_max/100)

for i, c in enumerate(reference_values[40].keys()):
    ax.text(i, 2, cluster_references_size[40][c], fontdict={
            'horizontalalignment': 'center', 'backgroundcolor': 'white', 'color': 'gray'})

ax.set_xticks(range(len(vmax_values)))
ax.set_xticklabels(list(reference_values[40].keys()), rotation=90)
ax.set_xlim(-1, len(vmax_values))
ax.set_yticks(range(0, 65, 5))
ax.set_xlabel("cluster")
ax.set_ylabel("SOCref [g/kg]")
ax.set_title("Reference value per cluster")
ax.grid()
plt.legend(title='Percentile')
plt.tight_layout()
plt.savefig(os.path.join(figures_folder, 'supplementaryfigure3.png'),
            format='png', dpi=600)
plt.close()

# Use median as percentile

vmax = 50
reference_values_batches = reference_values_batches[vmax]
cluster_references_size = cluster_references_size[vmax]
reference_values = reference_values[vmax]
reference_values_upper = reference_values_upper[vmax]
reference_values_lower = reference_values_lower[vmax]

# Select only LUCAS 2015 croplands that were also
# croplands in LUCAS 2009 for application

data = data[(data.LC0_Desc_2015 == "Cropland") &
            (data.LC1_2009.str.startswith('B'))]

data = data[
    [
        "POINT_ID",
        "lat",
        "lon",
        "clay",
        "silt",
        "sand",
        "coarse",
        "CaCO3",
        "toc",
        "cluster",
        "Cluster_climate",
        "Cluster_soil",
        "geometry",
        "Country",
        "C/N"
    ]
]

data.loc[:, "vref"] = data.apply(lambda x: reference_values[x.cluster], axis=1)

# Save results
data.to_csv(
    os.path.join(
        output_folder,
        'natural_references_per_pedoclimate.csv'
    ),
    index=False
)
