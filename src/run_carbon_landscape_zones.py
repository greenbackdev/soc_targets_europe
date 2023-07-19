"""
Calculates SOC reference values for each carbon landscape zone using
croplands as references. Selects only data that did not
change land cover between 2009 and 2015 and that are in the
features space of the data-driven reciprocal modelling.
"""

from maps_tools import add_scalebar, add_north_arrow
from models.carbon_landscape_zones import CarbonReferenceValuesComputer
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

data_path = os.path.join(
    'data', 'carbon_landscape_zones',
    'LUCAS2015_Croplands_CLZs.csv'
)

# Load data
data = pd.read_csv(data_path, low_memory=False)

# Merge with output from natural references per pedoclimate to get TOC
# (and filter only CLs that did not change LC)
data_nrppc = pd.read_csv(
    os.path.join(
        'output',
        'natural_references_per_pedoclimate.csv'
    ),
    low_memory=False
)
data_nrppc.drop("vref", axis=1, inplace=True)

data = data.merge(
    data_nrppc[['toc', 'geometry', 'POINT_ID']],
    on='POINT_ID'
)

# Select sites that are in the space feature of the
# data-driven reciprocal modeling
data_ddrm = pd.read_csv(
    os.path.join(
        'output',
        'data_driven_reciprocal_modelling.csv'
    ),
    low_memory=False
)
data_ddrm.drop("vref", axis=1, inplace=True)

data = data[data.POINT_ID.isin(data_ddrm.POINT_ID)]


data.loc[:, 'geometry'] = gpd.GeoSeries.from_wkt(data['geometry'])
data = gpd.GeoDataFrame(data, geometry='geometry')
data = data.set_crs("EPSG:4326")


def bootstrap_reference_values(
        n_vmax,
        data,
        bootstrap_size,
        n_runs,
        variable='toc'
):
    reference_values_batches = []
    for i in range(n_runs):
        reference_values, cluster_references_size = CarbonReferenceValuesComputer(
            n_vmax=n_vmax,
            data=data,
            bootstrap=True,
            bootstrap_size=bootstrap_size,
            bootstrap_random_state=i,
            variable=variable
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

for n_max in [80, 85, 90, 95]:
    reference_values_batches[n_max], cluster_references_size[n_max] = bootstrap_reference_values(
        n_vmax=n_max,
        data=data,
        bootstrap_size=0.85,
        n_runs=100
    )
    reference_values[n_max] = reference_values_batches[n_max].mean().to_dict()
    reference_values_upper[n_max] = reference_values_batches[n_max].apply(
        upper_limit_confidence_interval
    ).to_dict()
    reference_values_lower[n_max] = reference_values_batches[n_max].apply(
        lower_limit_confidence_interval
    ).to_dict()

# Supplementary Figure 4B

fig, ax = plt.subplots(figsize=(10, 5))

colors = ['red', 'gold', 'dodgerblue', 'lightskyblue']
shifts = np.arange(-0.3, 0.5, 0.2)

for n_max, color, shift in zip([80, 85, 90, 95], colors, shifts):

    vmax_values = [
        reference_values[n_max][c] for c in reference_values[n_max]
    ]
    vmax_uppervalues = [
        reference_values_upper[n_max][c] -
        reference_values[n_max][c] for c in reference_values[n_max]
    ]
    vmax_lowervalues = [
        reference_values[n_max][c] -
        reference_values_lower[n_max][c] for c in reference_values[n_max]
    ]

    ax.errorbar(
        x=range(len(vmax_values)) + shift, y=vmax_values,
        yerr=[vmax_lowervalues, vmax_uppervalues],
        fmt='o',
        c=color,
        label=n_max/100
    )

for i, c in enumerate(reference_values[80].keys()):
    ax.text(
        i,
        2,
        cluster_references_size[80][c],
        fontdict={
            'horizontalalignment': 'center',
            'backgroundcolor': 'white',
            'color': 'gray'
        }
    )

ax.set_xticks(range(len(vmax_values)))
ax.set_xticklabels(list(reference_values[80].keys()), rotation=90)
ax.set_xlim(-1, len(vmax_values))
ax.set_yticks(range(0, 65, 5))
ax.set_xlabel("Carbon Landscape Zone")
ax.set_ylabel("SOCref [g/kg]")
ax.set_title("Reference value per Carbon Landscape Zone")
ax.grid()
plt.legend(title='Percentile')
plt.tight_layout()
plt.savefig(os.path.join(figures_folder, 'supplementaryfigure4B.png'),
            format='png', dpi=600)
plt.close()

# Use 0.9 as percentile
data.loc[:, "vref"] = data.CLZs.apply(lambda x: reference_values[90][x])


clz_labels = sorted(data.CLZs.unique())
clusters_cmap_clz = plt.cm.get_cmap('tab20', len(clz_labels))
cluster_colors_clz = {lab: clusters_cmap_clz(
    (i) / len(clz_labels)) for i, lab in enumerate(clz_labels)}

# Figure 1B

europe = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
europe = europe[europe.continent == 'Europe']

cmap = 'rainbow'

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

europe[europe.name != "Russia"].geometry.boundary.plot(
    ax=ax, linewidth=1, facecolor='gray', alpha=0.1, color='black')
ax.set_xlim(-15, 40)
ax.set_ylim(32, 75)

data.plot(
    'CLZs',
    markersize=3,
    ax=ax,
    legend=True,
    legend_kwds={
        'fontsize': 20,
        'loc': 'lower right'
    },
    categorical=True,
    cmap=clusters_cmap_clz
)

add_scalebar(ax)
add_north_arrow(ax)
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)

plt.tight_layout()
plt.savefig(os.path.join(figures_folder, 'figure1B.png'),
            format='png', dpi=600)
plt.close()

# Supplementary Figure 2B

fig, ax = plt.subplots(figsize=(10, 10))

for clz in clz_labels:
    to_plot = data[data.CLZs == clz]
    if len(to_plot) > 29:
        sns.kdeplot(data=to_plot, x='toc', ax=ax,
                    color=cluster_colors_clz[clz], label=clz, marker='o')

ax.set_xlim(0, 120)
ax.set_xlabel("SOC", fontsize=28)
ax.set_ylabel("P(SOC)", fontsize=28)
ax.set_title('Carbon Landscape Zones', fontsize=28)
ax.legend(fontsize=16, ncol=2)

plt.savefig(os.path.join(figures_folder, 'supplementaryfigure2B.png'),
            format='png', dpi=600)
plt.close()

# Supplementary Figure 4C

europe = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
europe = europe[europe.continent == 'Europe']

cmap = 'rainbow_r'

fig, axs = plt.subplots(2, 3, figsize=(15, 10))

for ax, col in zip(axs.flatten(), [
    'CDI.PC1',
    'CDI.PC2',
    'CDI.PC3',
    'NPP.PC1',
    'NPP.PC2',
    'NPP.PC3'
]):

    europe[europe.name != "Russia"].geometry.boundary.plot(
        ax=ax, linewidth=1, facecolor='gray', alpha=0.1, color='black')
    ax.set_xlim(-15, 40)
    ax.set_ylim(32, 75)

    data.plot(col, markersize=3, ax=ax, legend=True, cmap=cmap)
    ax.set_title(col.replace('.', ' '))
    add_scalebar(ax, fontsize=10)
    add_north_arrow(
        ax,
        fontsize=10,
        x=0.25,
        y=0.1,
        arrow_length=0.07,
        width=1.5,
        headwidth=5,
        headlength=5
    )

plt.savefig(os.path.join(figures_folder, 'supplementaryfigure4C.png'),
            format='png', dpi=600)
plt.close()

# Save results
data.drop("geometry", axis=1, inplace=True)
data.to_csv(
    os.path.join(
        output_folder,
        'carbon_landscape_zones.csv'
    ),
    index=False
)
