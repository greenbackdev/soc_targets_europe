"""
Performs Pedoclimatic Clustering of LUCAS sites using
- Agglomerative Clustering for soil clustering (4 clusters)
- Gaussian Mixture for climate clustering (11 clusters).
Saves results to output/pedoclimatic_clustering.
"""

import os
import pickle
import geopandas as gpd
from sklearn import mixture
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

from importers.lucas_importer import LucasDataImporter
from models.pedoclimatic_clustering import PedoclimaticClustering

lucas_data_folder = os.path.join('data', 'lucas-esdac')
climate_data_file = os.path.join('data',
                                 'lucas_climate_data',
                                 'lucas_climate_data.csv')
importer = LucasDataImporter(lucas_data_folder,
                             climate_data_file)
lucas_data = importer.run()

output_folder = 'output'
os.makedirs(output_folder, exist_ok=True)
figures_folder = os.path.join(output_folder, 'figures')
os.makedirs(figures_folder, exist_ok=True)
tables_folder = os.path.join(output_folder, 'tables')
os.makedirs(tables_folder, exist_ok=True)

clustering_method_climate = mixture.GaussianMixture
clustering_method_soil = AgglomerativeClustering
kwargs_clustering_climate = {"covariance_type": "full"}
kwargs_clustering_soil = {"linkage": "ward"}
n_clusters_climate = 11
n_clusters_soil = 4

# Perform pedoclimatic clustering
pedoclimatic_clustering = PedoclimaticClustering(data=lucas_data,
                                                 clustering_method_climate=clustering_method_climate,
                                                 clustering_method_soil=clustering_method_soil,
                                                 n_clusters_climate=n_clusters_climate,
                                                 n_clusters_soil=n_clusters_soil,
                                                 kwargs_clustering_climate=kwargs_clustering_climate,
                                                 kwargs_clustering_soil=kwargs_clustering_soil)
pedoclimatic_clustering.run()
data = pedoclimatic_clustering.data

scaler_climate = pedoclimatic_clustering.scaler_climate
scaler_soil = pedoclimatic_clustering.scaler_soil
X_climate = pedoclimatic_clustering.X_climate
X_soil = pedoclimatic_clustering.X_soil
model_climate = pedoclimatic_clustering.model_climate
model_soil = pedoclimatic_clustering.model_soil
knn_soil = pedoclimatic_clustering.knn_soil

# Save results
data.to_csv(os.path.join(output_folder,
            'pedoclimatic_clustering.csv'),
            index=False)
output_path_models = os.path.join(
    output_folder,
    "pedoclimatic_clustering_models"
)
os.makedirs(output_path_models, exist_ok=True)
with open(
    os.path.join(output_path_models, "scaler_climate.p"),
    "wb"
) as f:
    pickle.dump(scaler_climate, f)
with open(os.path.join(output_path_models, "scaler_soil.p"),
          "wb"
          ) as f:
    pickle.dump(scaler_soil, f)
with open(
    os.path.join(output_path_models, "model_climate.p"),
    "wb"
) as f:
    pickle.dump(model_climate, f)
with open(
    os.path.join(output_path_models, "model_soil.p"),
    "wb",
) as f:
    pickle.dump(model_soil, f)
with open(
    os.path.join(output_path_models, "knn_soil.p"),
    "wb"
) as f:
    pickle.dump(knn_soil, f)

# Plots
cluster_labels_climate = sorted(data.Cluster_climate.unique())
cluster_labels_soil = sorted(data.Cluster_soil.unique())

clusters_cmap_climate = plt.cm.get_cmap('tab20', len(cluster_labels_climate))
cluster_colors_climate = {lab: clusters_cmap_climate(
    i / len(cluster_labels_climate)) for i, lab in enumerate(cluster_labels_climate)}

cluster_marker_soil = {0: 'o', 1: 'v', 2: 's', 3: 'X'}
clusters_cmap_soil = plt.cm.get_cmap('tab20', len(cluster_labels_soil))
cluster_colors_soil = {lab: clusters_cmap_soil(
    i / len(cluster_labels_soil)) for i, lab in enumerate(cluster_labels_soil)}

# Figure 1A

europe = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
europe = europe[europe.continent == 'Europe']

fig, axs = plt.subplots(1, 2, figsize=(20, 10))

for ax in axs:
    europe[europe.name != "Russia"].geometry.boundary.plot(
        ax=ax, linewidth=1, facecolor='gray', alpha=0.1, color='black')
    ax.set_xlim(-15, 40)
    ax.set_ylim(32, 75)

data.plot('Cluster_climate', markersize=3,
          ax=axs[0], legend=True, categorical=True, cmap=clusters_cmap_climate)
axs[0].set_title("Cluster_climate")

data.plot('Cluster_soil', markersize=3,
          ax=axs[1], legend=True, categorical=True, cmap=clusters_cmap_soil)
axs[1].set_title("Cluster_soil")

plt.savefig(os.path.join(figures_folder, 'figure1A.pdf'))
plt.close()

# figure 2

for c in cluster_labels_soil:
    fig = px.scatter_ternary(data[data.Cluster_soil == c],
                             a="clay", b="sand", c="silt", opacity=0.2,
                             title="Cluster soil {0:d}".format(c), width=500, height=500)
    fig.write_image(
        os.path.join(
            figures_folder,
            f'figure2_{c}.pdf'
        )
    )


# Table 1

data.groupby("Cluster_soil").CaCO3.describe().to_csv(
    os.path.join(tables_folder, 'table1.csv')
)

# Supplementary Figure 2A

fig, axs = plt.subplots(1, 3, figsize=(30, 10))

for ax, lc in zip(axs, ['Cropland', 'Grassland', 'Woodland']):
    for cc in cluster_labels_climate:
        for cs in cluster_labels_soil:
            to_plot = data[(data.Cluster_climate == cc) & (
                data.Cluster_soil == cs) & (data.LC0_Desc_2015 == lc)]
            if len(to_plot) > 29:
                sns.kdeplot(data=to_plot, x='toc', ax=ax,
                            color=cluster_colors_climate[cc], marker=cluster_marker_soil[cs], label="{0}-{1}".format(cc, cs))

    ax.set_xlim(0, 120)
    ax.set_xlabel("SOC [gC/kg]", fontsize=18)
    ax.set_ylabel("P(SOC)", fontsize=18)
    ax.set_title(lc+'s', fontsize=18)
    ax.legend(fontsize=16, ncol=2)

plt.savefig(os.path.join(figures_folder, 'supplementaryfigure2A.pdf'))
plt.close()
