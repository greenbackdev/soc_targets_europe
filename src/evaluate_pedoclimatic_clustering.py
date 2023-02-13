"""
Performs Pedoclimatic Clustering using Agglomerative Clustering and Gaussian
Mixture with 3 to 19 clusters. Plots the associated clustering metrics, maps
of the climate clusters, statistics of carbonates in the soil clusters and
texture triangles for the soil clusters.
"""

import os
from copy import copy

import geopandas as gpd
from sklearn import mixture
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

from importers.lucas_importer import LucasDataImporter
from models.pedoclimatic_clustering import PedoclimaticClustering


class PedoclimaticClusteringEvaluation:

    def __init__(
        self,
        clustering_method_climate,
        clustering_method_soil,
        kwargs_clustering_climate,
        kwargs_clustering_soil,
        n_clusters,
        figures_folder,
    ):
        self.clustering_method_climate = clustering_method_climate
        self.clustering_method_soil = clustering_method_soil
        self.kwargs_clustering_climate = kwargs_clustering_climate
        self.kwargs_clustering_soil = kwargs_clustering_soil
        self.n_clusters = n_clusters
        self.figures_folder = figures_folder

        os.makedirs(figures_folder, exist_ok=True)

    def _run_clustering(self):
        silhouette_scores_soil = []
        davies_bouldin_scores_soil = []
        silhouette_scores_climate = []
        davies_bouldin_scores_climate = []

        output_clustering = []

        for n in self.n_clusters:

            pedolcimatic_clustering = PedoclimaticClustering(
                data=data,
                clustering_method_climate=self.clustering_method_climate,
                clustering_method_soil=self.clustering_method_soil,
                n_clusters_climate=n,
                n_clusters_soil=n,
                kwargs_clustering_climate=self.kwargs_clustering_climate,
                kwargs_clustering_soil=self.kwargs_clustering_soil
            )
            pedolcimatic_clustering.run()
            output_clustering.append(copy(pedolcimatic_clustering.data))
            silhouette_scores_soil.append(
                pedolcimatic_clustering.silhouette_score_soil)
            davies_bouldin_scores_soil.append(
                pedolcimatic_clustering.davies_bouldin_score_soil)
            silhouette_scores_climate.append(
                pedolcimatic_clustering.silhouette_score_climate)
            davies_bouldin_scores_climate.append(
                pedolcimatic_clustering.davies_bouldin_score_climate)

        return (
            output_clustering,
            silhouette_scores_soil,
            davies_bouldin_scores_soil,
            silhouette_scores_climate,
            davies_bouldin_scores_climate
        )

    def _plot_metrics(
            self,
            silhouette_scores_soil,
            davies_bouldin_scores_soil,
            silhouette_scores_climate,
            davies_bouldin_scores_climate
    ):

        n_clusters = self.n_clusters
        figures_folder = self.figures_folder

        fig, axs = plt.subplots(2, 1, figsize=(8, 10))
        axs[0].plot(n_clusters, silhouette_scores_soil, 'o-')
        axs[1].plot(n_clusters, davies_bouldin_scores_soil, 'o-')
        axs[0].set_xticks(n_clusters)
        axs[0].set_xlabel('n_clusters')
        axs[0].set_ylabel('Silhouette score')
        axs[0].set_title('Soil - Silhouette Score')
        axs[0].grid()

        axs[1].set_xticks(n_clusters)
        axs[1].set_xlabel('n_clusters')
        axs[1].set_ylabel('Davies Bouldin score')
        axs[1].set_title('Soil - Davies Bouldin Score')
        axs[1].grid()
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                figures_folder,
                'clustering_evaluation_soil.pdf'
            )
        )
        plt.close()

        fig, axs = plt.subplots(2, 1, figsize=(8, 10))
        axs[0].plot(n_clusters, silhouette_scores_climate, 'o-')
        axs[1].plot(n_clusters, davies_bouldin_scores_climate, 'o-')
        axs[0].set_xticks(n_clusters)
        axs[0].set_xlabel('n_clusters')
        axs[0].set_ylabel('Silhouette score')
        axs[0].set_title('Climate - Silhouette Score')
        axs[0].grid()

        axs[1].set_xticks(n_clusters)
        axs[1].set_xlabel('n_clusters')
        axs[1].set_ylabel('Davies Bouldin score')
        axs[1].set_title('Climate - Davies Bouldin Score')
        axs[1].grid()
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                figures_folder,
                'clustering_evaluation_climate.pdf'
            )
        )
        plt.close()

        return

    def _plot_clusters_climate_maps(self, output_clustering):

        europe = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        europe = europe[europe.continent == 'Europe']

        figures_folder = os.path.join(
            self.figures_folder, 'climate_clusters_maps')
        os.makedirs(figures_folder, exist_ok=True)

        for df in output_clustering:
            cluster_labels_climate_ = sorted(df.Cluster_climate.unique())
            clusters_cmap_climate_ = plt.cm.get_cmap(
                'tab20', len(cluster_labels_climate_))

            fig, ax = plt.subplots(figsize=(10, 10))

            europe[europe.name != "Russia"].geometry.boundary.plot(
                ax=ax, linewidth=1, facecolor='gray', alpha=0.1, color='black')
            ax.set_xlim(-15, 40)
            ax.set_ylim(32, 75)

            df.plot('Cluster_climate', markersize=3, ax=ax,
                    legend=True, categorical=True, cmap=clusters_cmap_climate_)

            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    figures_folder,
                    f'{len(cluster_labels_climate_)}_climate_clusters.pdf'
                )
            )
            plt.close()

        return

    @staticmethod
    def _boxplot(x, y, df, palette=None, title=None, ax=None):
        grouped = df.loc[:, [x, y]].groupby([x]).median().sort_values(by=y)
        if palette:
            chart = sns.boxplot(x=x, y=y, data=df,
                                order=grouped.index, palette=palette, ax=ax)
        else:
            chart = sns.boxplot(x=x, y=y, data=df, order=grouped.index, ax=ax)
        chart.set(title=title)
        chart.set_xticklabels(chart.get_xticklabels(),
                              rotation=45, horizontalalignment='right')

    def _plot_clusters_soil_carbonates(self, output_clustering):
        figures_folder = os.path.join(
            self.figures_folder, 'soil_clusters_carbonates')
        os.makedirs(figures_folder, exist_ok=True)

        for df in output_clustering:
            cluster_labels_soil_ = sorted(df.Cluster_soil.unique())
            clusters_cmap_soil_ = plt.cm.get_cmap(
                'gist_rainbow', len(cluster_labels_soil_))
            cluster_colors_soil_ = {lab: clusters_cmap_soil_(
                i / len(cluster_labels_soil_)) for i, lab in enumerate(cluster_labels_soil_)}

            fig, ax = plt.subplots(figsize=(5, 5))
            self._boxplot(x="Cluster_soil", y='CaCO3', df=df,
                          palette=cluster_colors_soil_, ax=ax)
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    figures_folder,
                    f'{len(cluster_labels_soil_)}_soil_clusters.pdf'
                )
            )
            plt.close()

        return

    def _plot_clusters_soil_texture(self, output_clustering):
        figures_folder = os.path.join(
            self.figures_folder, 'soil_clusters_texture')
        os.makedirs(figures_folder, exist_ok=True)

        for df in output_clustering:
            cluster_labels_soil_ = sorted(df.Cluster_soil.unique())
            for c in cluster_labels_soil_:
                fig = px.scatter_ternary(df[df.Cluster_soil == c],
                                         a="clay", b="sand", c="silt",
                                         title="Cluster_soil {0:d}".format(c), width=500, height=500)
                fig.write_image(
                    os.path.join(
                        figures_folder,
                        f'{len(cluster_labels_soil_)}_soil_clusters-cluster_{c}.pdf'
                    )
                )
        return

    def run(self):
        (
            output_clustering,
            silhouette_scores_soil,
            davies_bouldin_scores_soil,
            silhouette_scores_climate,
            davies_bouldin_scores_climate
        ) = self._run_clustering()

        self._plot_metrics(
            silhouette_scores_soil,
            davies_bouldin_scores_soil,
            silhouette_scores_climate,
            davies_bouldin_scores_climate
        )

        self._plot_clusters_climate_maps(output_clustering)

        self._plot_clusters_soil_carbonates(output_clustering)

        self._plot_clusters_soil_texture(output_clustering)

        return (
            output_clustering,
            silhouette_scores_soil,
            davies_bouldin_scores_soil,
            silhouette_scores_climate,
            davies_bouldin_scores_climate
        )


if __name__ == "__main__":

    data_folder = os.path.join('data', 'lucas-esdac')
    climate_data_file = os.path.join('data',
                                     'lucas_climate_data',
                                     'lucas_climate_data.csv')
    importer = LucasDataImporter(data_folder,
                                 climate_data_file)
    data = importer.run()
    output_folder = 'output'
    figures_folder = os.path.join(
        output_folder,
        'pedoclimatic_clustering_evaluation'
    )
    os.makedirs(figures_folder, exist_ok=True)

    # ----- Agglomerative clustering -----
    ac_clustering_method_climate = AgglomerativeClustering
    ac_clustering_method_soil = AgglomerativeClustering
    ac_kwargs_clustering_climate = {"linkage": "ward"}
    ac_kwargs_clustering_soil = {"linkage": "ward"}
    ac_n_clusters = range(3, 20)
    ac_figures_folder = os.path.join(
        figures_folder, 'agglomerative_clustering')
    _ = PedoclimaticClusteringEvaluation(
        ac_clustering_method_climate,
        ac_clustering_method_soil,
        ac_kwargs_clustering_climate,
        ac_kwargs_clustering_soil,
        ac_n_clusters,
        ac_figures_folder
    ).run()

    # ----- Gaussian mixture -----
    gm_clustering_method_climate = mixture.GaussianMixture
    gm_clustering_method_soil = mixture.GaussianMixture
    gm_kwargs_clustering_climate = {"covariance_type": "full"}
    gm_kwargs_clustering_soil = {"covariance_type": "full"}
    gm_n_clusters = range(3, 20)
    gm_figures_folder = os.path.join(figures_folder, 'gaussian_mixture')
    _ = PedoclimaticClusteringEvaluation(
        gm_clustering_method_climate,
        gm_clustering_method_soil,
        gm_kwargs_clustering_climate,
        gm_kwargs_clustering_soil,
        gm_n_clusters,
        gm_figures_folder
    ).run()
