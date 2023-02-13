from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import mixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import os

import pandas as pd
pd.options.mode.chained_assignment = 'raise'


class PedoclimaticClustering:

    def __init__(
        self,
        data,
        clustering_method_climate=mixture.GaussianMixture,
        clustering_method_soil=AgglomerativeClustering,
        n_clusters_climate=11,
        n_clusters_soil=4,
        kwargs_clustering_climate={"covariance_type": "full"},
        kwargs_clustering_soil={"linkage": "ward"},
    ):
        """
        :param str data_path: path to the data folder
        :param class clustering_method_climate: clustering class for climate. In the current implementation, it assumes that it has a .predict() method. Default is mixture.GaussianMixture from sklearn.
        :param class clustering_method_climate: clustering class for soil. In the current implementation, it assumes that it dos not have a .predict() method. Default is AgglomerativeClustering from sklearn.cluster.
        :param int n_clusters_climate: number of climate clusters. Default is 11.
        :param int n_clusters_soil: number of soil clusters. Default is 4.
        :param dict kwargs_clustering_climate: arguments to initialize climate clustering instance. Default is {'covariance_type': 'full'}.
        :param dict kwargs_clustering_soil: arguments to initialize soil clustering instance. Default is {'linkage': 'ward'}.

        """
        self.data = data
        self.features_climate = [
            "tmp_mean",
            "tmp_std",
            "pre_sum",
            "aridity_sum",
        ]
        self.features_soil = ["clay", "sand", "CaCO3"]
        self.clustering_method_climate = clustering_method_climate
        self.clustering_method_soil = clustering_method_soil
        self.n_clusters_climate = n_clusters_climate
        self.n_clusters_soil = n_clusters_soil
        self.kwargs_clustering_climate = kwargs_clustering_climate
        self.kwargs_clustering_soil = kwargs_clustering_soil

    def scale_data(self):
        scaler_soil = StandardScaler()
        scaler_climate = StandardScaler()
        data = self.data
        scaler_climate = scaler_climate.fit(
            data[self.features_climate].values
        )
        scaler_soil = scaler_soil.fit(
            data[self.features_soil].values
        )

        X_climate = scaler_climate.transform(
            data[self.features_climate].values
        )
        X_soil = scaler_soil.transform(
            data[self.features_soil].values
        )
        return scaler_climate, scaler_soil, X_climate, X_soil

    @staticmethod
    def _train_clustering_model(X, method, n_components, kwargs):
        try:
            model = method(n_components=n_components, random_state=1, **kwargs)
        except TypeError:
            model = method(
                n_clusters=n_components, **kwargs
            )  # for Agglomerative Clustering
        model = model.fit(X)
        return model

    def train_clustering(self):

        model_climate = self._train_clustering_model(
            X=self.X_climate,
            method=self.clustering_method_climate,
            n_components=self.n_clusters_climate,
            kwargs=self.kwargs_clustering_climate,
        )
        model_soil = self._train_clustering_model(
            X=self.X_soil,
            method=self.clustering_method_soil,
            n_components=self.n_clusters_soil,
            kwargs=self.kwargs_clustering_soil,
        )

        # Train a KNN model then predicts the soil clusters of new data
        # (necessary because Agglomerative Clustering does not have a .predict() method)
        if not hasattr(model_soil, "predict"):
            clusters_soil = model_soil.labels_
            knn_soil = KNeighborsClassifier(n_neighbors=1)
            knn_soil.fit(self.X_soil, clusters_soil)

        else:
            knn_soil = None

        self.knn_soil = knn_soil

        return model_climate, model_soil, knn_soil

    def apply_clustering(self):

        data = self.data
        if not hasattr(self.model_climate, "predict"):
            clusters_climate = self.model_climate.labels_
            data.loc[:, "Cluster_climate"] = clusters_climate
        else:
            clusters_climate = self.model_climate.predict(self.X_climate)
            data.loc[:, "Cluster_climate"] = clusters_climate

        if not hasattr(self.model_soil, "predict"):
            clusters_soil = self.model_soil.labels_
            data.loc[:, "Cluster_soil"] = clusters_soil
        else:
            clusters_soil = self.model_soil.predict(self.X_soil)
            data.loc[:, "Cluster_soil"] = clusters_soil

        data.loc[:, "cluster"] = data.apply(
            lambda x: "-".join([str(x.Cluster_climate), str(x.Cluster_soil)]),
            axis=1
        )

        return data

    def evaluate_clustering(self):
        if hasattr(self.model_climate, "predict"):  # GaussianMixture
            bic_score_climate = self.model_climate.bic(self.X_climate)
            silhouette_score_climate = None
            davies_bouldin_score_climate = None
        else:  # Agglomerative Clustering
            silhouette_score_climate = silhouette_score(
                self.X_climate, self.model_climate.labels_, metric='euclidean')
            davies_bouldin_score_climate = davies_bouldin_score(
                self.X_climate, self.model_climate.labels_)
            bic_score_climate = None

        if hasattr(self.model_soil, "predict"):  # GaussianMixture
            bic_score_soil = self.model_soil.bic(self.X_soil)
            silhouette_score_soil = None
            davies_bouldin_score_soil = None
        else:  # Agglomerative Clustering
            silhouette_score_soil = silhouette_score(
                self.X_soil, self.model_soil.labels_, metric='euclidean')
            davies_bouldin_score_soil = davies_bouldin_score(
                self.X_soil, self.model_soil.labels_)
            bic_score_soil = None

        return silhouette_score_climate, davies_bouldin_score_climate, bic_score_climate, silhouette_score_soil, davies_bouldin_score_soil, bic_score_soil

    def run(self):

        self.scaler_climate, self.scaler_soil, self.X_climate, self.X_soil = self.scale_data()
        self.model_climate, self.model_soil, self.knn_soil = self.train_clustering()
        self.data = self.apply_clustering()
        (self.silhouette_score_climate,
         self.davies_bouldin_score_climate,
         self.bic_score_climate,
         self.silhouette_score_soil,
         self.davies_bouldin_score_soil,
         self.bic_score_soil) = self.evaluate_clustering()
