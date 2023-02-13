from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = 'raise'


class CarbonReferenceValuesComputer:

    def __init__(
        self,
        n_vmax: int,
        data: pd.DataFrame,
        references=["Grassland", "Woodland"],
        verbose=True,
        bootstrap=False,
        bootstrap_size=None,
        bootstrap_random_state=None
    ):
        self.data = data
        self.references = references
        self.features_soil = ['clay', 'sand', 'CaCO3']
        self.features_climate = ['tmp_mean',
                                 'tmp_std', 'pre_sum', 'aridity_sum']
        self.n_vmax = n_vmax
        self.verbose = verbose
        self.bootstrap = bootstrap
        self.bootstrap_size = bootstrap_size
        self.bootstrap_random_state = bootstrap_random_state

    @staticmethod
    def _get_neighbor(point_id, k, neighborhoods, distances, pointids):
        index = pointids.index(point_id)
        neighbor_index = int(neighborhoods[index][k])
        distance = distances[index][k]
        neighbor = int(pointids[neighbor_index])
        return neighbor, distance

    def _fill_up_cluster(self, df):
        # missing GLs and WLs
        n_missing = 30 - len(df)
        # Candidate additional GL/WL: nearest neighbors of each GL/WL already in df.
        # Scan first neighbors of each existing GL/WL:
        # - if n_candidates > n_missing: choose the ones with lowest distance from reference
        # - else: repeat using second neighbors
        reference_data = self.data

        to_add = []
        k = 0
        while n_missing > 0:
            candidates = {}
            candidates_ = df.apply(
                lambda x: self._get_neighbor(
                    x["POINT_ID"],
                    k,
                    self.neighborhoods_references,
                    self.distances_references,
                    self.pointids_references,
                ),
                axis=1,
            ).values
            for id, dist in candidates_:
                if id in candidates:
                    candidates[id] = min(candidates[id], dist)
                else:
                    candidates[id] = dist
            candidates = [
                (id, candidates[id])
                for id in candidates
                if id not in to_add and id not in df["POINT_ID"].values
            ]
            candidates = np.array(sorted(candidates, key=lambda x: x[1]))
            if len(candidates) > n_missing:
                to_add += list(candidates[0:n_missing, 0])
                break
            else:
                if len(candidates) > 0:
                    to_add += list(candidates[:, 0])
                n_missing -= len(candidates)
                k += 1
        df = pd.concat(
            [df, reference_data[reference_data["POINT_ID"].isin(to_add)]])
        return df

    def fill_up_clusters(self):
        """Calculates KNN of GLs and WLs and "fills up" combined clusters that
        contain less than 30 GL/WLs with neighboring GL/WLs. This ensures that
        Vmax values are calculated among at least 30 GL/WLs.
        """
        reference_data = self.data
        features_climate = self.features_climate
        features_soil = self.features_soil
        features = features_climate + features_soil
        X = reference_data[reference_data["LC0_Desc_2015"].isin(self.references)][
            ["POINT_ID"] + features
        ]
        X = X.set_index("POINT_ID")
        self.pointids_references = list(X.index)

        # Scale
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        knn_references = NearestNeighbors(n_neighbors=29)
        knn_references.fit(X)
        (
            self.distances_references,
            self.neighborhoods_references,
        ) = knn_references.kneighbors()

        data_filled_up = {}
        counts_references = (
            reference_data[reference_data["LC0_Desc_2015"].isin(
                self.references)]
            .groupby(["Cluster_climate", "Cluster_soil"])
            .count()
            ["POINT_ID"]
        )
        clusters_to_fill = list(
            counts_references[counts_references < 30].index
        )
        for cc, cs in clusters_to_fill:
            df = reference_data[
                (reference_data.Cluster_climate == cc)
                & (reference_data.Cluster_soil == cs)
                & (reference_data["LC0_Desc_2015"].isin(self.references))
            ]
            data_filled_up[(cc, cs)] = self._fill_up_cluster(df)
            if self.verbose:
                print(f"Filling up cluster {cc}-{cs}:")
                print(data_filled_up[(cc, cs)].groupby(
                    ["Cluster_climate", "Cluster_soil"])["POINT_ID"].count())
        return data_filled_up

    def _get_reference_values_TOC_cluster(
        self, cluster_climate, cluster_soil, data_filled_up
    ):
        data = self.data
        n_vmax = self.n_vmax
        cc = cluster_climate
        cs = cluster_soil

        if (cc, cs) in data_filled_up:
            df = data_filled_up[(cc, cs)]

        else:
            df = data[
                (data.Cluster_climate == cc)
                & (data.Cluster_soil == cs)
                & (data["LC0_Desc_2015"].isin(self.references))
            ]

        len_df = len(df)
        if self.bootstrap:
            df = df.sample(frac=self.bootstrap_size,
                           random_state=self.bootstrap_random_state, replace=True)

        vmax = df["toc"].quantile(n_vmax / 100)

        return vmax, len_df

    def run(self):
        # get reference values per cluster
        reference_values = {}
        cluster_references_size = {}
        data = self.data
        cluster_labels_climate = sorted(data.Cluster_climate.unique())
        cluster_labels_soil = sorted(data.Cluster_soil.unique())
        data_filled_up = self.fill_up_clusters()
        for cc in cluster_labels_climate:
            for cs in cluster_labels_soil:
                reference_values[f"{cc}-{cs}"] = {}
                vmax, len_df = self._get_reference_values_TOC_cluster(
                    cc, cs, data_filled_up
                )
                reference_values[f"{cc}-{cs}"] = vmax
                cluster_references_size[f"{cc}-{cs}"] = len_df

        return reference_values, cluster_references_size
