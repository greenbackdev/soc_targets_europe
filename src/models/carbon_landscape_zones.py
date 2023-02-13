import pandas as pd
pd.options.mode.chained_assignment = 'raise'


class CarbonReferenceValuesComputer:

    def __init__(
        self,
        n_vmax: int,
        data: pd.DataFrame,
        bootstrap=False,
        bootstrap_size=None,
        bootstrap_random_state=None,
        variable='toc'
    ):
        self.data = data
        self.n_vmax = n_vmax
        self.bootstrap = bootstrap
        self.bootstrap_size = bootstrap_size
        self.bootstrap_random_state = bootstrap_random_state
        self.variable = variable

    def _get_reference_values_TOC_cluster(
        self, cluster, data
    ):
        data = self.data
        n_vmax = self.n_vmax

        df = data[(data.CLZs == cluster)]

        len_df = len(df)
        if self.bootstrap:
            df = df.sample(frac=self.bootstrap_size,
                           random_state=self.bootstrap_random_state, replace=True)

        vmax = df[self.variable].quantile(n_vmax / 100)

        return vmax, len_df

    def run(self):
        # get reference values per cluster
        reference_values = {}
        cluster_references_size = {}
        data = self.data
        cluster_labels = sorted(data.CLZs.unique())
        for c in cluster_labels:
            reference_values[c] = {}
            vmax, len_df = self._get_reference_values_TOC_cluster(c, data)
            reference_values[c] = vmax
            cluster_references_size[c] = len_df

        return reference_values, cluster_references_size
