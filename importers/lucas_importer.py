import os

import numpy as np
import pandas as pd
import geopandas as gpd


class LucasDataImporter:

    def __init__(self, lucas_data_folder, climate_data_file, output_folder):
        self.lucas_data_folder = lucas_data_folder
        self.climate_data_file = climate_data_file
        self.output_folder = output_folder
        self.features_soil = [
            "clay_2009",
            "sand_2009",
            "CaCO3_2015"
        ]
        self.features_climate = [
            "tmp_mean",
            "tmp_std",
            "pre_sum",
            "aridity_sum",
        ]

    def _get_lucas_data(self):
        # Load LUCAS 2009 data
        lucas2009_metadata = pd.read_csv(
            os.path.join(self.lucas_data_folder,
                         "LUCAS_Micro_data/EU_2009_20200213.CSV")
        )
        lucas2009_gdf = gpd.read_file(
            os.path.join(
                self.lucas_data_folder, "LUCAS TopSoil 2009/package-for-ESDAC-20190927/SoilAttr_LUCAS2009/SoilAttr_LUCAS2009.shp")
        )
        lucas2009_gdf["has_soil"] = True
        lucas2009_metadata["has_metadata"] = True
        lucas2009_gdf = pd.merge(
            lucas2009_gdf, lucas2009_metadata, on="POINT_ID", how="inner"
        )
        lucas2009_gdf = gpd.GeoDataFrame(lucas2009_gdf)

        # Add Bulgaria and Romania, Malta and Cyprus (2012)
        lucas2012_metadata = pd.read_csv(
            os.path.join(self.lucas_data_folder,
                         "LUCAS_Micro_data/EU_2012_20200213.CSV")
        )
        lucas2012_metadata = lucas2012_metadata.drop(
            lucas2012_metadata[lucas2012_metadata.POINT_ID == '270.272 rows selected.'].index)
        lucas2012_metadata['POINT_ID'] = lucas2012_metadata['POINT_ID'].astype(
            float)
        lucas2012_metadata["has_metadata"] = True
        lucas2012_BG_RO_gdf = gpd.read_file(
            os.path.join(
                self.lucas_data_folder, "LUCAS TopSoil 2009/package-for-ESDAC-20190927/SoilAttr_BG_RO/SoilAttr_LUCAS_2012_BG_RO.shp")
        )
        lucas2012_CYP_MLT_gdf = gpd.read_file(
            os.path.join(
                self.lucas_data_folder, "LUCAS TopSoil 2009/package-for-ESDAC-20190927/SoilAttr_CYP_MLT/SoilAttr_LUCAS_2009_CYP_MLT.shp")
        )
        lucas2012_gdf = pd.concat(
            [lucas2012_BG_RO_gdf, lucas2012_CYP_MLT_gdf], ignore_index=True
        )
        lucas2012_gdf["has_soil"] = True
        lucas2012_BG = gpd.read_file(
            os.path.join(
                self.lucas_data_folder, "LUCAS TopSoil 2009/LUCAS_Romania_Bulgaria_2012/Bulgaria.csv")
        )
        lucas2012_RO = gpd.read_file(
            os.path.join(
                self.lucas_data_folder, "LUCAS TopSoil 2009/LUCAS_Romania_Bulgaria_2012/Romania.csv")
        )
        lucas2012_BG["sample_ID"] = lucas2012_BG["SoilID"]
        lucas2012_RO["sample_ID"] = lucas2012_RO["SoilID"]
        lucas2012_BG.POINT_ID = lucas2012_BG.POINT_ID.astype(int)
        lucas2012_RO["POINT_ID"] = lucas2012_RO.POINTID.astype(int)
        lucas2012_gdf = lucas2012_gdf.merge(
            lucas2012_BG[["POINT_ID", "sample_ID"]], on="POINT_ID", how="left"
        )
        lucas2012_gdf = lucas2012_gdf.merge(
            lucas2012_RO[["POINT_ID", "sample_ID"]], on="POINT_ID", how="left"
        )
        lucas2012_gdf = pd.merge(
            lucas2012_gdf, lucas2012_metadata, on="POINT_ID", how="inner"
        )

        # Merge 2012 into 2009
        lucas2009_gdf = pd.concat(
            [lucas2009_gdf, lucas2012_gdf], ignore_index=True
        )

        # Load LUCAS 2015 data
        lucas2015_metadata = pd.read_csv(
            os.path.join(self.lucas_data_folder,
                         "LUCAS_Micro_data/EU_2015_20200225.CSV")
        )
        lucas2015_metadata = lucas2015_metadata[
            ~lucas2015_metadata.POINT_ID.isna()
        ]
        lucas2015_metadata.drop(
            lucas2015_metadata[
                lucas2015_metadata.POINT_ID == "339.696 rows selected."
            ].index,
            inplace=True,
        )
        lucas2015_metadata.POINT_ID = lucas2015_metadata.POINT_ID.astype(
            np.int64
        )
        lucas2015_gdf = gpd.read_file(
            os.path.join(
                self.lucas_data_folder, "LUCAS TopSoil 2015/LUCAS2015_topsoildata_20200323/LUCAS_Topsoil_2015_20200323-shapefile/LUCAS_Topsoil_2015_20200323.shp")
        )
        lucas2015_gdf["has_soil"] = True
        lucas2015_metadata["has_metadata"] = True
        lucas2015_gdf = gpd.GeoDataFrame(lucas2015_gdf)
        lucas2015_gdf = pd.merge(
            lucas2015_gdf,
            lucas2015_metadata,
            left_on="Point_ID",
            right_on="POINT_ID",
            how="inner",
        )

        # Use consistent nomenclature for texture and pH between 2009 and 2015
        lucas2015_gdf["clay"] = lucas2015_gdf["Clay"]
        lucas2015_gdf["silt"] = lucas2015_gdf["Silt"]
        lucas2015_gdf["sand"] = lucas2015_gdf["Sand"]
        lucas2015_gdf["pHinH2O"] = lucas2015_gdf["pH_H20"]
        lucas2015_gdf.drop(["Clay", "Silt", "Sand"], axis=1, inplace=True)

        # Merge 2015 and 2009
        lucas_gdf = pd.merge(
            lucas2009_gdf,
            lucas2015_gdf,
            on="POINT_ID",
            how="outer",
            suffixes=("_2009", "_2015"),
        )
        lucas_gdf["geometry"] = lucas_gdf.apply(
            lambda r: r["geometry_2015"]
            if r["geometry_2015"]
            else r["geometry_2009"],
            axis=1,
        )
        lucas_gdf = gpd.GeoDataFrame(lucas_gdf)

        # for consistency of suffixes
        lucas_gdf["Country_2009"] = lucas_gdf["Country"]
        lucas_gdf["LC0_Desc_2015"] = lucas_gdf["LC0_Desc"]

        return lucas_gdf

    def preprocess_lucas_data(self):
        data = self.data
        data.POINT_ID = data.POINT_ID.astype(int)

        # drop two sites that share the same POINT_ID
        data = data.drop(
            index=data[
                data.POINT_ID
                == data[data.POINT_ID.duplicated()].POINT_ID.values[0]
            ].index
        )

        # custom columns ans NaNs
        data["OC_2009"].replace("<6", np.nan, inplace=True)
        data["CaCO3_2009"].replace("<0.5", np.nan, inplace=True)
        data.OC_2009 = data.OC_2009.astype(float)
        data.CaCO3_2009 = data.CaCO3_2009.astype(float)
        data["OC_2009"].replace(np.float64(-999), np.nan, inplace=True)
        data["CaCO3_2009"].replace(np.float64(-999), np.nan, inplace=True)
        data["clay_2009"].replace(np.float64(-999), np.nan, inplace=True)
        data["silt_2009"].replace(np.float64(-999), np.nan, inplace=True)
        data["sand_2009"].replace(np.float64(-999), np.nan, inplace=True)
        data["pHinH2O_2009"].replace(np.float64(-999), np.nan, inplace=True)
        data[data.filter(regex="GPS_LAT_.*").columns].replace(
            88.888888, np.nan, inplace=True
        )
        data[data.filter(regex="GPS_LONG_.*").columns].replace(
            88.888888, np.nan, inplace=True
        )
        data[data.filter(regex="OBS_DIST_.*").columns].replace(
            8888, np.nan, inplace=True
        )
        data[data.filter(regex="OBS_DIST_.*").columns].replace(
            -1, np.nan, inplace=True
        )
        data[data.filter(regex="GPS_PREC_.*").columns].replace(
            8888, np.nan, inplace=True
        )
        data.Country_2009 = data.Country_2009.str.upper()

        # get lat and lon from geometry
        data["lon"] = data["geometry"].x
        data["lat"] = data["geometry"].y
        return data

    def select_lucas_data(self):
        data = self.data

        # step 1: texture consistency between 2009 and 2015
        data = data.drop(
            data[
                (
                    data.has_soil_2009
                    & data.has_soil_2015
                    & ~(data.clay_2015.isna())
                    & ~(data.silt_2015.isna())
                    & ~(data.sand_2015.isna())
                    & (
                        (data.clay_2015 != data.clay_2009)
                        | (data.silt_2015 != data.silt_2009)
                        | (data.sand_2015 != data.sand_2009)
                    )
                )
            ].index
        )

        # step 2: filter mineral soils (OC < 120 g/kg)
        data = data.drop(
            data[(data.OC_2009 > 120) | (data.OC_2015 > 120)].index
        )

        # step 3: select sites that are Croplands, Grasslands or Woodlands in LUCAS 2015
        selected_land_covers = ["Cropland", "Woodland", "Grassland"]
        data = data[
            (
                data.LC0_Desc_2015.isin(selected_land_covers)
            )
        ]

        return data

    def _add_climate_data(self, data):

        climate_data = pd.read_csv(os.path.join(self.climate_data_file))
        data = data.merge(climate_data, on='POINT_ID')

        return data

    def _select_clustering_data(self, data):
        features = self.features_climate + self.features_soil

        # remove points with nan values in contextualization features
        data = data.dropna(subset=features)

        return data

    def _save_data(self, data):
        data.to_csv(os.path.join(self.output_folder,
                    '_selected_lucas_data.csv'),
                    index=False)

    def run(self):
        data = self._get_lucas_data()
        data = self._add_climate_data(data)
        data = self._select_clustering_data(data)
        self._save_data(data)

        return data
