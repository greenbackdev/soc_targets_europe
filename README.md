# Source code for "A new framework to estimate soil organic carbon content targets in European croplands."

This repository contains the code associated to the publication "A new framework to estimate soil organic carbon content targets in European croplands."

**Authors:** Lorenza Pacini, Pierre Arbelet, Songchao Chen, Aurélie Bacq-Labreuil, Christophe Calvaruso, Florian Schneider, Dominique Arrouays, Nicolas P.A. Saby, Lauric Cécillon, Pierre Barré.

Submitted to xxx in date xxx

## Requirements

Requires Python 3.9. For the packages requirements, see `Pipfile`.

```
pip install pipenv
````

```
pipenv install
```

## Necessary files to be included in `data/`

### LUCAS data

```
data/lucas-esdac
├── LUCAS TopSoil 2009
│   ├── LUCAS_Romania_Bulgaria_2012
│   │   ├── Bulgaria.csv
│   │   ├── Bulgaria.xlsx
│   │   ├── Romania.csv
│   │   └── Romania.xlsx
│   └── package-for-ESDAC-20190927
├── LUCAS TopSoil 2015
│   └──  LUCAS2015_topsoildata_20200323
└── LUCAS_Micro_data
    ├── EU_2009_20200213.CSV
    ├── EU_2012_20200213.CSV
    ├── EU_2015_20200225.CSV
    └── EU_2018_20200213.CSV
```

The data can be dowloaded from ESDAC and Eurostat at the following links.
 - LUCAS 2009/12 Topsoil: https://esdac.jrc.ec.europa.eu/content/lucas-2009-topsoil-data;
 - LUCAS 2015 Topsoil: https://esdac.jrc.ec.europa.eu/content/lucas2015-topsoil-data;
 - LUCAS 2009 land cover data: https://ec.europa.eu/eurostat/web/lucas/data/primary-data/2009;
 - LUCAS 2012 land cover data: https://ec.europa.eu/eurostat/web/lucas/data/primary-data/2012;
 - LUCAS 2015 land cover data: https://ec.europa.eu/eurostat/web/lucas/data/primary-data/2015.


 ### Climate data at LUCAS sites

```
data/lucas_climate_data
└── lucas_climate_data.csv
```

A csv file that contains climate, to be extracted at the theoretical LUCAS site locations, for example from the [Climate Research Unit](https://catalogue.ceda.ac.uk/uuid/89e1e34ec3554dc98594a5732622bce9) dataset. The list of the POINT_IDs that will be selected for treating the LUCAS dataset is in `selected_lucas_data_point_ids.csv`. It sufficient to extract climate data at these sites. The file should contain the following fields:

 - `POINT_ID` : LUCAS site identifier;
 - `tmp_mean`: mean annual temperature, averaged over 30 years (1988 to 2018);
 - `tmp_std`: standard devation of monthly average temperature over the year, averaged over 10 years (2008 to 2018);
 - `pre_sum`: total annual precipitation, averaged over 30 years (1988 to 2018);
 - `pet_sum`: annual potential evotranspiration, averaged over 10 years (2008 to 2018); 
 - `aridity_std`: `pet_sum` - `pre_sum`
 
### Data-driven reciprocal modeling

```
data/data_driven_reciprocal_modeling
└── cropToGrass_pred_content_no_CN_no_pH.csv
```

Output of the data-driven reciprocal modelling approach applied to LUCAS Topsoil 2015 data as described in 

Schneider, Florian, Christopher Poeplau, and Axel Don. ["Predicting ecosystem responses by data‐driven reciprocal modelling."](https://doi.org/10.1111/gcb.15817) Global Change Biology 27.21 (2021): 5670-5679.

but removing the pH and the C:N ratio from the predictors and predicting SOC contents instead of SOC stocks. The original code is available at https://doi.org/10.5281/zenodo.5171793.

### Carbon Landscape Zones

```
data/carbon_landscape_zones
└── LUCAS2015_Croplands_CLZs.csv
```

Output of the carbon-landscape-zones clustering approach applied to the selected LUCAS sites (croplands only). The approach is described in

Chen, Songchao, et al. ["National estimation of soil organic carbon storage potential for arable soils: A data-driven approach coupled with carbon-landscape zones."](https://doi.org/10.1016/j.scitotenv.2019.02.249) Science of the Total Environment 666 (2019): 355-367.

The following fields should be included:

 - For each month `x` in [1-12]:
    - `tmp_mean_month_x` = monthly mean temperature, average over 10 years (2005 to 2015).
    - `pre_sum_month_x` = monthly precipitation, average over 10 years (2005 to 2015).
    - `pet_sum_month_x` = monthly PET, average over 10 years (2005 to 2015).
 - `npp_mean_month_x` = mean daily NPP (extracted from 8-days data covering the month) multiplied by the number of days in the month. Average over 10 years (2005 to 2015).

 Climate data can be extracted from the [Climate Research Unit](https://catalogue.ceda.ac.uk/uuid/89e1e34ec3554dc98594a5732622bce9) dataset and NPP data can be extracted from [MODIS](https://developers.google.com/earth-engine/datasets/catalog/MODIS_006_MOD17A2H#description).


### MaOM capacity

```
data/maom_capacity
├── global_MOC.nc
└── global_MOCmax.nc
```

Data provided by the authors of

Georgiou, Katerina, et al. ["Global stocks and capacity of mineral-associated soil organic carbon."](https://doi.org/10.1038/s41467-022-31540-9) Nature communications 13.1 (2022): 3797.

The original data are available at https://doi.org/10.5281/zenodo.6539765.

### Land use data by NUTS 2 regions

```
data/Main farm land use by NUTS 2 regions
├── ef_lus_main__custom_3779332_page_spreadsheet.xlsx
└── extract.csv
```

The original data `ef_lus_main__custom_3779332_page_spreadsheet.xlsx` can be downloaded from [Eurostat](https://ec.europa.eu/eurostat/databrowser/view/EF_LUS_MAIN__custom_3779332/default/table?lang=en). The file `extract.csv` is an extract of the data that contains three columns: `Country`, `Arable land`, `Permanent crops` (see `Sheet 1` in `ef_lus_main__custom_3779332_page_spreadsheet.xlsx`).


## Scripts

Run scripts in the correct order by using
```
pipenv run python src/evaluate_pedoclimatic_clustering.py

pipenv run python src/run_pedoclimatic_clustering.py
pipenv run python src/run_natural_references_per_pedoclimate.py
pipenv run python src/run_data_driven_reciprocal_modelling.py
pipenv run python src/run_carbon_landscape_zones.py
pipenv run python src/run_maom_capacity.py

pipenv run python src/run_methods_comparison.py
```

Scripts description:
- `evaluate_pedoclimatic_clustering.py`: performs Pedoclimatic Clustering using Agglomerative Clustering and Gaussian Mixtures with 3 to 19 clusters. Plots:
    - clustering metrics;
    - maps of the climate clusters;
    - statistics of carbonates in the soil clusters;
    - texture triangles for the soil clusters.

- `run_pedoclimatic_clustering.py`: Performs Pedoclimatic Clustering of LUCAS sites using
    - Agglomerative Clustering for soil clustering (4 clusters);
    - Gaussian Mixture for climate clustering (11 clusters).

- `run_natural_references_per_pedoclimate.py`: NRpPc - Calculates SOC reference values for each pedoclimatic cluster (output of run_pedoclimatic_clustering.py) using natural references (grasslands and woodlands). Selects only data that did not change land cover between 2009 and 2015.

- `run_data_driven_reciprocal_modelling.py`: DDRM - Gets SOC reference values from the Data-driven reciprocal modelling output.

- `run_carbon_landscape_zones.py`: CLZs - Calculates SOC reference values for each carbon landscape zone using croplands as references. Selects only data that did not change land cover between 2009 and 2015 and that are in the features space of the data-driven reciprocal modelling.

- `run_maom_capacity.py`: Gets SOC reference values from the MaOM capacity output.

- `run_methods_comparison`: compares the four approaches, computes the ensemble modelling (median of the results of NRpPc, DDRM, CLZs), produces plots and maps.