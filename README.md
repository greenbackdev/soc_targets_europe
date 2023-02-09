# Souce code for "A new framework to estimate soil organic carbon content targets in European croplands."

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

## Scripts

Run scripts using
```
pipenv run python <script.py>
```

Avaiable scripts:
- `evaluate_pedoclimatic_clustering.py`: performs Pedoclimatic Clustering using Agglomerative Clustering and Gaussian Mixtures with 3 to 19 clusters. Plots:
    - clustering metrics;
    - maps of the climate clusters;
    - statistics of carbonates in the soil clusters;
    - texture triangles for the soil clusters.

- `run_pedoclimatic_clustering.py`: #TODO

- `run_natural_references_per_pedoclimate.py`: #TODO

- `run_carbon_landscape_zones.py` #TODO

- `run_data_driven_reciprocal_modelling.py` #TODO

- `run_maom_capacity.py` #TODO

## Necessary data

#TODO

