PIP = pip
PYTHON = python
PIPENV_RUN = pipenv run $(PYTHON)

.PHONY: help setup

help:
	@echo "---------------HELP-----------------"
	@echo "To setup the project type:"
	@echo "\tmake setup"
	@echo "--Installs required libraries in a pipenv environment"
	@echo ""
	@echo "To run the pedoclimatic clustering evalution type:"
	@echo "\tmake evaluate_pedoclimatic_clustering"
	@echo "Performs Pedoclimatic Clustering using Agglomerative Clustering and"
	@echo "Gaussian Mixtures with 3 to 19 clusters. Plots:"
	@echo "\t- clustering metrics;"
	@echo "\t- maps of the climate clusters;"
	@echo "\t- statistics of carbonates in the soil clusters;"
	@echo "\t- texture triangles for the soil clusters."
	@echo ""
	@echo "To compute the pedoclimatic clusters type:"
	@echo "\tmake compute_pedoclimatic_clusters"
	@echo "Performs Pedoclimatic Clustering of LUCAS sites using"
	@echo "\t- Agglomerative Clustering for soil clustering (4 clusters);"
	@echo "\t- Gaussian Mixture for climate clustering (11 clusters)."
	@echo ""
	@echo "To compute the Natural References per Pedoclimate reference values type:"
	@echo "\tmake compute_natural_references_per_pedoclimate"
	@echo "NRpPc - Calculates SOC reference values for each pedoclimatic cluster"
	@echo "using natural references (grasslands and woodlands)."
	@echo "Selects only data that did not change land cover between 2009 and 2015."
	@echo ""
	@echo "To compute the Data-Driven Reciprocal Modelling reference values type:"
	@echo "\tmake compute_data_driven_reciprocal_modelling"
	@echo "DDRM - Gets SOC reference values from the Data-driven reciprocal modelling output."
	@echo ""
	@echo "To compute the Carbon Landscape Zones reference values type:"
	@echo "\tmake compute_carbon_landscape_zones"
	@echo "CLZs - Calculates SOC reference values for each carbon landscape zone using croplands as references."
	@echo "Selects only data that did not change land cover between 2009 and 2015"
	@echo "and that are in the features space of the data-driven reciprocal modelling."
	@echo ""
	@echo "To compute the MaOM capacity reference values type:"
	@echo "\tmake compute_maom_capacity"
	@echo "Gets SOC reference values from the MaOM capacity output."
	@echo ""
	@echo "To compute the refrence values using the four methods"
	@echo "(NRpPc, DDRM, CLZs and MaOM capacity),"
	@echo "compute the ensemble modelling (median of NRpPc, DDRM and CLZs)"
	@echo "and produce the corresponding maps, type:"
	@echo "\tmake run_ensemble"
	@echo "------------------------------------"

setup: Pipfile
	$(PIP) install pipenv
	pipenv install

evaluate_pedoclimatic_clustering:
	$(PIPENV_RUN) src/evaluate_pedoclimatic_clustering.py

compute_pedoclimatic_clusters:
	$(PIPENV_RUN) src/run_pedoclimatic_clustering.py

compute_natural_references_per_pedoclimate:
	$(PIPENV_RUN) src/run_natural_references_per_pedoclimate.py

compute_data_driven_reciprocal_modelling:
	$(PIPENV_RUN) src/run_data_driven_reciprocal_modelling.py

compute_carbon_landscape_zones:
	$(PIPENV_RUN) src/run_carbon_landscape_zones.py

compute_maom_capacity:
	$(PIPENV_RUN) src/run_maom_capacity.py

run_ensemble: compute_natural_references_per_pedoclimate compute_data_driven_reciprocal_modelling compute_carbon_landscape_zones compute_maom_capacity
	$(PIPENV_RUN) src/run_ensemble.py
