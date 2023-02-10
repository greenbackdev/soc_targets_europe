from importers.lucas_importer import LucasDataImporter
import os


if __name__ == "__main__":

    data_folder = os.path.join('data', 'lucas-esdac')
    output_folder = 'output'
    os.makedirs(output_folder, exist_ok=True)
    climate_data_file = os.path.join('data',
                                     'lucas_climate_data',
                                     'lucas_climate_data.csv')
    importer = LucasDataImporter(data_folder,
                                 climate_data_file,
                                 output_folder)
    data = importer.run()
