import numpy as np
import os
from netCDF4 import Dataset
import requests
import tempfile

class DataPreparation:

    def __init__(self, variable_name, output_directory):
        self.variable_name = variable_name
        self.output_directory = output_directory
        self.latest_year = 2022
        self.initial_year = 1950
        self.url = "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/Dailies/pressure/"

    def fetch_data_from_url(self, year):
        data_url = os.path.join(self.url, f"{self.variable_name}.{year}.nc")

        response = requests.get(data_url)
        with tempfile.NamedTemporaryFile() as temp_file:
            temp_file.write(response.content)
            with Dataset(temp_file.name, mode='r') as data_file:
                return data_file[self.variable_name][:]

    def subset_lat_lon(self, data_array):
        return data_array[:, :, :29, :]

    def merge_data_into_single_file(self):
        yearly_data = []
        for year in range(self.initial_year, self.latest_year + 1):
            data_array = self.fetch_data_from_url(year)
            if data_array is not None:
                subsetted_array = self.subset_lat_lon(data_array)
                yearly_data.append(subsetted_array)

        merged_array = np.concatenate(yearly_data)
        np.save(os.path.join(self.output_directory, f"{self.variable_name}_merged_data.npy"), merged_array)
        return merged_array

    def standardize_data(self, merged_array):
        data_mean = np.mean(merged_array, axis=0, dtype="float32")
        data_std = np.std(merged_array, axis=0, dtype="float32")
        standardized_data = (merged_array - data_mean) / data_std
        np.save(os.path.join(self.directory, f"{self.variable_name}_standardized_data.npy"), standardized_data)
        return standardized_data

    def create_consecutive_triplets(self, standardized_data):
        subset_1 = standardized_data[:-2]
        subset_2 = standardized_data[1:-1]
        subset_3 = standardized_data[2:]
        stacked_array = np.stack([subset_1, subset_2, subset_3], axis=0)
        np.save(os.path.join(self.directory, f"{self.variable_name}_triplets_data.npy"), stacked_array)

    def process_data(self):
        merged_data = self.merge_data_into_single_file()
        standardized_data = self.standardize_data(merged_data)
        self.create_consecutive_triplets(standardized_data)


if __name__ == "__main__":
    variable_list = ["hgt", "air", "uwnd", "vwnd", "omega"]
    for variable_name in variable_list:
        data_preparation = DataPreparation(variable_name=variable_name, output_directory="./data")
        data_preparation.process_data()