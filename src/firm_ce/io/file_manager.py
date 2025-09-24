import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray


class ImportCSV:
    """
    Class for importing CSV files from a given repository.
    """

    def __init__(self, repository: Path) -> None:
        """
        Initialize the importer with a path to the repository.

        Parameters:
        -------
        repository (Path): Path to the directory containing CSV files.
        """

        self.repository = Path(repository)
        self.config_filenames = (
            "scenarios",
            "generators",
            "fuels",
            "lines",
            "storages",
            "config",
            "initial_guess",
            "datafiles",
        )

        if not self.repository.is_dir():
            raise FileNotFoundError(f"Repository {repository} does not exist.")

    def get_data(self, filename: str) -> Dict[str, Dict[str, Any]]:
        """
        Load a CSV file into a nested dictionary using 'id' as the index.

        Parameters:
        -------
        filename (str): Name of the CSV file to load.

        Returns:
        -------
        Dict[str, Dict[str, Any]]: Dictionary of records keyed by ID.
        """

        filepath = self.repository.joinpath(filename)
        if not filepath.is_file():
            raise FileNotFoundError(f"File {filepath} does not exist.")
        imported_dict = pd.read_csv(filepath, index_col="id").to_dict(orient="index")
        for idx in imported_dict:
            imported_dict[idx]["id"] = idx
        return imported_dict

    def get_config_dict(self) -> Dict[str, Dict[str, Any]]:
        return {fn: self.get_data(fn + ".csv") for fn in self.config_filenames}


class ImportDatafile:
    """
    Class for importing a single CSV datafile into a dictionary of NDArrays.
    """

    def __init__(self, repository: Path, filename: str) -> None:
        """
        Initialize the datafile importer.

        Parameters:
        -------
        repository (Path): Directory containing the CSV file.
        filename (str): Name of the CSV datafile to import.
        """

        self.repository = Path(repository)
        self.filename = filename

        if not self.repository.is_dir():
            raise FileNotFoundError(f"Repository {repository} does not exist.")

    def __repr__(self) -> str:
        return f"ImportDatafile ({self.filename!r})"

    def get_data(self) -> Dict[str, NDArray]:
        """
        Load the CSV data into a dictionary of column-wise NumPy arrays.

        Returns:
        -------
        Dict[str, NDArray]: Dictionary where keys are column names and values are NumPy arrays.
        """
        filepath = self.repository.joinpath(self.filename)
        if not filepath.is_file():
            raise FileNotFoundError(f"File {filepath} does not exist.")

        df = pd.read_csv(filepath)
        return {col: df[col].to_numpy() for col in df.columns}


class DataFile:
    """
    Container for a named datafile and its content.
    """

    def __init__(self, filename: str, datafile_type: str, file_directory: str) -> None:
        """
        Initialize the DataFile with a filename and its type.

        Parameters:
        -------
        filename (str): Name of the datafile.
        datafile_type (str): Descriptive type of the datafile (e.g., 'time series').
        """
        self.name = filename
        self.type = datafile_type
        self.file_directory = file_directory
        self.data = ImportDatafile(file_directory, filename).get_data()

    def __repr__(self) -> str:
        return f"DataFile ({self.name!r}, {self.type!r}, {self.data!r})"


class ResultFile:
    def __init__(
        self,
        file_type: str,
        target_directory: str,
        header: List[str],
        data_array: Union[NDArray[np.float64], NDArray[np.int64]],
        decimals: Union[int, None] = None,
    ):
        self.name = file_type + ".csv"
        self.type = file_type
        self.target_directory = target_directory
        self.header = header
        self.data = data_array
        self.decimals = decimals

    def __repr__(self) -> str:
        return f"ResultFile ({self.type!r})"

    def write(self):
        with open(os.path.join(self.target_directory, self.name), mode="w", newline="") as file:
            writer = csv.writer(file)
            if self.header:
                writer.writerow(self.header.split(",") if isinstance(self.header, str) else self.header)
            for row in self.data:
                writer.writerow(np.round(row, decimals=self.decimals) if self.decimals is not None else row)
        print(f"Saved {self.name} to {self.target_directory}")
        return None


def import_config_csvs(config_directory: str) -> Dict[str, Any]:
    """
    Load all model configuration CSVs into a single dictionary.

    Returns:
    -------
    Dict[str, Any]: A dictionary containing model configuration data.
    """

    csv_importer = ImportCSV(config_directory)
    data = csv_importer.get_config_dict()

    return data
