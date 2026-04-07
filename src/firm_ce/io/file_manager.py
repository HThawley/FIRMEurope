from pathlib import Path
from typing import Any, Dict, Union
from numbers import Number

# import numpy as np
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
            "nodes",
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
        try:
            imported_dict = pd.read_csv(filepath, index_col="id")
            for col in imported_dict.columns:
                if col in ("filename", "units"):
                    continue
                if hasattr(imported_dict[col], "str"):
                    imported_dict[col] = imported_dict[col].str.lower()
            imported_dict = imported_dict.to_dict(orient="index")
            for idx in imported_dict:
                imported_dict[idx]["id"] = idx
        except Exception as e:
            print(f"Error on {filepath}")
            raise e
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
        df.columns = [str(col).lower() for col in df.columns]
        return {col: df[col].to_numpy() for col in df.columns}


class DataFile:
    """
    Container for a named datafile and its content.
    """

    def __init__(self, datafile_filenames_dict: Dict, file_directory: str) -> None:
        """
        Initialize the DataFile with a filename and its type.

        Parameters:
        -------
        datafile_filenames_dict (dict): datafiles.csv object read in as a dict
        datafile_type (str): Descriptive type of the datafile (e.g., 'time series').
        """
        self.name = datafile_filenames_dict["filename"]
        self.type = datafile_filenames_dict["datafile_type"]
        self.units = datafile_filenames_dict["units"]
        self.file_directory = file_directory
        self.data = ImportDatafile(self.file_directory, self.name).get_data()

    def __repr__(self) -> str:
        return f"DataFile ({self.name!r}, {self.type!r}"  # , {self.data!r})"


class ResultFile:
    """
    Container class for saving model results into a CSV file.

    Handles writing headers, formatting data, and rounding values
    before output.S
    """

    def __init__(
        self,
        report: str,
        target_directory: str,
        data: pd.DataFrame,
        decimals: Union[int, None] = None,
        file_ext: str = "csv",
        write_kwargs: dict = None,
    ):
        """
        Initialise a result file object.

        Parameters:
        ----------
        file_type (str): Identifier for the file (used as the filename base).
        target_directory (str): Directory where the file will be saved.
        header (List[str]): List of column headers for the CSV file.
        data_array (Union[NDArray[npfloat], NDArray[npint]]): Data to
            write into the CSV file.
        decimals (Union[int, None], optional): Number of decimal places to round
            data values to. If None, values are written without rounding.
        """
        file_ext = file_ext.removeprefix(".")
        dots = report.count(".")
        if dots > 1:
            raise ValueError(f"'report' has too many '.'s, ({dots})")
        elif dots == 1:
            if (implied_type := report.split(".")[-1].lower()) != file_ext.lower():
                raise ValueError(f"implied file type ({implied_type}) does not match the declared file type ({file_ext})")

        self.file_ext = file_ext.lower()
        self.report = report.split(".")[0]
        self.target_directory = target_directory
        self.file_path = f"{target_directory}/{report}.{self.file_ext}"
        self.decimals = decimals
        self.data = data
        self.write_kwargs = write_kwargs if write_kwargs is not None else {}

    def __repr__(self) -> str:
        return f"ResultFile ({self.report!r})"

    def _round_value(self, value):
        if self.decimals is None:
            return value
        try:
            if isinstance(value, Number):
                return round(value, self.decimals)
            return value
        except (TypeError, ValueError):
            return value

    def _round_data(self) -> pd.DataFrame:
        if self.decimals is None:
            return self.data
        return self.data.map(self._round_value)

    def write(self, **write_kwargs):
        """
        Write the data to a file based on file type.

        Each row of data_array is written to the file. Optionally,
        values are rounded to the specified number of decimals.

        Returns:
        -------
        None.

        Side-effects:
        ------------
        Creates a file in target_directory with the name
        <target_directory>/<file_type>.<file_ext> and prints a confirmation message.
        """
        if self.file_ext == "csv":
            self.write_csv(**write_kwargs)
        elif self.file_ext == "xlsx":
            self.write_xlsx(**write_kwargs)
        # elif self.file_ext == "json":
            # self.write_json()
        else:
            raise NotImplementedError("Only 'csv', 'xlsx' are currently supported.")

    def write_csv(self, **write_kwargs):
        """
        Write the data to a CSV file.

        Multi-line headers are possible.

        Each row of data_array is written to the file. Optionally,
        values are rounded to the specified number of decimals.

        Returns:
        -------
        None.

        Side-effects:
        ------------
        Creates a CSV file in target_directory with the name
        <report>.csv and prints a confirmation message.
        """
        default_kwargs = {
            "header": False,
            "index": True,
            "mode": "x",
        }
        for k, v in self.write_kwargs.items():
            # overwrite default kwargs with user-supplied kwargs
            default_kwargs[k] = v
        for k, v in write_kwargs.items():
            # overwrite __init__ supplied kwargs with user-supplied kwargs
            default_kwargs[k] = v

        if self.decimals is not None:
            self.data = self._round_data()

        self.data.to_csv(self.file_path, **default_kwargs)

        print(f"Saved {self.report} to {self.target_directory}")
        return None

    def write_xlsx(self, **write_kwargs):
        """
        Write the data to an Excel file.

        Multi-line headers are possible.

        Each row of data_array is written to the file. Optionally,
        values are rounded to the specified number of decimals.

        Returns:
        -------
        None.

        Side-effects:
        ------------
        Creates a xlsx file in target_directory with the name
        <report>.xlsx and prints a confirmation message.
        """
        default_kwargs = {
            "header": False,
            "index": True,
            "mode": "x",
            "sheet_name": self.report,
        }
        for k, v in self.write_kwargs.items():
            # overwrite default kwargs with user-supplied kwargs
            default_kwargs[k] = v
        for k, v in write_kwargs.items():
            # overwrite __init__ supplied kwargs with user-supplied kwargs
            default_kwargs[k] = v

        if self.decimals is not None:
            self.data = self._round_data()

        with pd.ExcelWriter(self.file_path, mode=default_kwargs.pop("mode")) as writer:
            self.data.to_excel(writer, **default_kwargs)

        print(f"Saved {self.report} to {self.target_directory}")
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
