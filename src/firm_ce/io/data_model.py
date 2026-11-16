from firm_ce.io.config_validation import is_nan, validate_config
from firm_ce.io.file_manager import import_config_csvs

__all__ = ["ModelData", "is_nan", "validate_config"]


class ModelData:
    def __init__(self, config_directory: str, data_directory: str) -> None:
        self.config_directory = config_directory
        self.data_directory = data_directory

        # Get the config settings for the csvs
        self.config_data = import_config_csvs(config_directory=config_directory)

        # Get the model name
        self.model_name = self.get_model_name()

        # Set all the relevant parameters
        self.scenarios = self.config_data["scenarios"]
        self.nodes = self.config_data["nodes"]
        self.generators = self.config_data["generators"]
        self.fuels = self.config_data["fuels"]
        self.lines = self.config_data["lines"]
        self.storages = self.config_data["storages"]
        self.config = self.config_data["config"]
        self.x0s = self.config_data["initial_guess"]
        self.datafiles = self.config_data["datafiles"]

    def validate(self) -> bool:
        return validate_config(self)

    def get_model_name(self) -> str:
        model_name = None

        if "config" in self.config_data:
            for record in self.config_data["config"].values():
                if record["name"] == "model_name":
                    model_name = record["value"]
                    break

        if model_name is None:
            model_name = "Model"

        return model_name
