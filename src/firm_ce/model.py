import time
from datetime import datetime

from firm_ce.common.constants import DEBUG
from firm_ce.common.exceptions import ValidationError
from firm_ce.io.validate import ModelData
from firm_ce.optimisation.statistics import Statistics
from firm_ce.system.parameters import ModelConfig
from firm_ce.system.scenario import Scenario


class Model:
    """
    Primary interface for performing a long-term energy planning optimisation using the FIRM framework.

    Notes:
    -------
    - Input configuration files are loaded into the ModelData dataclass and validated before constructing the Scenarios.
    - Time-series data files are loaded after calling the solve() method. Data files are only loaded for one Scenario at a time
    in order to manage memory. After the optimisation for a Scenario has completed and the results saved, data files are unloaded
    from the Scenario.
    - Loggers are handled separately for each Scenario. All results, including log files, are saved in the `results` folder. A
    sub-directory for the Model instance contains separate sub-directories for each Scenario instances.

    Attributes:
    -------
    config_directory (str): Filesystem path to the configuration directory. Defaults to the example `inputs/config` folder.
    data_directory (str): Filesystem path to the directory containing input data files. Defaults to the example `inputs/data`
        folder.
    config (ModelConfig): Data class containing validated model configuration used for the optimisation settings and model-level
        metadata.
    datafile_filenames_dict (Dict[str, Dict[str, str]]): Raw data imported from the `datafiles.csv` config file for the Model.
        Each row of the config file is associated with an item in all_datafiles.
    scenarios (Dict[str, Scenario]): Mapping from scenario name to initialised Scenario instances constructed from the validated
        ModelData.
    """

    def __init__(
        self,
        config_directory: str = "inputs/config",
        data_directory: str = "inputs/data",
        logging_flag: bool = True,
    ) -> None:
        """
        Initialises a Model instance.

        Parameters:
        -------
        config_directory (str): Filesystem path to the directory containing input data files. Defaults to the example
            `inputs/data` folder.
        data_directory (str): Filesystem path to the directory containing input data files. Defaults to the example
            `inputs/data` folder.
        logging_flag (bool): If True, creates a model-level folder in `results` containing the Scenario sub-directories and
            log files. When set to false, no model-level results folder is created and the log is stored in `results/temp`
            instead (useful when generating Statistic instance directly using an initial guess).
        """
        self.config_directory = config_directory
        self.data_directory = data_directory
        model_data = ModelData(config_directory=self.config_directory, logging_flag=logging_flag)

        if not model_data.validate():
            raise ValidationError(
                "Model failed validation. Check the `log.txt` and modify the config and data files to resolve errors."
            )

        self.config = ModelConfig(model_data.config)
        self.datafile_filenames_dict = model_data.datafiles
        self.scenarios = {
            model_data.scenarios[scenario_idx]["scenario_name"]: Scenario(model_data, scenario_idx)
            for scenario_idx in model_data.scenarios
        }

    def solve(self) -> None:
        """
        Execute an optimisation for each Scenario: load datafiles, run the optimisation, generate and write results,
        then unload data before moving to the next Scenario.

        Parameters:
        -------
        None.

        Returns:
        -------
        None.

        Side-effects:
        -------
        Modification of the Scenario objects, primarily through loading exogenous time-series data files (modifying
        the Scenario.fleet.generators, Scenario.network.nodes, and Scenario.static) and the creation of a Solver instance in
        Scenario.solver. The optimisation is managed through the Solver, with jitclass attributes for the Scenario remaining
        *static* unmodified instances throughout the optimisation process. These static instances are copied in separate worker
        processes for the optimisation to create dynamic instances that are safe to modify during the optimisation. The dynamic
        instances are not actually contained in the Model instance.
        """
        for scenario in self.scenarios.values():
            start_time = time.time()
            start_time_str = datetime.fromtimestamp(start_time).strftime("%d/%m/%Y %H:%M:%S")
            scenario.logger.info(f"Started scenario {scenario.name} at {start_time_str}.")

            scenario.load_datafiles(self.datafile_filenames_dict, self.data_directory)
            datafile_loadtime = time.time()
            datafile_loadtime_str = datetime.fromtimestamp(datafile_loadtime).strftime("%d/%m/%Y %H:%M:%S")
            scenario.logger.info(
                f"Datafiles loaded at {datafile_loadtime_str} ({datafile_loadtime - start_time:.4f} seconds)."
            )

            de_result = scenario.solve(self.config)

            solve_time = time.time()
            solve_time_str = datetime.fromtimestamp(solve_time).strftime("%d/%m/%Y %H:%M:%S")
            scenario.logger.info(
                f"Optimisation completed at {solve_time_str} ({(solve_time - datafile_loadtime)/(60*60):.4f} hours)."
            )

            if self.config.type == "single_time":
                scenario.statistics = Statistics(
                    de_result.x,
                    scenario.static,
                    scenario.fleet,
                    scenario.network,
                    scenario.results_dir,
                    scenario.name,
                    self.config.balancing_type,
                    self.config.fixed_costs_threshold,
                    True,
                )
                scenario.statistics.generate_result_files()
                scenario.statistics.write_results()
                if DEBUG:
                    scenario.statistics.dump()
                results_time = time.time()
                results_time_str = datetime.fromtimestamp(results_time).strftime("%d/%m/%Y %H:%M:%S")
                scenario.logger.info(f"Results saved at {results_time_str} ({results_time - solve_time:.4f} seconds).")

            scenario.unload_datafiles()

            end_time = time.time()
            end_time_str = datetime.fromtimestamp(end_time).strftime("%d/%m/%Y %H:%M:%S")
            scenario.logger.info(
                f"Scenario completed at {end_time_str} (Total time taken: {(end_time - start_time)/(60*60):.4f} hours)."
            )

        return None
