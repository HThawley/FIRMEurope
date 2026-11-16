from datetime import datetime
import os
import shutil
from typing import List
import numpy as np

from firm_ce.analysis.statistics import Statistics
from firm_ce.analysis.validate import Validation, ValidationTensor
from firm_ce.common.exceptions import ValidationError
from firm_ce.common.logging import init_model_logger, init_results_directory
from firm_ce.io.data_model import ModelData
from firm_ce.system.config import ModelConfig
from firm_ce.system.scenario import Scenario


class Model:
    """
    Primary interface for performing a long-term energy planning optimisation using the FIRM framework.
    """

    def __init__(
        self,
        config_directory: str = "inputs/config",
        data_directory: str = "inputs/data",
        logging_flag: bool = True,
        **kwargs,
    ) -> None:
        self.config_directory = config_directory
        self.data_directory = data_directory

        model_data = ModelData(
            config_directory=self.config_directory,
            data_directory=self.data_directory,
        )

        model_data.config_data.update(kwargs)

        model_data.results_dir = init_results_directory(
            model_data.model_name,
            logging_flag,
            str(model_data.config_data.get("model_location", "new")),
        )
        if not os.path.isdir(model_data.results_dir):
            os.mkdir(model_data.results_dir)
        shutil.copy(
            os.path.join(self.config_directory, "config.csv"),
            os.path.join(model_data.results_dir, "config_archive.csv"),
        )
        model_data.logger = init_model_logger(model_data.results_dir)

        if not model_data.validate():
            raise ValidationError(
                "Model failed validation. Check the `log.txt` and modify the config and data files to resolve errors."
            )

        self.config = ModelConfig(model_data.config)

        if self.config.type == "mhmga" and not logging_flag:
            print(f"WARNING: [MHMGA] Most results will be lost if logging is disabled ({logging_flag=})")

        self.datafile_filenames_dict = model_data.datafiles
        self.scenarios = {
            model_data.scenarios[scenario_id]["scenario_name"]: Scenario(model_data, self.config, scenario_id)
            for scenario_id in model_data.scenarios
        }

    def _run_scenario(self, scenario: Scenario, label: str) -> None:
        start_time = datetime.now()
        start_time_str = start_time.strftime("%d/%m/%Y %H:%M:%S")
        scenario.logger.info(f"[{label}] Started scenario '{scenario.name}' at {start_time_str}.")

        scenario.load_datafiles()
        datafile_loadtime = datetime.now()
        datafile_loadtime_str = datafile_loadtime.strftime("%d/%m/%Y %H:%M:%S")
        scenario.logger.info(
            f"[{label}] Datafiles loaded at {datafile_loadtime_str} "
            f"({(datafile_loadtime - start_time).total_seconds():.4f} seconds)."
        )

        result = scenario.solve()

        solve_time = datetime.now()
        solve_time_str = solve_time.strftime("%d/%m/%Y %H:%M:%S")
        scenario.logger.info(
            f"[{label}] Optimisation completed at {solve_time_str} "
            f"({(solve_time - datafile_loadtime).total_seconds()/(60*60):.4f} hours)."
        )

        best_x = result.x if hasattr(result, "x") else result
        scenario.build_and_evaluate_solution(best_x)

        if self.config.backend == "tensor":
            scenario.validation = ValidationTensor(scenario.solutionTensor, scenario.solution_dir, scenario)
            scenario.validation.validate()
            scenario.validation.dump_logs()

            scenario.statistics = Statistics(scenario, solutionTensor=scenario.solutionTensor)
            scenario.statistics.generate_result_files(write=True)
        else:
            scenario.validation = Validation(scenario.solution, scenario.solution_dir)
            scenario.validation.validate()
            scenario.validation.dump_logs()

            scenario.statistics = Statistics(scenario, solution=scenario.solution)
            scenario.statistics.generate_result_files()

        results_time = datetime.now()
        results_time_str = results_time.strftime("%d/%m/%Y %H:%M:%S")
        scenario.logger.info(
            f"[{label}] Results saved at {results_time_str} "
            f"({(results_time - solve_time).total_seconds():.4f} seconds)."
        )

        scenario.unload_datafiles()

        end_time = datetime.now()
        end_time_str = end_time.strftime("%d/%m/%Y %H:%M:%S")
        scenario.logger.info(
            f"[{label}] Scenario completed at {end_time_str} "
            f"(Total time taken: {(end_time - start_time).total_seconds()/(60*60):.4f} hours)."
        )
        return None

    def single_time(self, scenario: Scenario) -> None:
        return self._run_scenario(scenario, "Single Time")

    def mhmga(self, scenario: Scenario) -> None:
        return self._run_scenario(scenario, "MHMGA")

    def solve(self, *, scenarios: str | List[str] = "all") -> None:
        """
        Execute an optimisation for each Scenario: load datafiles, run the optimisation, generate and write results,
        then unload data before moving to the next Scenario.
        """
        if isinstance(scenarios, (list, tuple)):
            scenarios = [s.lower() for s in scenarios]

        for scenario in self.scenarios.values():
            scenario_name = scenario.name.lower()
            if isinstance(scenarios, str):
                if scenarios != "all" and scenario_name != scenarios:
                    continue
            elif isinstance(scenarios, list):
                if scenario_name not in scenarios:
                    continue

            if self.config.type == "single_time":
                self.single_time(scenario)
            elif self.config.type == "mhmga":
                self.mhmga(scenario)

        return None

    def inspect_recombination(
        self,
        scenario_name: str,
        starting_population: np.ndarray,
        objectives: np.ndarray = None,
        constraints: np.ndarray = None,
        evaluate_offspring: bool = True,
        **hyperparameters,
    ) -> dict:
        scenario = None
        for s in self.scenarios.values():
            if s.name.lower() == scenario_name.lower():
                scenario = s
                break

        if scenario is None:
            raise ValueError(f"Scenario '{scenario_name}' not found.")

        scenario.logger.info(f"[Inspect] Loading datafiles for {scenario.name}...")
        scenario.load_datafiles()

        results = scenario.inspect_mhmga_recombination(
            starting_population,
            objectives,
            constraints,
            evaluate_offspring,
            **hyperparameters,
        )

        scenario.unload_datafiles()
        scenario.logger.info(f"[Inspect] Completed for {scenario.name}.")

        return results
