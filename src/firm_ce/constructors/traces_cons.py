# type: ignore
from typing import Dict

import numpy as np
from numpy.typing import NDArray

from firm_ce.common.exceptions import ValidationError
from firm_ce.common.typing import npfloat
from firm_ce.fast_methods import generator_m, storage_m, node_m, fuel_m
from firm_ce.io.file_manager import DataFile
from firm_ce.system.components import Fleet_InstanceType
from firm_ce.system.topology import Network_InstanceType


unit_multiples = {
    "C": 1.0,  # CF - capacity factor
    "c": 1.0,  # CF - capacity factor
    "-": 1.0,  # unitless (capacity factor)
    "G": 1.0,  # maths will be done in GW|GWh
    "M": 0.001,
    "k": 0.000_001,
    "T": 1000.0,
}


def select_datafile(
    datafile_type: str,
    object_name: str,
    datafiles_imported_dict: Dict[str, DataFile],
    limit_timesteps: int = None,
    yeartuple: tuple[int] = None,
    error_on_fail: bool = True,
) -> NDArray[npfloat]:
    """
    Locates and returns the a data trace of a specified datafile_type associated with
    either a Generator or Node object based upon the object's name.

    Parameters:
    -------
    datafile_type (str): The type of datafile. Either 'generation', 'flexible_annual_limit',
        'demand', 'inflow'
    object_name (str): The name attribute of the Generator or Node instance.
    datafiles_imported_dict (Dict[str, DataFile]): A dictionary of DataFile instances, where
        the key is a str of the id in `config/datafiles.csv`.

    Returns:
    -------
    NDArray[npfloat]: A 1-dimensional numpy array containing the data trace for the
        specified datafile_type and object_name. If no trace was found, an empty array
        is returned.
    """
    matching_datafiles = [df for df in datafiles_imported_dict.values() if df.type == datafile_type]

    trace = np.empty((0,), dtype=npfloat)
    for datafile in matching_datafiles:
        if object_name in datafile.data.keys():
            trace = np.array(datafile.data[object_name], dtype=npfloat)
            if limit_timesteps is not None:
                # debug supercedes yeartuple specified in scenario
                trace = trim_with_timesteps(trace, limit_timesteps)
            elif yeartuple is not None:
                trace = trim_with_years(trace, datafile.data["year"], yeartuple)

            trace *= unit_multiples[datafile.units[0]]
            return trace

    if error_on_fail:
        raise ValidationError(f"No matching datafiles: {datafile_type=}, {object_name=}")

    return trace


def trim_with_timesteps(
    data: NDArray[npfloat],
    limit_timesteps: int,
) -> NDArray[npfloat]:
    """
    Trims the data array to only include the first `limit_timesteps` entries.

    Parameters:
    -------
    data (NDArray[npfloat]): The full data array.
    limit_timesteps (int): The number of timesteps to retain.

    Returns:
    -------
    NDArray[npfloat]: The trimmed data array.
    """
    return data[:limit_timesteps]


def trim_with_years(
    data: NDArray[npfloat],
    year_trace: NDArray[npfloat],
    yeartuple: tuple[int],
) -> NDArray[npfloat]:
    """
    Trims the data array to only include entries within the specified year range.

    Parameters:
    -------
    data (NDArray[npfloat]): The full data array.
    year_trace (NDArray[npfloat]): An array indicating the year corresponding to each entry in `data`.
    yeartuple (tuple[int]): A tuple containing the first and last year to retain (inclusive).

    Returns:
    -------
    NDArray[npfloat]: The trimmed data array.
    """
    firstyear, finalyear = yeartuple
    in_time_mask = np.ones(data.shape, np.bool_)
    if firstyear not in ('auto', 'none', ''):
        in_time_mask *= year_trace >= firstyear
    if finalyear not in ('auto', 'none', ''):
        in_time_mask *= year_trace <= finalyear
    return data[in_time_mask]


def load_datafiles_to_fuels(
    fleet: Fleet_InstanceType,
    datafiles_imported_dict: Dict[str, DataFile],
    yeartuple: tuple[int] = None,
) -> None:
    for fuel in fleet.fuels.values():
        fuel_m.load_data(
            fuel,
            select_datafile(
                "fuel_constraint",
                fuel.name,
                datafiles_imported_dict,
                None,
                yeartuple,
            ),
        )

    return None


def load_datafiles_to_generators(
    fleet: Fleet_InstanceType,
    datafiles_imported_dict: Dict[str, DataFile],
    resolution: float,
    limit_timesteps: int = None,
    yeartuple: tuple[int] = None,
) -> None:
    """
    Iterates through all generators in the fleet and loads their time-series data to each
    instance. The baseload, solar, and wind generators are expected to have 'generation'
    traces defining their capacity factor in each time interval, and the flexible generators
    are expected to have a 'flexible_annual_limit' trace defining their maximum generation
    in each year.

    Parameters:
    -------
    fleet (Fleet_InstanceType): A static instance of the Fleet jitclass.
    datafiles_imported_dict (Dict[str, DataFile]): A dictionary of DataFile instances, where
        the key is the id in `config/datafiles.csv`.
    resolution (float): The time resolution of each interval for the input data [hours/interval].

    Returns:
    -------
    None.

    Side-effects:
    -------
    The data_status, data, and annual_constraints_data attributes of each generator object are
    modified.

    The residual_load at the node where each generator is located is also updated. The update
    to residual load is based upon the initial capacity, resolution, and generation trace. This
    means that load_datafiles_to_network must be run before load_datafiles_to_generators.
    """
    for generator in fleet.generators.values():
        generator_m.load_data(
            generator,
            select_datafile(
                "generation",
                generator.name,
                datafiles_imported_dict,
                limit_timesteps,
                yeartuple,
                not generator.is_flexible,
            ),
            resolution,
        )

    return None


def load_datafiles_to_storages(
    fleet: Fleet_InstanceType,
    datafiles_imported_dict: Dict[str, DataFile],
    limit_timesteps: int = None,
    yeartuple: tuple[int] = None,
) -> None:
    """
    Iterates through all storages in the fleet and loads their time-series data to each
    instance which takes a datafile. The storages may have an 'inflow' traces defining the
    inflow of energy to the storage in each time interval. This is usually used for modelling
    hydro assets.

    Parameters:
    -------
    fleet (Fleet_InstanceType): A static instance of the Fleet jitclass.
    datafiles_imported_dict (Dict[str, DataFile]): A dictionary of DataFile instances, where
        the key is the id in `config/datafiles.csv`.
    resolution (float): The time resolution of each interval for the input data [hours/interval].

    Returns:
    -------
    None.

    Side-effects:
    -------
    The data_status, data, attributes of each storage object are modified.
    """
    for storage in fleet.storages.values():
        if storage.inflows:
            storage_m.load_data(
                storage,
                select_datafile("inflow", storage.name, datafiles_imported_dict, limit_timesteps, yeartuple),
            )
    return None


def load_datafiles_to_network(
    network: Network_InstanceType,
    datafiles_imported_dict: Dict[str, DataFile],
    limit_timesteps: int = None,
    yeartuple: tuple[int] = None,
) -> None:
    """
    Iterates through all nodes in the network and loads their time-series 'demand' data to each
    instance. The demand data is in units of MW.

    Parameters:
    -------
    network (Network_InstanceType): A static instance of the Network jitclass.
    datafiles_imported_dict (Dict[str, DataFile]): A dictionary of DataFile instances, where
        the key is the id in `config/datafiles.csv`.

    Returns:
    -------
    None.

    Side-effects:
    -------
    The data_status and data attributes of each node object are modified.

    The residual_load at the node where each generator is located is initialised with a copy
    of the demand trace.
    """
    for node in network.nodes.values():
        node_m.load_data(
            node,
            select_datafile("demand", node.name, datafiles_imported_dict, limit_timesteps, yeartuple),
        )
    return None


def unload_data_from_fuels(
        fleet: Fleet_InstanceType
):
    for fuel in fleet.fuels.values():
        fuel_m.unload_data(fuel)
    return None


def unload_data_from_generators(
        fleet: Fleet_InstanceType
):
    """
    Iterates through all generators and unloads time-series data. Allows large amounts of
    memory to be cleared before running an optimisation for a new scenario.

    Parameters:
    -------
    fleet (Fleet_InstanceType): A static instance of the Fleet jitclass.

    Returns:
    -------
    None.

    Side-effects:
    -------
    The data_status, data, and annual_constraints_data attributes of each generator object are
    modified.
    """
    for generator in fleet.generators.values():
        generator_m.unload_data(generator)
    return None


def unload_data_from_storages(
        fleet: Fleet_InstanceType
):
    """
    Iterates through all storages and unloads time-series data. Allows large amounts of
    memory to be cleared before running an optimisation for a new scenario.

    Parameters:
    -------
    fleet (Fleet_InstanceType): A static instance of the Fleet jitclass.

    Returns:
    -------
    None.

    Side-effects:
    -------
    The data_status, data attributes of each generator object are modified.
    """
    for storage in fleet.storages.values():
        if storage.inflows:
            storage_m.unload_data(storage)
    return None


def unload_data_from_network(
        network: Network_InstanceType
):
    """
    Iterates through all nodes and unloads time-series data. Allows large amounts of
    memory to be cleared before running an optimisation for a new scenario.

    Parameters:
    -------
    network (Network_InstanceType): A static instance of the Network jitclass.

    Returns:
    -------
    None.

    Side-effects:
    -------
    The data_status and data attributes of each node object are modified.
    """
    for node in network.nodes.values():
        node_m.unload_data(node)
    return None
