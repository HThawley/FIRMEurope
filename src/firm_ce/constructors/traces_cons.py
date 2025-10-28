# type: ignore
from typing import Dict

import numpy as np
from numpy.typing import NDArray

from firm_ce.common.exceptions import ValidationError
from firm_ce.fast_methods import generator_m, reservoir_m, node_m
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
    error_on_fail=True,
) -> NDArray[np.float64]:
    """
    Locates and returns the a data trace of a specified datafile_type associated with
    either a Generator or Node object based upon the object's name.

    Parameters:
    -------
    datafile_type (str): The type of datafile. Either 'generation', 'flexible_annual_limit',
        'demand', 'reservoir_inflow'
    object_name (str): The name attribute of the Generator or Node instance.
    datafiles_imported_dict (Dict[str, DataFile]): A dictionary of DataFile instances, where
        the key is a str of the id in `config/datafiles.csv`.

    Returns:
    -------
    NDArray[np.float64]: A 1-dimensional numpy array containing the data trace for the
        specified datafile_type and object_name. If no trace was found, an empty array
        is returned.
    """

    matching_datafiles = [df for df in datafiles_imported_dict.values() if df.type == datafile_type]

    trace = np.empty((0,), dtype=np.float64)
    for datafile in matching_datafiles:
        if object_name in datafile.data.keys():
            trace = np.array(datafile.data[object_name], dtype=np.float64)
            trace *= unit_multiples[datafile.units[0]]
            return trace

    if error_on_fail:
        raise ValidationError(f"No matching datafiles: {datafile_type=}, {object_name=}")

    return trace


def load_datafiles_to_generators(
    fleet: Fleet_InstanceType,
    datafiles_imported_dict: Dict[str, DataFile],
    resolution: float,
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
        if generator.unit_type == "flexible":
            generation, flexible = False, True
        else:
            generation, flexible = True, False

        generator_m.load_data(
            generator,
            select_datafile("generation", generator.name, datafiles_imported_dict, generation),
            select_datafile("flexible_annual_limit", generator.name, datafiles_imported_dict, flexible),
            resolution,
        )

    return None


def load_datafiles_to_reservoirs(
    fleet: Fleet_InstanceType,
    datafiles_imported_dict: Dict[str, DataFile],
) -> None:
    """
    Iterates through all reservoirs in the fleet and loads their time-series data to each
    instance. The reservoirs are expected to have an 'inflow' traces defining the inflow of
    energy to the reservoir in each time interval.

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
    The data_status, data, attributes of each reservoir object are modified.
    """
    for reservoir in fleet.reservoirs.values():
        reservoir_m.load_data(
            reservoir,
            select_datafile("reservoir_inflow", reservoir.name, datafiles_imported_dict),
        )
    return None


def load_datafiles_to_network(
    network: Network_InstanceType,
    datafiles_imported_dict: Dict[str, DataFile],
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
            select_datafile("demand", node.name, datafiles_imported_dict),
        )
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


def unload_data_from_reservoirs(
        fleet: Fleet_InstanceType
):
    """
    Iterates through all reservoirs and unloads time-series data. Allows large amounts of
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
    for reservoir in fleet.reservoirs.values():
        reservoir_m.unload_data(reservoir)
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
