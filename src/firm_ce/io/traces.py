# type: ignore
from typing import Dict

import numpy as np
from numpy.typing import NDArray

from firm_ce.common.exceptions import ValidationError
from firm_ce.common.typing import npfloat
from firm_ce.io.file_manager import DataFile
from firm_ce.scalar.entities.components import Fleet_InstanceType
from firm_ce.scalar.entities.topology import Network_InstanceType
from firm_ce.scalar.methods import fuel_m, generator_m, node_m, storage_m


unit_multiples = {
    "C": 1.0,  # CF - capacity factor
    "c": 1.0,  # CF - capacity factor
    "-": 1.0,  # unitless (capacity factor)
    "G": 1.0,  # maths will be done in GW|GWh
    "M": 0.001,
    "k": 0.000_001,
    "T": 1000.0,
}


def aggregate_trace(trace: np.ndarray, factor: int, method: str) -> np.ndarray:
    """
    Aggregates a 1D time-series array by a given integer factor.
    """
    if factor <= 1:
        return trace

    n_full = len(trace) // factor
    remainder = len(trace) % factor

    # Fast path for perfect divisibility
    if remainder == 0:
        reshaped = trace.reshape(-1, factor)
        if method == "mean":
            return reshaped.mean(axis=1)
        elif method == "sum":
            return reshaped.sum(axis=1)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    # Handle non-divisible traces
    full_part = trace[: n_full * factor].reshape(-1, factor)
    rem_part = trace[n_full * factor :]

    if method == "mean":
        agg_full = full_part.mean(axis=1)
        agg_rem = np.array([rem_part.mean()])
        return np.concatenate((agg_full, agg_rem))
    elif method == "sum":
        agg_full = full_part.sum(axis=1)
        # Scale the remainder sum to match the expected window length
        agg_rem = np.array([rem_part.sum() / remainder * factor])
        return np.concatenate((agg_full, agg_rem))
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def select_datafile(
    datafile_type: str,
    object_name: str,
    datafiles_imported_dict: Dict[str, DataFile],
    limit_timesteps: int = None,
    yeartuple: tuple[int] = None,
    error_on_fail: bool = True,
    interval_aggregation: int = 1,
) -> NDArray[npfloat]:
    """
    Locates and returns a data trace of a specified datafile_type associated with
    either a Generator, Storage, Fuel, or Node object based upon the object's name.
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

            if interval_aggregation > 1 and datafile_type not in ("fuel_constraint", "flexible_annual_limit"):
                method = "sum" if datafile_type == "inflow" else "mean"
                trace = aggregate_trace(trace, interval_aggregation, method)

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
    """
    return data[:limit_timesteps]


def trim_with_years(
    data: NDArray[npfloat],
    year_trace: NDArray[npfloat],
    yeartuple: tuple[int],
) -> NDArray[npfloat]:
    """
    Trims the data array to only include entries within the specified year range.
    """
    firstyear, finalyear = yeartuple
    in_time_mask = np.ones(data.shape, np.bool_)
    if firstyear not in ("auto", "none", ""):
        in_time_mask *= year_trace >= firstyear
    if finalyear not in ("auto", "none", ""):
        in_time_mask *= year_trace <= finalyear
    return data[in_time_mask]


def load_datafiles_to_fuels(
    fleet: Fleet_InstanceType,
    datafiles_imported_dict: Dict[str, DataFile],
    yeartuple: tuple[int] = None,
    interval_aggregation: int = 1,
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
                interval_aggregation=interval_aggregation,
            ),
        )

    return None


def load_datafiles_to_generators(
    fleet: Fleet_InstanceType,
    datafiles_imported_dict: Dict[str, DataFile],
    resolution: float,
    limit_timesteps: int = None,
    yeartuple: tuple[int] = None,
    interval_aggregation: int = 1,
) -> None:
    """
    Iterates through all generators in the fleet and loads their time-series data to each
    instance.
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
                interval_aggregation=interval_aggregation,
            ),
            resolution,
        )

    return None


def load_datafiles_to_storages(
    fleet: Fleet_InstanceType,
    datafiles_imported_dict: Dict[str, DataFile],
    limit_timesteps: int = None,
    yeartuple: tuple[int] = None,
    interval_aggregation: int = 1,
) -> None:
    """
    Iterates through all storages in the fleet and loads their time-series data to each
    instance which takes a datafile.
    """
    for storage in fleet.storages.values():
        if storage.inflows:
            storage_m.load_data(
                storage,
                select_datafile(
                    "inflow",
                    storage.name,
                    datafiles_imported_dict,
                    limit_timesteps,
                    yeartuple,
                    interval_aggregation=interval_aggregation,
                ),
            )
    return None


def load_datafiles_to_network(
    network: Network_InstanceType,
    datafiles_imported_dict: Dict[str, DataFile],
    limit_timesteps: int = None,
    yeartuple: tuple[int] = None,
    demand_multiple: float = 1.0,
    interval_aggregation: int = 1,
) -> None:
    """
    Iterates through all nodes in the network and loads their time-series 'demand' data to each
    instance.
    """
    for node in network.nodes.values():
        node_m.load_data(
            node,
            select_datafile(
                "demand",
                node.name,
                datafiles_imported_dict,
                limit_timesteps,
                yeartuple,
                interval_aggregation=interval_aggregation,
            ),
            npfloat(demand_multiple),
        )
    return None


def unload_data_from_fuels(fleet: Fleet_InstanceType):
    for fuel in fleet.fuels.values():
        fuel_m.unload_data(fuel)
    return None


def unload_data_from_generators(fleet: Fleet_InstanceType):
    for generator in fleet.generators.values():
        generator_m.unload_data(generator)
    return None


def unload_data_from_storages(fleet: Fleet_InstanceType):
    for storage in fleet.storages.values():
        if storage.inflows:
            storage_m.unload_data(storage)
    return None


def unload_data_from_network(network: Network_InstanceType):
    for node in network.nodes.values():
        node_m.unload_data(node)
    return None
