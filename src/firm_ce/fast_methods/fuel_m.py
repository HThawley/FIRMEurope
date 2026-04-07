# type: ignore
import numpy as np

from firm_ce.common.constants import FASTMATH, TOLERANCE, BOUNDSCHECK
from firm_ce.common.exceptions import raise_static_modification_error, raise_getting_unloaded_data_error
from firm_ce.common.typing import nbfloat, unicode_type, npfloat, nbintp, nbintp
from firm_ce.common.jit_overload import njit
from firm_ce.system.components import Fuel, Fuel_InstanceType


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def create_dynamic_copy(
    fuel_instance: Fuel_InstanceType,
) -> Fuel_InstanceType:
    """
    A 'static' instance of the Fuel jitclass (Fuel.static_instance=True) is copied
    and marked as a 'dynamic' instance (Fuel.static_instance=False).

    Static instances are created during Model initialisation and supplied as arguments
    to the differential evolution. These arguments are references to the original jitclass instances (not copies).
    Candidate solutions within the differential evolution are tested in embarrasingly parrallel,
    making it unsafe for multiple workers to similtaneously modify the same memory referenced
    across each process.

    Instead, each worker must create a deep copy of the referenced instance that is safe to modify
    within that worker process. Not all attributes within a dynamic instance are safe to modify.
    Only attributes that are required to be modified when testing the candidate solution are
    copied in order to save memory. If an attribute is unsafe to modify after copying, it will
    be marked with a comment that says "This remains static" in the create_dynamic_copy fast_method for
    that jitclass.

    Parameters:
    -------
    fuel_instance (Fuel_InstanceType): A static instance of the Fuel jitclass.
    nodes_typed_dict (DictType(nbintp, Node_InstanceType)): A typed dictionary of
        all Node jitclass instances for the scenario. Key defined as Node.order.
    lines_typed_dict (DictType(nbintp, Line_InstanceType)): A typed dictionary of
        all Line jitclass instances for the scenario. Key defined as Line.order.

    Returns:
    -------
    Fuel_InstanceType: A dynamic instance of the Fuel jitclass.
    """

    fuel_copy = Fuel(
        False,
        fuel_instance.id,
        fuel_instance.name,
        fuel_instance.cost,
        fuel_instance.emissions
    )
    fuel_copy.annual_limit = fuel_instance.annual_limit.copy()
    fuel_copy.remaining_energy = fuel_instance.remaining_energy.copy()
    fuel_copy.data_status = fuel_instance.data_status

    return fuel_copy


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def allocate_memory(fuel_instance: Fuel_InstanceType, intervals_count: int) -> None:
    """
    Allocates memory for attributes of the Fleet jitclass that are required to be modified when testing candidate solutions
    within the differential evolution.

    Parameters:
    -------
    fuel_instance (Fuel_InstanceType): A dynamic instance of the Fuel jitclass.
    intervals_count (int): The number of time intervals in the scenario, used to dimension arrays.
    """
    if len(fuel_instance.annual_limit) > 0:
        fuel_instance.remaining_energy = np.zeros(intervals_count, dtype=npfloat)
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def load_data(
    fuel_instance: Fuel_InstanceType,
    annual_limit: nbfloat[:],
) -> None:
    """
    Load the c annual constraint data to the Fuel instance. This is done before solving a Scenario.

    Parameters:
    -------
    fuel_instance (Fuel_InstanceType): An instance of the Fuel jitclass.
    annual_limits (nbfloat[:]): Array containing the annual limits for a flexible Fuel.
        Each element provides the maximum annual generation (GWh) for a given year for the Fuel.

    Returns:
    -------
    None.
        scalar value to allow for variable time step simplified balancing methods to be developed in future.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the Fuel instance: data_status, annual_limit.
    Attributes modified for the referenced Fuel.line: lt_flows.
    Attributes modified for the referenced Generator.node: residual_load.
    """
    fuel_instance.annual_limit = annual_limit
    fuel_instance.data_status = True

    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def unload_data(fuel_instance: Fuel_InstanceType) -> None:
    """
    Unload the capacity factor trace and flexible annual constraint data from the Fuel instance. This is done
    after solving a Scenario to reduce memory usage.

    Parameters:
    -------
    fuel_instance (Fuel_InstanceType): An instance of the Fuel jitclass.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the Fuel instance: data_status, annual_limit.
    """
    fuel_instance.annual_limit = np.empty((0,), dtype=npfloat)
    fuel_instance.data_status = False
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def get_data(
    fuel_instance: Fuel_InstanceType,
    data_type: unicode_type,
) -> nbfloat[:]:
    """
    Gets the specified data_type from the Fuel instance.

    Parameters:
    -------
    fuel_instance (Fuel_InstanceType): An instance of the Fuel jitclass.
    data_type (unicode_type): String associated with the data array.

    Returns:
    -------
    nbfloat[:]: The data array associated with data_type.

    Raises:
    -------
    RuntimeError: Raised if data_status is False or if data_type does not correspond
        to any data arrays for the Fuel jitclass.
    """
    if not fuel_instance.data_status:
        raise_getting_unloaded_data_error()

    if data_type == "annual_limit":
        return fuel_instance.annual_limit
    else:
        raise RuntimeError("Invalid data_type argument for Fuel.get_data(data_type).")
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def get_annual_limit(
    fuel_instance: Fuel_InstanceType,
    year: nbintp,
) -> None:
    return get_data(fuel_instance, "annual_limit")[year]


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def initialise_annual_limits(fuel_instance: Fuel_InstanceType, year: int, first_t: int) -> None:
    """
    Initialises the remaining_energy attribute of the Fuel jitclass to the annual limit for the
    current year at the start of each year.

    Parameters:
    -------
    fuel_instance (Fuel_InstanceType): A dynamic instance of the Fuel jitclass.
    year (int): The current year in the scenario.
    first_t (int): The index of the first time interval in the current year.
    """
    if fuel_instance.static_instance:
        raise_static_modification_error()
    if len(fuel_instance.annual_limit) > 0:
        fuel_instance.remaining_energy[first_t - 1] = fuel_instance.annual_limit[year]
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def assign_trickling_reserves(fuel_instance: Fuel_InstanceType) -> None:
    fuel_instance.trickling_reserves = (
        fuel_instance.deficit_block_max_energy - fuel_instance.deficit_block_min_energy
    )
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def update_deficit_block_bounds(
    fuel_instance: Fuel_InstanceType,
    remaining_energy: nbfloat,
) -> None:
    """
    Update the temporary minimum and maximum remaining energy values for the flexible Generator in the
    deficit block. These values are updated in each time interval for the deficit block. The minimum
    and maximum remaining energies are used to define the trickling reserves that must be retained in
    the precharging period leading up to the deficit block such that the Generator is capable of dispatching
    during the deficit block.

    Parameters:
    -------
    fuel_instance (Fuel_InstanceType): An instance of the Fuel jitclass.
    remaining_energy (nbfloat): The remaining energy in a time interval that a flexible Generator has
        available for the calendar year such that it complies with its annual generation constraint.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the Fuel instance: deficit_block_max_energy, deficit_block_min_energy.
    """
    fuel_instance.deficit_block_min_energy = min(fuel_instance.deficit_block_min_energy, remaining_energy)
    fuel_instance.deficit_block_max_energy = max(fuel_instance.deficit_block_max_energy, remaining_energy)
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def initialise_deficit_block(
    fuel_instance: Fuel_InstanceType,
    interval: nbintp,
) -> None:
    """
    Upon resolving a deficit block, initialise the temporary remaining energy,
    max remaining energy, and min remaining energy values for a flexible Generator. These temporary
    variables are updated while performing unit committment in the reverse time direction for each time interval
    in the deficit block.

    Parameters:
    -------
    fuel_instance (Fuel_InstanceType): An instance of the Fuel jitclass.
    interval (nbintp): Index for the first time interval immediately following the deficit block.
        During unit committment for the deficit block, time intervals will decrease in value (reverse
        time).

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the Generator instance: remaining_energy_temp_reverse, deficit_block_max_energy,
        deficit_block_min_energy.
    """
    fuel_instance.remaining_energy_temp_reverse = fuel_instance.remaining_energy[interval - 1]
    fuel_instance.deficit_block_max_energy = fuel_instance.remaining_energy_temp_reverse
    fuel_instance.deficit_block_min_energy = fuel_instance.remaining_energy_temp_reverse


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def initialise_precharging_flags(
    fuel_instance: Fuel_InstanceType,
    interval: nbintp,
) -> None:
    """
    Initialises the trickling flag for a flexible Generator once precharging in the lead-up to the deficit
    block begins. The trickling flag is True if the flexible Generator has sufficient energy remaining for
    the calendar year such that it still retains the trickling reserves required to dispatch in the subsequent
    deficit block. When the trickling flag is True, a flexible Generator is assumed to be available for
    trickle charging a Storage precharger.

    Parameters:
    -------
    fuel_instance (Fuel_InstanceType): An instance of the Fuel jitclass.
    interval (nbintp): Index for the first time interval of the deficit block (immediately following the
        precharging period). Time intervals during the precharging period will decrease in value (reverse
        time).

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the Generator instance: trickling_flag.
    """
    fuel_instance.trickling_flag = (
        fuel_instance.remaining_energy[interval] - fuel_instance.trickling_reserves > TOLERANCE
    )
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def update_precharging_flags(
    fuel_instance: Fuel_InstanceType,
    interval: nbintp,
) -> None:
    """
    At the start of a time interval within the precharging period, the remaining trickling reserves and
    trickling flag for the flexible Fuel is updated. The remaining trickling reserves define the
    amount of energy available for trickle charging, ensuring that the Fuel retains sufficient
    reserves to dispatch during the deficit block immediately after the precharging period.

    The trickling flag is True if the flexible Fuel has sufficient energy remaining for
    the calendar year such that it still retains the trickling reserves required to dispatch in the subsequent
    deficit block. When the trickling flag is True, a flexible Fuel is assumed to be available for
    trickle charging a Storage precharger.

    Parameters:
    -------
    fuel_instance (Fuel_InstanceType): An instance of the Fuel jitclass.
    interval (nbintp): Index for the current time interval in the precharging period. Time intervals during
        the precharging period will decrease in value (reverse time).

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the Fuel instance: remaining_trickling_reserves, trickling_flag.
    """
    fuel_instance.remaining_trickling_reserves = max(
        fuel_instance.remaining_energy[interval] - fuel_instance.trickling_reserves, 0.0
    )
    fuel_instance.trickling_flag = (
        fuel_instance.remaining_trickling_reserves > TOLERANCE
    ) and fuel_instance.trickling_flag

    return None
