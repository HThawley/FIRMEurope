from typing import List, Any


from numba import njit
import numpy as np
from numpy.typing import NDArray


def parse_comma_separated(value: str, lower: bool = True) -> List[str]:
    """
    Parse a comma-separated string into a list of trimmed, non-empty strings.

    Parameters:
    -------
    value (str): A string containing comma-separated values.
    lower (bool): A boolean indicating whether to format strings in lower case

    Returns:
    -------
    List[str]: A list of cleaned strings with whitespace removed and empty entries excluded.
        Optionally all in lower case.
    """
    if lower:
        return [item.strip().lower() for item in value.split(",") if item.strip()]
    return [item.strip() for item in value.split(",") if item.strip()]


def safe_divide(num: float, denom: float, fail: float = 0.0) -> float:
    """Safe division for calculating levelised costs when total dispatch energy from the asset is 0."""
    return num / denom if denom != 0 else fail


@njit
def safe_divide_array(
    num: NDArray[np.float64],
    denom: NDArray[np.float64],
    fail: np.float64 = 0.0,
) -> NDArray[np.float64]:
    """ Zero-safe division of two arrays. """
    retarr = num.copy().ravel()
    denom_ravel = denom.ravel()
    for i in range(retarr.size):
        if denom_ravel[i] == 0.0:
            retarr[i] = fail
        else:
            retarr[i] /= denom_ravel[i]
    return retarr.reshape(num.shape)


def safe_divide2(num: float, denom: Any, zero_fail: float = 0.0) -> Any:
    """Returns denom if denom is not numeric. Otherwise calls safe_divide"""
    if isinstance(denom, (float, int)):
        return safe_divide(num, denom, zero_fail)
    else:
        return denom
