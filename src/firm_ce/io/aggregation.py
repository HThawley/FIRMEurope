import numpy as np


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
        if method == 'mean':
            return reshaped.mean(axis=1)
        elif method == 'sum':
            return reshaped.sum(axis=1)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    # Handle non-divisible traces
    full_part = trace[:n_full * factor].reshape(-1, factor)
    rem_part = trace[n_full * factor:]

    if method == 'mean':
        agg_full = full_part.mean(axis=1)
        agg_rem = np.array([rem_part.mean()])
        return np.concatenate((agg_full, agg_rem))
    elif method == 'sum':
        agg_full = full_part.sum(axis=1)
        # Scale the remainder sum to match the expected window length
        agg_rem = np.array([rem_part.sum() / remainder * factor])
        return np.concatenate((agg_full, agg_rem))
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
