"""
Numba types and the TypedDict and TypedList classes are overloaded, enabling
JIT to be switched off for debugging with the Python interpreter.
"""

from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

from firm_ce.common.constants import JIT_ENABLED, BITDEPTH

EvaluationRecord_Type = Tuple[str, str, float, float, float, NDArray[np.float64], NDArray[np.float32]]
BroadOptimumVars_Type = Tuple[int, str, bool, str]
BandCandidates_Type = Dict[str, Tuple[List[float], List[float]]]

if JIT_ENABLED:
    from numba.core.types import DictType, ListType, UniTuple, boolean, float64, float32, int64, int32, unicode_type
    from numba.typed.typeddict import Dict as TypedDict
    from numba.typed.typedlist import List as TypedList
else:

    def _make_mock_type(np_type):
        class MockType:
            def __new__(cls, value=0):
                return np_type(value)

            @classmethod
            def __class_getitem__(cls, key):
                return NDArray[np_type]

        MockType.__name__ = f"_{np_type.__name__.capitalize()}"
        return MockType

    int64 = _make_mock_type(np.int64)
    int32 = _make_mock_type(np.int32)
    float64 = _make_mock_type(np.float64)
    float32 = _make_mock_type(np.float32)
    boolean = _make_mock_type(np.bool_)
    unicode_type = _make_mock_type(np.str_)

    def UniTuple(ty, n: int):
        _map = {
            float64: float,
            float32: float,
            int64: int,
            int32: int,
            boolean: bool,
            unicode_type: str,
            float: float,
            int: int,
            bool: bool,
            str: str,
        }
        base = _map[ty]
        return Tuple[base, ...]

    def DictType(key_ty, val_ty):
        try:
            return Dict[key_ty, val_ty]
        except Exception:
            return Dict

    def ListType(val_ty):
        try:
            return List[val_ty]
        except Exception:
            return List

    class TypedDict(dict):
        def __init__(self, key_type=None, value_type=None):
            super().__init__()
            self.key_type = key_type
            self.value_type = value_type

        @staticmethod
        def empty(key_type=None, value_type=None):
            return TypedDict(key_type, value_type)

    class TypedList(list):
        def __init__(self, value_type=None):
            super().__init__()
            self.value_type = value_type

        @staticmethod
        def empty_list(value_type=None):
            return TypedList(value_type)

if BITDEPTH == 64:
    nbfloat = float64
    nbint = int64
    nbintp = int64

    npfloat = np.float64
    npint = np.int64
    npintp = np.int64

if BITDEPTH == 32:
    nbfloat = float32
    nbint = int32
    nbintp = int64  # native 64 bit indexing

    npfloat = np.float32
    npint = np.int32
    npintp = np.int64  # native 64 bit indexing
