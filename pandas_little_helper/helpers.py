# -*- coding: utf-8 -*-
""" This module contains useful structures and methods
"""
from typing import List, Union, overload, Any, Callable
import pandas as pd


AllValidDatatypes = Union[ pd.Series ] # At this point there's only series supprt
ValidDatatypeFunc = Callable[[AllValidDatatypes], AllValidDatatypes]
NoneOrCallable = Union[ ValidDatatypeFunc, None ]

@overload
def get_list_if_single_entity(entity: Union[str,
            Callable[[AllValidDatatypes], AllValidDatatypes], int, None]) -> List:
    ...
@overload
def get_list_if_single_entity(entity: List[Any]) -> List[Any]:
    ...
def get_list_if_single_entity(entity: Any) -> List[Any]:
    """ if a lone entity is passed, steamline into a list
    """
    if isinstance(entity, list):
        return entity.copy()
    return [entity]


class EncoderFunction:
    """ This class handles the way encoders handle methods.
    """
    def __init__(self, name: str, func: ValidDatatypeFunc,
                fallback_func: NoneOrCallable=None):
        self.name: str = name
        self.func: ValidDatatypeFunc = func
        self.fallback_func: ValidDatatypeFunc = func if fallback_func is None else fallback_func

    def run(self, data: AllValidDatatypes) -> AllValidDatatypes:
        """ Run target function """
        return self.func(data)

    def run_fallback(self, data: AllValidDatatypes) -> AllValidDatatypes:
        """ Run fallback function """
        return self.fallback_func(data)
