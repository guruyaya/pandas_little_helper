from typing import List, Union, overload, Any, Callable
import pandas as pd

AllValidDatatypes: Any = Union[ pd.Series ] # At this point there's only series supprt
ValidDatatypeFunc: Any = Callable[[AllValidDatatypes], AllValidDatatypes]
NoneOrCallable: Any = Union[ ValidDatatypeFunc, None ]

@overload
def get_list_if_single_entity(entity: Union[str,
            Callable[[AllValidDatatypes], AllValidDatatypes], int, None]) -> List[Any]:
    ...
@overload
def get_list_if_single_entity(entity: List[Any]) -> List[Any]:
    ...

class EncoderFunction:
    name: str

    def __init__(self, name: str, func: ValidDatatypeFunc,
                fallback_func: NoneOrCallable=None): ...

    @staticmethod
    def func(s: AllValidDatatypes) -> float: ...

    @staticmethod
    def fallback_func(s: AllValidDatatypes) -> float: ...

    def run(self, data: AllValidDatatypes) -> AllValidDatatypes: ...
    def run_fallback(self, data: AllValidDatatypes) -> AllValidDatatypes: ...
