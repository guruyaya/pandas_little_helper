# -*- coding: utf-8 -*-

"""This module contains a list of transformers I commanly used accross my Data Sciense projects.

Todo:
    * Create a command line tool
    * Create tests for this module
    * Add some more transformers
"""
from typing import List, Union, Dict, Any, Callable
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from pandas_little_helper.helpers import get_list_if_single_entity, EncoderFunction

# NOTE: ignoring type hints are all around this file to handle Series, as they don't work
# well on 3.8. Expecting future improvments on that matter
# BaseEstimator, TransformerMixin dont have stubs yet.
class TargetEncoder(BaseEstimator, TransformerMixin): # type: ignore
    """ This transformer is designed to handle categorical data. It gets a dataframe or numpy,
    using grouping, it applies a function on the whole group, and keeps the answer. This usually
    uses mean function on the target, allthogh this function is designed to handle veraity of tasks

    possible preinstalled target functions:
        mean: the mean of the target.
        median: the median of the target.
        std: standart deviation of the target.
        size: number of elements in group. Fallback on missing group is 0
        sum: sum of target. Fallback on missing group is 0
        min: minimum of this group.
        max: maximum of this group.
        mean_plus_std: mean plus the standart deviation.
        mean_minus_std: mean minus the standart deviation.
    Args:
        group_cols (str or list of str): the columns used for grouping.
                    if a list is used, this function will use multiindex
        target_functions (list of strings or EncoderFunction): the function to be applied on the
                    target. default: 'mean'
        target_col_y (bool): the target col is in the y column. If false target col is X
                    (future implementaion)
        target_col_name (str): applies if y is a dataframe or if target col is in X. Name of the
                    col as target. Default to None (future implementaion)
        fallback_to_na (bool): instead of running the fallback function, return na default False
        fill_na: Either None of series that may contain nulls. If not None, will return all values,
                and only fill in nulls
    Attributes:
        possible_target_functions: dict of possible target functions
    """
    # TODO: Callable should not accept and return any. mypy complains # pylint: disable=fixme
    return_0: Callable[[pd.Series], Any] = lambda df: 0 # type: ignore
    get_mean: Callable[[pd.Series], Any] = lambda df: df.mean() # type: ignore
    get_median: Callable[[pd.Series], Any] =  lambda df: df.median() # type: ignore
    get_std: Callable[[pd.Series], Any] =  lambda df: np.std(df.to_numpy()) # type: ignore
    get_min: Callable[[pd.Series], Any] =  lambda df: df.min() # type: ignore
    get_max: Callable[[pd.Series], Any] =  lambda df: df.max() # type: ignore
    get_size: Callable[[pd.Series], Any] =  lambda df: df.shape[0] # type: ignore
    get_sum: Callable[[pd.Series], Any] =  lambda df: df.sum() # type: ignore
    get_mean_plus_std: Callable[[pd.Series], Any] = (  # type: ignore
        lambda df: np.mean(df.to_numpy()) + np.std(df.to_numpy())
    )
    get_mean_minus_std: Callable[[pd.Series], Any] = ( # type: ignore
        lambda df: np.mean(df.to_numpy()) - np.std(df.to_numpy())
    )

    possible_target_functions: Dict[str, EncoderFunction] = {
        'mean': EncoderFunction('mean', get_mean),
        'median': EncoderFunction('median', get_median),
        'std': EncoderFunction('std', get_std),
        'min': EncoderFunction('min', get_min),
        'max': EncoderFunction('max', get_max),
        'size': EncoderFunction('size', get_size, return_0),
        'sum': EncoderFunction('sum', get_sum, return_0),
        'mean_plus_std': EncoderFunction('mean_plus_std', get_mean_plus_std),
        'mean_minus_std': EncoderFunction('mean_minus_std', get_mean_minus_std),
    }

    found_results_: Dict[str, Any]
    missing_groups_results_: Dict[str, Any]

    def __init__(self,
                group_cols: Union[str, List[str]],
                target_functions: Union[None, List[ Union[EncoderFunction, str] ]] = None,
                target_col_y: bool = True,
                target_col_name: Union[None, str] = None,
                fallback_to_na: bool = False,
                fill_na: Union[None, pd.Series] = None): # type: ignore

        super().__init__()
        self.group_cols = group_cols
        self.target_functions = target_functions
        self.target_col_y = target_col_y
        self.target_col_name = target_col_name
        self.fallback_to_na = fallback_to_na
        self.fill_na = fill_na

        self._applied_target_functions: List[Union[str, EncoderFunction]] = \
            ['mean', ] if self.target_functions is None else self.target_functions

    def _replace_strings_with_encoder_function(self,
        function_list: List[ Union[EncoderFunction, str] ]
    ) -> List[EncoderFunction]:
        """ Replaces the strings with preinstalled functions """
        out: List[EncoderFunction] = []
        for func in function_list:
            if isinstance(func, str):
                out.append( self.possible_target_functions[ func ] )
            else:
                out.append(func)
        return out

    # This will become something else when sklearn add type hints
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Any: # type: ignore
        """ Fitting function for model
            X: The DataFrame / Numpy array used to extract data
            y: The target of the query. Note: doesn't have to be the target of this model
        """
        if len(X) != len(y):
            raise BaseException('X size is {} and y size is {}'.format(len(X), len(y)))

        X_copy:pd.DataFrame = X.copy()
        y_copy:pd.Series = y.copy() # type: ignore

        target_functions: List[ EncoderFunction ] = \
            self._replace_strings_with_encoder_function(self._applied_target_functions)

        # convert group cols into string to avoid mypy error
        group_cols = get_list_if_single_entity(self.group_cols)
        group_cols = [str(col) for col in group_cols]

        # mypy claims there's no groupby in Series. Using "type: ignore" to avoid errors
        # TODO: Check when pandas updrages, remove data-sciense-types # pylint: disable=fixme
        groups_y = y_copy.groupby([X_copy[col] for col in group_cols]) # type: ignore

        self.found_results_ = {}
        self.missing_groups_results_ = {}
        for tfunc in target_functions:
            self.found_results_[tfunc.name] = groups_y.apply(tfunc.func)
            if not self.fallback_to_na:
                self.missing_groups_results_[tfunc.name] = tfunc.fallback_func(y_copy)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """ gets the data and transforms it into prepared data
        """
        if not hasattr(self,'found_results_'):
            raise NotFittedError()

        group_cols = get_list_if_single_entity(self.group_cols)
        # This is a bug in mypy. It expcted an Index. However because I pass a list,
        # this will definatly be a MultiIndex. TODO: Check if future pandas fixed this one
        X_copy_index: pd.MultiIndex = X.set_index(group_cols).index # type: ignore
        out = pd.DataFrame([], columns=list(self.found_results_.keys()))

        for function_name in self.found_results_:
            out[function_name] = X_copy_index.map(self.found_results_[function_name])
            if not self.fallback_to_na:
                out[function_name] = out[function_name].fillna(
                    self.missing_groups_results_[function_name])

        return out.to_numpy()

    def get_feature_names(self) -> List[str]:
        """ standart sklearn returning of params
        """
        return list(self.found_results_.keys())
