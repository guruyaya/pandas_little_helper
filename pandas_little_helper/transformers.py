# -*- coding: utf-8 -*-

"""Example Google style docstrings.

This module contains a list of transformers I commanly used accross my Data Sciense projects.

Todo:
    * Create a command line tool
    * Create tests for this module
    * Add some more transformers
"""
from collections.abc import Callable
from typing import List, Union, Dict, Any, overload
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

@overload
def get_list_if_single_entity(entity: Union[str, Callable, int, None]) -> List:
    ...
@overload
def get_list_if_single_entity(entity: List) -> List:
    ...
def get_list_if_single_entity(entity) -> List:
    """ if a lone entity is passed, steamline into a list
    """
    if isinstance(entity, list):
        return entity.copy()
    return [entity]

class TargetEncoder(BaseEstimator, TransformerMixin):
    """ This transformer is designed to handle categorical data. It gets a dataframe or numpy,
    using grouping, it applies a function on the whole group, and keeps the answer. This usually
    uses mean function on the target, allthogh this function is designed to handle veraity of tasks

    Args:
        group_cols (str or list of str): the columns used for grouping.
                    if a list is used, this function will use multiindex
        target_functions (str, callable or list of either str or callables): the function to be
                    applied on the target. If str is passed, it has to be one of those suggested in
                    possible_target_functions:
                        mean: the mean of the target
                        median: the median of the target
                        std: standart deviation of the target
                        size: number of elements in group
                        min: minimum of this group
                        max: maximum of this group
                        mean_plus_std: mean plus the standart deviation
                        mean_minus_std: mean minus the standart deviation
        target_functions_names(str, list str): if names are needed for the target functions.
        This affects the get_param_names output. By default / if None is passed, it uses the
                    generic names given in the string. if Callable is passed without a name
                    it's called function_{i} where i is the location of the function in the list
    Attributes:
        possible_target_functions: dict of possible target functions


    """
    possible_target_functions: Dict['str', Callable] = {
        'mean': np.mean,
        'median': np.median,
        'std': np.std,
        'min': np.min,
        'max': np.max,
        'size': len,
        'mean_plus_std': lambda df: np.mean(df) + np.std(df),
        'mean_minus_std': lambda df: np.mean(df) - np.std(df),
    }

    found_results_: Dict[str, Any] = {}
    missing_groups_results_: Dict[str, Any] = {}

    def __init__(self,
                group_cols: Union[str, int, List[Union[str]]] ,
                target_functions: Union[Callable, str, List[ Union[Callable, str] ]]='mean',
                target_functions_names: Union[str, List[str]]=None,
                fill_in_name_format:str ='function{}'):

        super().__init__()
        self.group_cols = group_cols
        self.target_functions = target_functions
        self.target_functions_names = target_functions_names
        self.fill_in_name_format = fill_in_name_format

    def _fill_in_function_names(self, target_functions_names: List[str],
                        target_functions: List[Union[str, Callable]]) -> List[str]:
        """ fills in function name with original or fill_in
        """
        for i, name in enumerate(target_functions_names):
            if name is None:
                if isinstance(target_functions[i], str): # Fill in with original function name
                    target_functions_names[i] = str(target_functions[i]) # prevents mypy error
                else: # Probably Callable
                    target_functions_names[i] = self.fill_in_name_format.format(i)
        return target_functions_names

    def _fill_in_string_target_functions(self,
        target_functions: List[ Union[Callable, str]]) -> List[ Callable]:
        callables_list: List[Callable] = []
        for function in target_functions:
            if isinstance(function, str):
                try:
                    callables_list.append(self.possible_target_functions[function])
                except KeyError as e:
                    raise BaseException(f'{function} is not defined in possible functions') from e
            else:
                callables_list.append(function)
        return callables_list

    @staticmethod
    def _except_if_not_same_len(my_vars: Dict[str, Any]) -> bool:
        """ throws exception if all lists are not in the same lengh
        """
        iter_vars = iter(my_vars.items())
        first = next(iter_vars)
        first_key = first[0]
        first_len = len(first[1])

        for key, my_list in iter_vars:
            if len(my_list) != first_len:
                raise BaseException("{} length {} does not match {} length {}".format(
                    key, len(my_list), first_key, first_len
                ))
        return True

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """ Fitting function for model
            X: The DataFrame / Numpy array used to extract data
            y: The target of the query. Note: doesn't have to be the target of this model
        """
        if len(X) != len(y):
            raise BaseException('X size is {} and y size is {}'.format(len(X), len(y)))
        X_copy:pd.DataFrame = X.copy()
        y_copy = y.copy()

        target_functions = get_list_if_single_entity(self.target_functions)
        target_functions_names = get_list_if_single_entity(self.target_functions_names)

        if target_functions_names == [None,]: # In case No target function name was passed
            target_functions_names = [None,] * len(target_functions) # create None fill-ins for all

        self._except_if_not_same_len({'target_functions': target_functions,
                                    'target_functions_names': target_functions_names})

        # Filling in the Nones in names
        target_functions_names = \
            self._fill_in_function_names(target_functions_names, target_functions)

        # convert group cols into string to avoid mypy error
        group_cols = get_list_if_single_entity(self.group_cols)
        group_cols = [str(col) for col in group_cols]
        #mypy claims there's no groupby in Series. Using "type: ignore" to avoid errors
        groups_y = y_copy.groupby([X_copy[col] for col in group_cols]) # type: ignore
        target_functions_callables = self._fill_in_string_target_functions(target_functions)

        for name, func in zip(target_functions_names, target_functions_callables):
            self.found_results_[name] = groups_y.apply(func)

        for name, func in zip(target_functions_names, target_functions_callables):
            self.missing_groups_results_[name] = func(y)

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """ gets the data and transforms it into prepared data
        """
        group_cols = get_list_if_single_entity(self.group_cols)
        # This is a bug in mypy. It expcted an Index. However because I pass a list,
        # this will definatly be a MultiIndex
        X_copy_index: pd.MultiIndex = X.set_index(group_cols).index # type: ignore
        out = pd.DataFrame([], columns=list(self.found_results_.keys()))

        for function_name in self.found_results_:
            out[function_name] = X_copy_index.map(self.found_results_[function_name])
            out[function_name].fillna(self.missing_groups_results_[function_name])
        return out.to_numpy()

    def get_feature_names(self):
        """ standart sklearn returning of params
        """
        return list(self.found_results_.keys())
