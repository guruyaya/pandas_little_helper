import unittest
import sys
import pathlib
import pandas as pd
import numpy as np

parent_path = str( pathlib.Path(__file__).parent.parent.absolute() )

sys.path.append(parent_path)

from pandas_little_helper.transformers import (
    TargetEncoder
)

first_train_X = pd.DataFrame({
    'group1': ['A', 'A', 'A', 'B', 'C', 'B', 'D', 'D'],
    'group2': ['1', '1', '2', '3', '2', '1', '4', '2'],
    'group3': ['X', 'X', 'X', 'X', 'Y', 'Y', 'Y', 'Y'],
    'value1': [10, 20, 10, 30, 60, 60, 100, 10]
})
first_train_y = pd.Series(
    [7, 15, 2, 11, 77, 90, 109, 1]
)

first_test_X = pd.DataFrame({
    'group1': ['A', 'A', 'C', 'C', 'F'],
    'group2': ['1', '3', '2', '2', '3'],
    'group3': ['X', 'X', 'X', 'Y', 'Y'],
    'value1': [10, 20, 10, 30, 90]
})
fail1_train_y = pd.Series(
    [102, 9, 15, 1, 11, 77, 90, 109, 1] # too many on train group
)


class TestTargetEncoder(unittest.TestCase):
    def test_no_params(self):
        te = TargetEncoder('group1')
        train_transformed = te.fit_transform(first_train_X, first_train_y)
        expecting_found_results = {'A': 8.0, 'B': 50.5, 'C': 77.0, 'D': 55.0}
        for key, value in expecting_found_results.items():
            self.assertEqual(value, te.found_results_['mean'].loc[key],
                'On key {}'.format(key))

        expecting_missing_results = {'mean': 39.0}
        self.assertEqual(expecting_missing_results['mean'], te.missing_groups_results_['mean'])

        expecting_train_transformed = [8., 8., 8., 50.5,77., 50.5, 55., 55.]
        for i, val in enumerate(expecting_train_transformed):
            self.assertAlmostEqual(train_transformed.item(i,0), val, msg=f'Item number {i}')

        test_transformed = te.transform(first_test_X)
        expected_test_transformed = [8., 8., 77., 77.0, 39.0]
        for i, val in enumerate(expected_test_transformed):
            self.assertAlmostEqual(test_transformed.item(i,0), val, msg=f'Item number {i}')

        with self.assertRaises(BaseException):
            te = TargetEncoder('group1')
            te.fit(first_train_X, fail1_train_y)

    def test_no_params_multiindex(self):
        te = TargetEncoder(['group1', 'group2'])
        train_transformed = te.fit_transform(first_train_X, first_train_y)
        expecting_found_results = {('A','1'): 11.0, ('A','2'): 2.0, ('B','1'): 90.0,
                                    ('B','3'): 11.0, ('C','2'): 77.0, ('D','2'): 1.0,}
        for key, value in expecting_found_results.items():
            self.assertEqual(value, te.found_results_['mean'].loc[key],
                'On key {}'.format(key))

        expecting_train_transformed = [11., 11., 2., 11.,77., 90., 109., 1.]
        for i, val in enumerate(expecting_train_transformed):
            self.assertAlmostEqual(train_transformed.item(i,0), val, msg=f'Item number {i}')

        test_transformed = te.transform(first_test_X)
        expected_test_transformed = [11., 39., 77., 77.0, 39.0]
        for i, val in enumerate(expected_test_transformed):
            self.assertAlmostEqual(test_transformed.item(i,0), val, msg=f'Item number {i}')

    def test_with_all_named_functions(self):
        functions = list( TargetEncoder.possible_target_functions.keys() )
        te = TargetEncoder('group1', functions)
        train_transformed = te.fit_transform(first_train_X, first_train_y)
        expecting_found_results = {
            'mean': {'A': 8.0, 'B': 50.5, 'C': 77.0, 'D': 55.0},
            'median': {'A': 7.0, 'B': 50.5, 'C': 77.0, 'D': 55.0},
            'size': {'A': 3, 'B': 2, 'C': 1, 'D': 2},
            'sum': {'A': 24, 'B': 101, 'C': 77, 'D': 110},
            'std': {'A': 5.354, 'B': 39.5, 'C': 0.0, 'D': 54},
            'min': {'A': 2, 'B': 11, 'C': 77.0, 'D': 1},
            'max': {'A': 15, 'B': 90, 'C': 77.0, 'D': 109},
            'mean_plus_std': {'A': 13.354, 'B': 90, 'C': 77.0, 'D': 109},
            'mean_minus_std': {'A': 2.646, 'B': 11, 'C': 77.0, 'D': 1}
        }

        for function_name in functions:
            values_list = expecting_found_results[function_name]
            for key, value in values_list.items():
                self.assertAlmostEqual(value, te.found_results_[function_name].loc[key],3,
                    msg='On funcion {} key {}'.format(function_name, key))

        expecting_missing_results = {'mean': 39.0, 'median': 13.0, 'std': 42.0446, 'sum': 0.,
            'min': 1.0, 'max': 109.0, 'size': 0.0, 'mean_plus_std': 81.045, 'mean_minus_std': -3.045}

        for function_name in functions:
            value = expecting_missing_results[function_name]
            self.assertAlmostEqual(value, te.missing_groups_results_[function_name], 3, 
            msg=f"On function {function_name}")

        expected_test_transformed = [
            [8.0, 7.0, 5.354, 2, 15, 3, 24., 13.354, 2.646],
            [8.0, 7.0, 5.354, 2, 15, 3, 24., 13.354, 2.646],
            [77.,77.,0.,77.,77.,1.,77.,77.,77.,],
            [77.,77.,0.,77.,77.,1.,77.,77.,77.,],
            [39.0, 13.0, 42.0446, 1.0, 109.0, 0.0, 0.0, 81.045, -3.045],
        ]

        test_transformed = te.transform(first_test_X)
        for i, val_list in enumerate(expected_test_transformed):
            for fnum, function_name in enumerate(functions):
                val = val_list[fnum]
                self.assertAlmostEqual(test_transformed.item(i,fnum), val, 3,
                    msg=f'Item number {i} function: {function_name}')

    def test_with_custom_unnamed_functions(self):
        functions = [lambda s: np.quantile(s, 0.99), 'min', lambda s: np.max(s)]
        te = TargetEncoder('group1', functions)
        te.fit(first_train_X, first_train_y)

        expecting_found_results = {
            'function0': {'A': 14.84, 'B': 89.21, 'C': 77.0, 'D': 107.92},
            'min': {'A': 2, 'B': 11, 'C': 77.0, 'D': 1},
            'function2': {'A': 15, 'B': 90, 'C': 77.0, 'D': 109},
        }
        for function_name, values_list in expecting_found_results.items():
            for key, value in values_list.items():
                self.assertAlmostEqual(value, te.found_results_[function_name].loc[key],3,
                    msg='On funcion {} key {}'.format(function_name, key))

        expecting_missing_results = {'function0': 107.67, 'min': 1, 'function2': 109}
        for key, value in expecting_missing_results.items():
            self.assertAlmostEqual(value, te.missing_groups_results_[key], 3,
                    msg='On key {key}')

        features = te.get_feature_names()

        for i, key in enumerate(expecting_found_results):
            self.assertEqual(key, features[i])

    def test_with_custom_partly_named_function(self):
        functions = [lambda s: np.quantile(s, 0.99), 'min', lambda s: np.max(s)]
        functions_names = [None, None, 'MAXIMUM']
        te = TargetEncoder('group1', functions, functions_names)
        te.fit(first_train_X, first_train_y)
        expecting_found_results = {
            'function0': {'A': 14.84, 'B': 89.21, 'C': 77.0, 'D': 107.92},
            'min': {'A': 2, 'B': 11, 'C': 77.0, 'D': 1},
            'MAXIMUM': {'A': 15, 'B': 90, 'C': 77.0, 'D': 109},
        }
        for function_name, values_list in expecting_found_results.items():
            for key, value in values_list.items():
                self.assertAlmostEqual(value, te.found_results_[function_name].loc[key],3,
                    msg='On funcion {} key {}'.format(function_name, key))

        test_transformed = te.transform(first_test_X)
        expected_test_transformed =[
            [14.84, 2., 15., ],
            [14.84, 2., 15., ],
            [77., 77., 77., ],
            [77., 77., 77., ],
            [107.67, 1., 109., ]
        ]
        for i, val_list in enumerate(expected_test_transformed):
            for fnum, function_name in enumerate(functions):
                val = val_list[fnum]
                self.assertAlmostEqual(test_transformed.item(i,fnum), val, 3,
                    msg=f'Item number {i} function: {function_name}')

        features = te.get_feature_names()

        for i, key in enumerate(expecting_found_results):
            self.assertEqual(key, features[i])

        with self.assertRaises(BaseException):
            functions = [lambda s: np.quantile(s, 0.99), 'min', lambda s: np.max(s)]
            functions_names = [None, None, 'MAXIMUM', 'Banaan']
            te = TargetEncoder('group1', functions, functions_names)
            te.fit(first_train_X, first_train_y)

if __name__ == '__main__':
    unittest.main()
