from typing import List
from unittest import TestCase, skip
import numpy as np
import os
import pandas as pd

import dibs


########################################################################################################################

single_test_file_location = dibs.config.DEFAULT_PIPELINE__PRIME__CSV_TEST_FILE_PATH


class TestCheckArg(TestCase):

    # ensure_numpy_arrays_are_same_shape
    def test__ensure_numpy_arrays_are_same_shape__shouldRunWithoutError__whenArraysAreSameShape(self):
        data1 = [[1, 2, 3], [1, 2, 3]]
        data2 = [[1, 2, 3], [1, 2, 3]]
        arr1 = np.array(data1)
        arr2 = np.array(data2)
        dibs.check_arg.ensure_numpy_arrays_are_same_shape(arr1, arr2)

    def test__ensure_numpy_arrays_are_same_shape__shouldRunWithoutError__whenOnlyOneArraySubmitted(self):
        data1 = [[1, 2, 3], [1, 2, 3]]
        arr1 = np.array(data1)
        try:
            dibs.check_arg.ensure_numpy_arrays_are_same_shape(arr1)
            self.assertTrue(True)
        except TypeError as te:
            raise te

    def test__ensure_numpy_arrays_are_same_shape__ShouldErrorOut__whenArraysDifferentShapes(self):
        # Arrange
        data1 = [[1, 2, 3], [1, 2, 3]]
        data2 = [[1, 2, 3, 4], [1, 2, 3, 4]]
        expected_error = ValueError

        arr1 = np.array(data1)
        arr2 = np.array(data2)
        func = dibs.check_arg.ensure_numpy_arrays_are_same_shape

        # Act/Assert
        self.assertRaises(expected_error, func, arr1, arr2)

    def test__ensure_numpy_arrays_are_same_shape__shouldErrorOut__whenOneInputIsNotAnArray(self):
        data1 = [[1, 2, 3], [1, 2, 3]]
        data2 = [[1, 2, 3, 4], [1, 2, 3, 4]]
        arr1 = np.array(data1)
        list2 = data2
        expected_err = TypeError
        func = dibs.check_arg.ensure_numpy_arrays_are_same_shape

        self.assertRaises(expected_err, func, arr1, list2)

    # Ensure type
    def test__ensure_type__shouldRunWithoutError__whenGivenSingularCorrectExpectedType(self):
        # Arrange
        integer_var = 1
        expected_type = int
        # Act, Assert
        try:
            dibs.check_arg.ensure_type(integer_var, expected_type)
            self.assertTrue(True)  # TODO: low: is this line necessary for test to pass?
        except TypeError as te:
            raise te

    def test__ensure_type__shouldRunWithoutError__whenGivenMultipleCorrectExpectedTypes(self):
        # Arrange
        integer_var = 1
        expected_types_tuple = (int, float)
        # Act, Assert
        try:
            dibs.check_arg.ensure_type(integer_var, expected_types_tuple)
            self.assertTrue(True)  # TODO: low: is this line necessary for test to pass?
        except TypeError as te:
            raise te

    def test__ensure_type__shouldRunWithoutError__whenGivenMultipleCorrectExpectedTypesAsStarArgs(self):
        # Arrange
        integer_var = 1
        expected_types_tuple = (int, float)
        # Act, Assert
        try:
            dibs.check_arg.ensure_type(integer_var, *expected_types_tuple)
            self.assertTrue(True)  # TODO: low: is this line necessary for test to pass?
        except TypeError as te:
            raise te

    def test__ensure_type__shouldProduceError__whenGivenSingularIncorrectExpectedType(self):
        # Arrange
        integer_var = 1
        expected_type = float

        self.assertRaises(TypeError, dibs.check_arg.ensure_type, integer_var, expected_type)

    def test__ensure_type__shouldProduceError__whenGivenMultipleIncorrectExpectedTypes(self):
        # Arrange
        integer_var = 1
        expected_type = (float, str)

        self.assertRaises(TypeError, dibs.check_arg.ensure_type, integer_var, expected_type)

    def test__ensure_type__shouldProduceError__whenGivenSingularIncorrectExpectedTypeAsStarArgs(self):
        # Arrange
        integer_var = 1
        expected_type = (float, str)

        self.assertRaises(TypeError, dibs.check_arg.ensure_type, integer_var, *expected_type)


    def test__(self):
        # Arrange

        # Act

        # Assert

        self.assertEqual(None, None)
        # self.assertRaises()

