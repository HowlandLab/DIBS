from unittest import TestCase, skip
import numpy as np
import os
import pandas as pd

import dibs


test_file_name = dibs.config.DEFAULT_PIPELINE__PRIME__CSV_TEST_FILE_PATH


class TestConfig(TestCase):

    def test__get_data_source_from_file_path(self):
        # Arrange
        name = expected_output = 'MyPipeline'
        input_path = f'/usr/home/pipelines/{name}.pipeline'

        # Act
        actual_output = dibs.config.get_data_source_from_file_path(input_path)

        # Assert
        self.assertEqual(expected_output, actual_output)

    def test__get_data_source_from_file_path__2(self):
        # Arrange
        name = expected_output = 'MyPipeline'
        input_path = f'C:\\Users\\MyUser\\DIBS\\{name}.pipeline'

        # Act
        actual_output = dibs.config.get_data_source_from_file_path(input_path)

        # Assert
        self.assertEqual(expected_output, actual_output)

