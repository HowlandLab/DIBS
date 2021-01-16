from unittest import TestCase, skip
import numpy as np
import os
import pandas as pd

import dibs


test_file_name = dibs.config.TEST_FILE__PipelineMimic__CSV__TRAIN_DATA_FILE_PATH


class TestConfig(TestCase):

    def test__get_data_source_from_file_path__shouldWorkAsExpected__whenGivenALinuxPath(self):
        # Arrange
        expected_output = 'MyPipeline'
        input_path = f'/usr/home/pipelines/{expected_output}.pipeline'

        # Act
        actual_output = dibs.config.get_data_source_from_file_path(input_path)

        # Assert
        self.assertEqual(expected_output, actual_output)

    def test__get_data_source_from_file_path__shouldWorkAsExpected__whenGivenAWindowsPath(self):
        # Arrange
        expected_output = 'MyPipeline'
        input_path = f'C:\\Users\\MyUser\\DIBS\\{expected_output}.pipeline'

        # Act
        actual_output = dibs.config.get_data_source_from_file_path(input_path)

        # Assert
        self.assertEqual(expected_output, actual_output)

