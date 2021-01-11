from unittest import TestCase, skip
import itertools
import numpy as np
import pandas as pd

import dibs


########################################################################################################################

single_test_file_location = dibs.config.DEFAULT_PIPELINE__PRIME__CSV_TEST_FILE_PATH


####################

class TestFeatureEngineering(TestCase):

    def test__adaptively_filter_dlc_output__shouldReturnSameNumberOfRowsAsInput__always(self):
        # Arrange
        df_input = dibs.io.read_csv(single_test_file_location, nrows=dibs.config.max_rows_to_read_in_from_csv)
        input_num_rows = len(df_input)
        # Act
        df_output, _ = dibs.feature_engineering.adaptively_filter_dlc_output(df_input)
        output_num_rows = len(df_output)

        # Assert
        err_msg = f"""
{df_input.to_string()}

{df_output.to_string()}
TODO: improve error message
""".strip()
        self.assertEqual(input_num_rows, output_num_rows, err_msg)

    def test__average_distance_between_n_features__shouldCalculateAverageLocationOfFeature__whenOneArraySubmitted(self):
        # TODO:
        # Arrange
        data = [[1, 2, 3],
                [4, 5, 6], ]
        arr_input = np.array(data)
        arr_expected_output = np.array(data)
        # Act
        arr_actual_output = dibs.feature_engineering.average_vector_between_n_vectors(arr_input)
        # Assert
        is_equal = (arr_expected_output == arr_actual_output).all()
        err = f"""
Expected output:
{arr_expected_output}

Actual output:
{arr_actual_output}
""".strip()  # TODO: elaborate error message to suss out potential problems
        self.assertTrue(is_equal, err)

    def test__average_distance_between_n_features__shouldCalculateAverageLocationOfFeature__whenTwoArraysSubmitted(self):
        # TODO: finish off this test, then remove this TODO if passes.
        # Arrange
        data_1 = [[0, 2],
                  [2, 2], ]
        data_2 = [[5, 2],
                  [1, 1], ]
        data_expected_output = [[2.5, 2.0],
                                [1.5, 1.5], ]
        arr_input_1 = np.array(data_1)
        arr_input_2 = np.array(data_2)
        arr_expected_output = np.array(data_expected_output)
        # Act
        arr_actual_output = dibs.feature_engineering.average_vector_between_n_vectors(arr_input_1, arr_input_2)
        # Assert
        is_equal = (arr_expected_output == arr_actual_output).all()
        err = f"""
Expected output:
{arr_expected_output}

Actual output:
{arr_actual_output}

"""  # TODO: elaborate error message to suss out potential problems
        self.assertTrue(is_equal, err)

    @skip  # TODO: Temporarily skipped while the test is being finished. It's not finished!
    def test__average_distance_between_n_features__shouldCalculateAverageLocationOfFeature__whenThreeArraysSubmitted(self):
        # TODO: finish the 3rd data set and also the expected data output
        # Arrange
        data_1 = [[0, 2],
                  [2, 2], ]
        data_2 = [[5, 2],
                  [1, 1], ]
        data_3 = [[],  # TODO 1/2
                  [], ]
        data_expected_output = [[2.5, 2.],
                                [1.5, 1.5],
                                [], ]  # TODO 2/2
        arr_input_1 = np.array(data_1)
        arr_input_2 = np.array(data_2)
        arr_input_3 = np.array(data_3)
        arr_expected_output = np.array(data_expected_output)
        # Act
        arr_actual_output = dibs.feature_engineering.average_vector_between_n_vectors(arr_input_1, arr_input_2, arr_input_3)
        # Assert
        is_equal = (arr_expected_output == arr_actual_output).all()
        err = f"""
Expected output:
{arr_expected_output}

Actual output:
{arr_actual_output}
""".strip()  # TODO: elaborate error message to suss out potential problems
        self.assertTrue(is_equal, err)

    def test__distance_between_2_arrays(self):
        # Arrange
        data_1 = [[5, 2, 3], ]
        arr_1 = np.array(data_1)
        data_2 = [[20, 15.5, 7], ]
        arr_2 = np.array(data_2)
        expected_output_distance: float = 20.573040611440984

        # Act
        actual_output_distance: float = dibs.feature_engineering.distance_between_two_arrays(arr_1, arr_2)

        # Assert
        err_msg = f"""
expected output: {expected_output_distance}

actual output: {actual_output_distance}
""".strip()
        self.assertEquals(expected_output_distance, actual_output_distance, err_msg)

    def test__average_arr_location(self):
        # Arrange
        data_1 = [[5., 2., 3.], ]
        arr_1 = np.array(data_1)
        data_2 = [[20., 15.5, 7.], ]
        arr_2 = np.array(data_2)
        exp_data = [[(5+20)/2., (2+15.5)/2, (3+7)/2], ]
        expected_output_arr = np.array(exp_data)

        # Act
        actual_output_arr = dibs.feature_engineering.average_vector_between_n_vectors(arr_1, arr_2)

        # Assert
        is_equals = (expected_output_arr == actual_output_arr).all()
        err_msg = f"""
expected_output_arr output: {expected_output_arr}

actual actual_output_arr: {actual_output_arr}
""".strip()
        self.assertTrue(is_equals, err_msg)

    def test__attach_average_feature_xy__shouldOnlyAttach2ColumnsInResult(self):
        """ Test that the number of output columns matches expected """
        # Arrange
        df = dibs.io.read_csv(single_test_file_location)
        df_cols_set = set(df.columns)
        output_feature_name = 'AvgFeature'
        expected_num_cols: int = len(df_cols_set) + 2

        # (Ensure col names exist for test)
        feature1, feature2 = 'Forepaw/Shoulder1', 'Forepaw/Shoulder2'
        for body_part, xy in itertools.product((feature1, feature2), ('x', 'y')):
            feat_xy = f'{body_part}_{xy}'
            assert feat_xy in df_cols_set, f'Column "{feat_xy}" not found in DataFrame. Cannot complete test. Columns = {list(df.columns)}'

        # Act
        df_output: pd.DataFrame = dibs.feature_engineering.attach_average_bodypart_xy(df, feature1, feature2, output_feature_name, copy=True)
        actual_num_output_cols: int = len(df_output.columns)

        # Assert
        self.assertEqual(expected_num_cols, actual_num_output_cols)



