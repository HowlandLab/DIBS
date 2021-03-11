from unittest import TestCase, skip
import itertools
import numpy as np
import pandas as pd

import dibs


def are_two_arrays_equal_including_possible_NaN_entries(a: np.ndarray, b: np.ndarray):
    try:
        np.testing.assert_equal(a, b)
    except AssertionError:
        return False
    return True


def anotherequalityimplementation(a, b):
    # TODO: lowest: confirm that this works
    return np.allclose(a, b, equal_nan=True)


########################################################################################################################

single_test_file_location = dibs.config.TEST_FILE__PipelineMimic__CSV__TRAIN_DATA_FILE_PATH


########################################################################################################################

class TestFeatureEngineering(TestCase):

    @skip  # TODO: Temporary skip due to performance issues
    def test__adaptively_filter_dlc_output__shouldReturnSameNumberOfRowsAsInput__always(self):
        # Arrange
        df_input = dibs.io.read_csv(single_test_file_location, nrows=dibs.config.max_rows_to_read_in_from_csv)
        input_num_rows = len(df_input)
        # Act
        df_output, _ = dibs.feature_engineering.adaptively_filter_dlc_output(df_input)
        output_num_rows = len(df_output)

        # Assert
        err_msg = f"""
Input:
{df_input.to_string()}

Output:
{df_output.to_string()}
""".strip()  # TODO: improve error message
        self.assertEqual(input_num_rows, output_num_rows, err_msg)

    # Array feature functions
    def test__average_vector_between_n_vectors__shouldCalculateAverageLocationOfFeature__whenOneArraySubmitted(self):
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

    def test__average_vector_between_n_vectors__shouldCalculateAverageLocationOfFeature__whenTwoArraysSubmitted(self):
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

    def test__distance_between_two_arrays__shouldWorkAsExpected__whenGivenTwoArrays(self):
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
        self.assertEqual(expected_output_distance, actual_output_distance, err_msg)

    def test__average_vector_between_n_vectors__shouldProduceSameOutputAsNapkinMath__whenSetUpAsSuch(self):
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

    def test__attach_average_bodypart_xy__shouldOnlyAttach2ColumnsInResult(self):
        """
        Test that the number of output columns matches expected output.
        Upon creating a new "average bodypart", it should add 2 extra columns, one for the x and one for the y values
        """
        # Arrange
        df = dibs.io.read_csv(single_test_file_location)
        df_cols_set = set(df.columns)
        output_feature_name = 'AvgFeature'
        expected_num_cols: int = len(df_cols_set) + 2

        # (Ensure col names exist for test)
        feature1, feature2 = 'ForepawLeft', 'ForepawRight'
        for body_part, xy in itertools.product((feature1, feature2), ('x', 'y')):
            feat_xy = f'{body_part}_{xy}'
            assert feat_xy in df_cols_set, f'Column "{feat_xy}" not found in DataFrame. Cannot complete test. Columns = {list(df.columns)}'

        # Act
        df_output: pd.DataFrame = dibs.feature_engineering.attach_average_bodypart_xy(df, feature1, feature2, output_feature_name, copy=True)
        actual_num_output_cols: int = len(df_output.columns)

        # Assert
        self.assertEqual(expected_num_cols, actual_num_output_cols)

    # delta angle
    def test__delta_angle__shouldReturnZero__whenNoChangeOccurs(self):
        # TODO: review test fundies
        # Arrange
        a_x, a_y = 1, 1
        b_x, b_y = 1, 1
        expected_output = 0.
        # Act
        actual_output = dibs.feature_engineering.delta_angle(a_x, a_y, b_x, b_y)
        # Assert
        self.assertEqual(expected_output, actual_output)

    def test__delta_angle__shouldReturnZero__whenPositionChangesButSameAngle(self):
        # TODO: review test fundies
        # Arrange
        x0, y0 = 1, 1
        x1, y1 = 100, 100
        expected_output = 0.
        # Act
        actual_output = dibs.feature_engineering.delta_angle(x0, y0, x1, y1)
        # Assert
        self.assertEqual(expected_output, actual_output)

    def test__delta_angle_given_angles__lazy__shouldBe180__whenObvious(self):
        """ Check lazy angle """
        # Arrange
        angle0 = 0
        angle1 = 180
        expected_output = 180.
        # Act
        actual_output = dibs.feature_engineering.delta_angle_given_angles__lazy(angle0, angle1)
        # Assert
        self.assertEqual(expected_output, actual_output)

    # delta angle using two body parts arrays data
    def test__delta_two_body_parts_angle_killian_try__shouldGiveZeroes__whenNoAngleChangesOccur(self):
        # Arrange
        data_xy_1 = [[1, 1],
                     [1, 1],
                     [1, 1], ]
        data_xy_2 = [[2, 2],
                     [2, 2],
                     [2, 2], ]
        data_for_expected_output = [np.NaN,
                                    0,
                                    0, ]
        arr_xy_1 = np.array(data_xy_1)
        arr_xy_2 = np.array(data_xy_2)
        expected_output = np.array(data_for_expected_output)
        # Act
        actual_output = dibs.feature_engineering.delta_two_body_parts_angle_killian_try(arr_xy_1, arr_xy_2)
        # and actual_output[0] != expected_output[0]
        self.assertTrue(actual_output[0] != expected_output[0], f'{actual_output[0]}, {expected_output[0]}')

        # Assert
        # is_equal = (expected_output == actual_output).all()
        is_equal = are_two_arrays_equal_including_possible_NaN_entries(expected_output, actual_output)
        err = f"""
Expected output = {expected_output}

Actual output   = {actual_output}
"""
        self.assertTrue(is_equal, err)

    def test__delta_two_body_parts_angle_killian_try__shouldGiveZeroes__whenNoAngleChangesOccurAndDifferentLocations(self):
        """
        TODO: When a single delta function is decided-upon,

        """
        # Arrange
        data_xy_1 = [
            [1, 1],
            [2, 2],
            [3, 3],
        ]
        data_xy_2 = [
            [0, 0],
            [1, 1],
            [2, 2],
        ]
        data_for_expected_output = [
            np.NaN,
            0,
            0,
        ]
        arr_xy_1 = np.array(data_xy_1)
        arr_xy_2 = np.array(data_xy_2)
        expected_output: np.ndarray = np.array(data_for_expected_output)
        expected_output_minus_first_row: np.ndarray = expected_output[1:]
        # Act
        actual_output: np.ndarray = dibs.feature_engineering.delta_two_body_parts_angle_killian_try(arr_xy_1, arr_xy_2)
        actual_output_minus_first_row: np.ndarray = actual_output[1:]
        # Assert

        is_equal = (expected_output_minus_first_row == actual_output_minus_first_row).all()
        err = f"""
    Expected output = {expected_output}

    Actual output   = {actual_output}
    """
        self.assertTrue(is_equal, err)

    def test__delta_two_body_parts_angle_killian_try__shouldGetZeroChange__whenCoordinatesDoNotMove(self):
        """
        TODO: When a single delta function is decided-upon,
        """
        # Arrange
        data_xy_1 = [
            [1, 1],
            [1, 1],
            [1, 1],
        ]
        arr_xy_1 = np.array(data_xy_1)
        data_xy_2 = [
            [2, 2],
            [2, 2],
            [2, 2],
        ]
        arr_xy_2 = np.array(data_xy_2)
        data_for_expected_output = [
            np.NaN,
            0,
            0,
        ]
        expected_output = np.array(data_for_expected_output)
        expected_output_minus_first_row = expected_output[1:]
        # Act
        actual_output = dibs.feature_engineering.delta_two_body_parts_angle_killian_try(arr_xy_1, arr_xy_2)
        actual_output_minus_first_row = actual_output[1:]
        self.assertTrue(actual_output[0] != expected_output[0], f'{actual_output[0]}, {expected_output[0]}')
        # Assert

        is_equal = (expected_output_minus_first_row == actual_output_minus_first_row).all()
        err = f"""
    Expected output = {expected_output}

    Actual output   = {actual_output}
    """
        self.assertTrue(is_equal, err)

    ### is_angle_change_positive #################################################################################
    def test__is_angle_change_positive__shouldBeTrue__whenSetUpAsSuchInQuadrant1(self):
        x0, y0 = 10, 0
        x1, y1 = 1, 1
        expected = True
        actual = dibs.feature_engineering.is_angle_change_positive(x0, y0, x1, y1)
        self.assertEqual(expected, actual)

    def test__is_angle_change_positive__shouldBeTrue__whenSetUpAsSuchInQuadrant2(self):
        x0, y0 = 0, 10
        x1, y1 = -1, 1
        expected = True
        actual = dibs.feature_engineering.is_angle_change_positive(x0, y0, x1, y1)
        self.assertEqual(expected, actual)

    def test__is_angle_change_positive__shouldBeTrue__whenSetUpAsSuchInQuadrant3(self):
        x0, y0 = -10, 0
        x1, y1 = -1, -1
        expected = True
        # Act
        actual = dibs.feature_engineering.is_angle_change_positive(x0, y0, x1, y1)
        # Assert
        angle = dibs.feature_engineering.delta_angle_between_two_vectors_starting_at_origin(x0, y0, x1, y1)
        err = f"""
        Angle = {angle}
        """
        self.assertEqual(expected, actual, err)

    def test__is_angle_change_positive__shouldBeTrue__whenSetUpAsSuchInQuadrant4(self):
        x0, y0 = 0, -10
        x1, y1 = 1, -0.5
        expected = True
        actual = dibs.feature_engineering.is_angle_change_positive(x0, y0, x1, y1)
        angle = dibs.feature_engineering.delta_angle_between_two_vectors_starting_at_origin(x0, y0, x1, y1)
        err = f"""
        Angle = {angle}
        """
        self.assertEqual(expected, actual, err)

    def test__is_angle_change_positive__shouldBeFalse__whenSetUpAsSuchInQuadrant1(self):
        x0, y0 = 1, 1
        x1, y1 = 10, 0
        expected = False
        actual = dibs.feature_engineering.is_angle_change_positive(x0, y0, x1, y1)
        self.assertEqual(expected, actual)

    def test__is_angle_change_positive__shouldBeFalse__whenSetUpAsSuchInQuadrant2(self):
        x0, y0 = -1, 1
        x1, y1 = 0, 1
        expected = False
        actual = dibs.feature_engineering.is_angle_change_positive(x0, y0, x1, y1)
        angle = dibs.feature_engineering.delta_angle_between_two_vectors_starting_at_origin(x0, y0, x1, y1)
        err = f"""
        Angle = {angle}
        """
        self.assertEqual(expected, actual, err)

    def test__is_angle_change_positive__shouldBeFalse__whenSetUpAsSuchInQuadrant3(self):
        x0, y0 = -1, -1
        x1, y1 = -10, 0
        expected = False
        # Act
        actual = dibs.feature_engineering.is_angle_change_positive(x0, y0, x1, y1)
        # Assert
        angle = dibs.feature_engineering.delta_angle_between_two_vectors_starting_at_origin(x0, y0, x1, y1)
        err = f"""
        Angle = {angle}
        """
        self.assertEqual(expected, actual, err)

    def test__is_angle_change_positive__shouldBeFalse__whenSetUpAsSuchInQuadrant4(self):
        x0, y0 = 1, -1
        x1, y1 = 0, -10
        expected = False
        actual = dibs.feature_engineering.is_angle_change_positive(x0, y0, x1, y1)
        angle = dibs.feature_engineering.delta_angle_between_two_vectors_starting_at_origin(x0, y0, x1, y1)
        err = f"""
Angle = {angle}
"""
        self.assertEqual(expected, actual, err)

    def test__is_angle_change_positive__shouldBeFalse__whenMovingFromQuadrant1ToQuandrant4(self):
        x0, y0 = 1, 1
        x1, y1 = 1, -1
        expected = False
        actual = dibs.feature_engineering.is_angle_change_positive(x0, y0, x1, y1)
        angle = dibs.feature_engineering.delta_angle_between_two_vectors_starting_at_origin(x0, y0, x1, y1)
        err = f"""
    Angle = {angle}
    """
        self.assertEqual(expected, actual, err)

    def test__is_angle_change_positive__shouldBeFalse__whenMovingFromQuadrant2ToQuandrant1InACornerCase(self):
        x0, y0 = -1, 0
        x1, y1 = 0, 1
        expected = False
        actual = dibs.feature_engineering.is_angle_change_positive(x0, y0, x1, y1)
        angle = dibs.feature_engineering.delta_angle_between_two_vectors_starting_at_origin(x0, y0, x1, y1)
        err = f"""
    Angle = {angle}
    """
        self.assertEqual(expected, actual, err)

    def test__is_angle_change_positive__shouldBeFalse__whenNegativeMoving(self):
        x0, y0 = -1, -1
        x1, y1 = 0, 1
        expected = False
        actual = dibs.feature_engineering.is_angle_change_positive(x0, y0, x1, y1)
        angle = dibs.feature_engineering.delta_angle_between_two_vectors_starting_at_origin(x0, y0, x1, y1)
        err = f"""
    Angle = {angle}
    """
        self.assertEqual(expected, actual, err)

    def test__is_angle_change_positive__shouldBeFalse__whenNegativeMoving2(self):
        x0, y0 = -2, 0
        x1, y1 = 0, 2
        expected = False
        actual = dibs.feature_engineering.is_angle_change_positive(x0, y0, x1, y1)
        angle = dibs.feature_engineering.delta_angle_between_two_vectors_starting_at_origin(x0, y0, x1, y1)
        err = f"""
    Angle = {angle}
    """
        self.assertEqual(expected, actual, err)

    # delta_angle_between_two_vectors_starting_at_origin
    def test__delta_angle_between_two_vectors_starting_at_origin__shouldBeNegative90__whenPointsGoQuadrant2ToQuadrant1(self):
        x0, y0 = -10, 0
        x1, y1 = 0, 10
        expected = -90.0
        # Act
        actual = dibs.feature_engineering.delta_angle_between_two_vectors_starting_at_origin(x0, y0, x1, y1)
        # Assert
        err = f"""
When moving from ({x0},{y0}) to ({x1},{y1}):

Expected  = {expected}
Actual    = {actual}
                    """.strip()
        self.assertEqual(expected, actual, err)

    def test__delta_angle_between_two_vectors_starting_at_origin__shouldBeNegative90__whenPointsGoQuadrant3ToQuadrant2(self):
        x0, y0 = -10, -10
        x1, y1 = -1, 1
        expected = -90.0
        # Act
        actual = dibs.feature_engineering.delta_angle_between_two_vectors_starting_at_origin(x0, y0, x1, y1)
        # Assert
        err = f"""
When moving from ({x0},{y0}) to ({x1},{y1}):

Expected  = {expected}
Actual    = {actual}
                    """.strip()
        self.assertEqual(expected, actual, err)

    def test__delta_angle_between_two_vectors_starting_at_origin__shouldBeNegative90__whenPointsGoQuadrant4ToQuadrant3(self):
        x0, y0 = 10, -10
        x1, y1 = -1, -1
        expected = -90.0
        # Act
        actual = dibs.feature_engineering.delta_angle_between_two_vectors_starting_at_origin(x0, y0, x1, y1)
        # Assert
        err = f"""
When moving from ({x0},{y0}) to ({x1},{y1}):

Expected  = {expected}
Actual    = {actual}
                    """.strip()
        self.assertEqual(expected, actual, err)

    def test__delta_angle_between_two_vectors_starting_at_origin__shouldBeNegative90__whenPointsGoQuadrant1ToQuadrant4(self):
        x0, y0 = 1, 1
        x1, y1 = 1, -1
        expected = -90.0
        # Act
        actual = dibs.feature_engineering.delta_angle_between_two_vectors_starting_at_origin(x0, y0, x1, y1)
        # Assert
        err = f"""
When moving from ({x0},{y0}) to ({x1},{y1}):

Expected  = {expected}
Actual    = {actual}
            """
        self.assertEqual(expected, actual, err)

    # delta_angle_between_two_vectors_by_all_positions()
    def test__delta_angle_between_two_vectors_by_all_positions__shouldGet90degrees__whenPointsGoQuadrant1ToQuadrant4(self):

        x0, y0 = 1, 1
        x1, y1 = 1, -1
        expected = -90.0
        # Act
        actual = dibs.feature_engineering.delta_angle_between_two_vectors_by_all_positions(0, 0, x0, y0, 0, 0, x1, y1)
        # Assert
        err = f"""
Expected  = {expected}
Actual    = {actual}
"""
        self.assertEqual(expected, actual, err)

    def test__delta_angle_between_two_vectors_by_all_positions__shouldGet90degrees__whenAllPointsInQuadrant1(self):
        ax0, ay0 = 1, 1
        bx0, by0 = 2, 2
        ax1, ay1 = 3, 3
        bx1, by1 = 2, 4
        expected = 90.0

        # Act
        actual = dibs.feature_engineering.delta_angle_between_two_vectors_by_all_positions(ax0, ay0, bx0, by0, ax1, ay1, bx1, by1)
        # Assert
        err = f"""

        """
        self.assertEqual(expected, actual, err)

    def test__delta_angle_between_two_vectors_by_all_positions__shouldGet90degrees__whenAllPointsInQuadrant2(self):
        ax0, ay0 = -1, 1
        bx0, by0 = -2, 2
        ax1, ay1 = -3, 3
        bx1, by1 = -2, 4
        expected = -90.0

        # Act
        actual = dibs.feature_engineering.delta_angle_between_two_vectors_by_all_positions(ax0, ay0, bx0, by0, ax1, ay1, bx1, by1)
        # Assert
        err = f"""

        """
        self.assertEqual(expected, actual, err)

    def test__delta_angle_between_two_vectors_by_all_positions__shouldGet90degrees__whenAllPointsInQuadrant3(self):
        ax0, ay0 = -1, -1
        bx0, by0 = -2, -2
        ax1, ay1 = -3, -3
        bx1, by1 = -2, -4
        expected = 90.0

        # Act
        actual = dibs.feature_engineering.delta_angle_between_two_vectors_by_all_positions(ax0, ay0, bx0, by0, ax1, ay1, bx1, by1)
        # Assert
        err = f"""

        """
        self.assertEqual(expected, actual, err)

    def test__delta_angle_between_two_vectors_by_all_positions__shouldGet90degrees__whenAllPointsInQuadrant4(self):
        ax0, ay0 = 1, -1
        bx0, by0 = 2, -2
        ax1, ay1 = 3, -3
        bx1, by1 = 2, -4
        expected = -90.0

        # Act
        actual = dibs.feature_engineering.delta_angle_between_two_vectors_by_all_positions(ax0, ay0, bx0, by0, ax1, ay1, bx1, by1)
        # Assert
        err = f"""

        """
        self.assertEqual(expected, actual, err)

    def test__delta_angle_between_two_vectors_by_all_positions__shouldBeZeroDegrees__whenGivenCoordinatesThatAreCongruent(self):
        ax0, ay0 = 1, 1
        bx0, by0 = 2, 2
        ax1, ay1 = 3, 3
        bx1, by1 = 4, 4

        expected = 0.0

        # Act
        actual = dibs.feature_engineering.delta_angle_between_two_vectors_by_all_positions(ax0, ay0, bx0, by0, ax1,
                                                                                           ay1, bx1, by1)
        # Assert
        err = f"""

        """
        self.assertEqual(expected, actual, err)

    def test__angle_between_two_vectors_by_all_positions__shouldReturn180Degrees__whenGiven180DegreesChangeAroundOrigin(self):
        ax0, ay0 = 0, 0
        bx0, by0 = 1, 0
        ax1, ay1 = 0, 0
        bx1, by1 = -1, 0

        expected = 180.0

        # Act
        actual = dibs.feature_engineering.delta_angle_between_two_vectors_by_all_positions(ax0, ay0, bx0, by0, ax1,
                                                                                           ay1, bx1, by1)
        # Assert
        err = f"""

        """
        self.assertEqual(expected, actual, err)

    def test__angle_between_two_vectors_by_all_positions__shouldReturn180Degrees__whenGiven180DegreesChange(self):
        ax0, ay0 = 1, 1
        bx0, by0 = 2, 1
        ax1, ay1 = -1, -1
        bx1, by1 = -2, -1

        expected = 180.0

        # Act
        actual = dibs.feature_engineering.delta_angle_between_two_vectors_by_all_positions(
            ax0, ay0, bx0, by0,
            ax1, ay1, bx1, by1)
        # Assert
        err = f"""

        """
        self.assertEqual(expected, actual, err)

    def test__attach_angle_between_bodyparts__shouldResultInOneNanAndOneValue(self):
        # TODO: review inputs & outputs
        # Arrange
        output_feature_name = 'DeltaAngle'
        snout, tail = 'Snout', 'Tail'
        cols = [f'{snout}_x', f'{snout}_y', f'{tail}_x', f'{tail}_y']
        data = [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [0, 1, 0, 0],
        ]
        df = pd.DataFrame(data, columns=cols)
        expected_output_data = [
            np.NaN,
            45.,
            45,
        ]
        expected_output = pd.concat([df, pd.DataFrame(expected_output_data, columns=[output_feature_name])], axis=1)
        expected_output_array = expected_output.values
        expected_output_array_minus_first_row = expected_output_array[1:]
        # Act
        actual_output_dataframe = dibs.feature_engineering.attach_angle_between_bodyparts(df, snout, tail,
                                                                                          output_feature_name=output_feature_name)
        actual_output_array = actual_output_dataframe.values
        actual_output_array_minus_first_row = actual_output_array[1:]
        # Assert
        is_equal = (expected_output_array_minus_first_row == actual_output_array_minus_first_row).all()
        err = f"""
Expected (shape={expected_output_array.shape}) = 
{expected_output_array}

Actual    (shape={actual_output_array.shape})  =
{actual_output_array}
"""
        self.assertTrue(is_equal, err)

    def test__delta_angle_between_two_vectors_by_all_positions__shouldReturnNANIfT0VectorHasNoAngle(self):
        ax0, ay0 = 0, 0
        bx0, by0 = 0, 0
        ax1, ay1 = 0, 0
        bx1, by1 = 1, 1

        expected = np.NaN

        # Act
        actual = dibs.feature_engineering.delta_angle_between_two_vectors_by_all_positions(
            ax0, ay0, bx0, by0, ax1, ay1, bx1, by1)
        # Assert
        err = f"""
-------------------
Expected = {expected}
Actual   = {actual}
-------------------
""".strip()
        is_equal = np.isnan(actual)
        self.assertTrue(is_equal, err)

    ### WIP Section. Tests that are either unfinished or not passing go here.
    @skip  # TODO: test looks fine but review function implementation
    def test__delta_angle_given_all_positions__shouldGet90degrees__whenAllPointsInQuadrant1(self):
        # First vector begins at 45 degrees
        ax0, ay0 = 1, 1
        bx0, by0 = 2, 1
        # Second vector begins at -45 degrees
        ax1, ay1 = 5, 5
        bx1, by1 = 4, 6
        # Therefore, delta total is 90deg
        expected = 90.0

        # Act
        actual = dibs.feature_engineering.delta_angle_given_all_positions(ax0, ay0, bx0, by0, ax1, ay1, bx1, by1)
        # Assert
        err = f"""

        """
        self.assertEqual(expected, actual, err)

    @skip  # TODO: test looks fine but review function implementation
    def test__delta_angle_given_all_positions__shouldGet90degrees__whenAllPointsInQuadrant2(self):
        ax0, ay0 = -1, 1
        bx0, by0 = -2, 2
        ax1, ay1 = -3, 3
        bx1, by1 = -2, 4
        expected = 90.0

        # Act
        actual = dibs.feature_engineering.delta_angle_given_all_positions(ax0, ay0, bx0, by0, ax1, ay1, bx1, by1)
        # Assert
        err = f"""

        """
        self.assertEqual(expected, actual, err)

    @skip  # TODO: test looks fine but review function implementation
    def test__delta_angle_given_all_positions__shouldGet90degrees__whenAllPointsInQuadrant3(self):
        ax0, ay0 = -1, -1
        bx0, by0 = -2, -2
        ax1, ay1 = -3, -3
        bx1, by1 = -2, -4
        expected = 90.0

        # Act
        actual = dibs.feature_engineering.delta_angle_given_all_positions(ax0, ay0, bx0, by0, ax1, ay1, bx1, by1)
        # Assert
        err = f"""

        """
        self.assertEqual(expected, actual, err)
    pass

    def test__delta_two_body_parts_angle_killian_try__shouldGetTwoNANSToStart__whenOneBodyPartRemainsStationary(self):
        # TODO: MED/HIGH: WIP: verify expected output!
        # Create a
        # Arrange
        data_xy_bodypart1 = [
            [0, 0],
            [0, 0],
            [0, 0],
        ]
        data_xy_bodypart2 = [
            [0, 0],
            [1, 1],
            [-1, 1],
        ]
        data_for_expected_output = [
            np.NaN,
            np.NaN,
            90.,
        ]
        arr_xy_1 = np.array(data_xy_bodypart1)
        arr_xy_2 = np.array(data_xy_bodypart2)
        expected_output = np.array(data_for_expected_output)

        # Act
        actual_output = dibs.feature_engineering.delta_two_body_parts_angle_killian_try(
            body_part_arr_1=arr_xy_1,
            body_part_arr_2=arr_xy_2,
        )

        # Assert
        is_equal = are_two_arrays_equal_including_possible_NaN_entries(expected_output, actual_output)
        err = f"""
    Expected output = {expected_output}

    Actual output   = {actual_output}
    """
        self.assertTrue(is_equal, err)

    @skip  # TODO: test looks fine but review function implementation
    def test__delta_angle_given_all_positions__shouldGet90degrees__whenAllPointsInQuadrant4(self):
        ax0, ay0 = 1, -1
        ax1, ay1 = 3, -3
        bx0, by0 = 2, -2
        bx1, by1 = 2, -4
        expected = 90.0

        # Act
        actual = dibs.feature_engineering.delta_angle_given_all_positions(ax0, ay0, bx0, by0, ax1, ay1, bx1, by1)
        # Assert
        err = f"""
        Expected = {expected}
        Actual   = {actual}

        """
        self.assertEqual(expected, actual, err)

    @skip  # TODO: MED: review expected output
    def test__delta_two_body_parts_angle_killian_try__lazy_delta_angle___(self):
        # TODO: WIP: verify expected output!
        # Create a
        # Arrange
        data_xy_1 = [
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0]
        ]
        data_xy_2 = [
            [0, 0],
            [0, 0],
            [1, 1],
            [2, 2],
        ]
        data_for_expected_output = [
            np.NaN,
            0,
            45,
            0.,
        ]
        arr_xy_1 = np.array(data_xy_1)
        arr_xy_2 = np.array(data_xy_2)
        expected_output = np.array(data_for_expected_output)

        # Act
        actual_output = dibs.feature_engineering.delta_two_body_parts_angle_killian_try(
            body_part_arr_1=arr_xy_1,
            body_part_arr_2=arr_xy_2)

        # Assert
        is_equal = are_two_arrays_equal_including_possible_NaN_entries(expected_output, actual_output)
        err = f"""
Expected output = {expected_output}

Actual output   = {actual_output}
"""
        self.assertTrue(is_equal, err)

    ### END ###

    @skip
    def test__stub2(self):
        # Arrange
        # Act
        # Assert
        self.assertTrue(True)

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