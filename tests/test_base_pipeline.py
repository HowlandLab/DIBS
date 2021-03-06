"""
Testing the BasePipeline class

https://docs.python.org/3/library/unittest.html
"""
from typing import Set
from unittest import TestCase, skip
import os
import random

from dibs.logging_enhanced import get_current_function, get_caller_function
import dibs

# logger = dibs.config.initialize_logger(__name__)


csv__train_data__file_path__TRAINING_DATA = dibs.config.TEST_FILE__PipelineMimic__CSV__TRAIN_DATA_FILE_PATH
csv__train_data__file_path__PREDICT_DATA = dibs.config.TEST_FILE__PipelineMimic__CSV__PREDICT_DATA_FILE_PATH
default_pipeline_class = dibs.pipeline.PipelineMimic


########################################################################################################################

def get_unique_pipe_name() -> str:
    name = f'Pipeline__{get_caller_function()}__{random.randint(0, 100_000_000_000)}__{dibs.config.runtime_timestr}'
    return name


def get_unique_pipeline_loaded_with_data() -> dibs.base_pipeline.BasePipeline:
    p = default_pipeline_class(get_unique_pipe_name())
    data_source_file_path = csv__train_data__file_path__TRAINING_DATA
    p = p.add_train_data_source(data_source_file_path)
    p = p.add_predict_data_source(csv__train_data__file_path__PREDICT_DATA)
    p = p.set_params(
        tsne_perplexity=30,
        tsne_early_exaggeration=100,
        tsne_learning_rate=50,

    )

    return p


########################################################################################################################

class TestPipeline(TestCase):

    ### Checking in on data during build process
    def test__build__shouldNotHaveNullFrameNumbers__whenOperatingNormally(self):
        # Arrange
        expected_num_of_rows_with_null_frame_values = 0
        p = get_unique_pipeline_loaded_with_data()
        p = p.set_params(tsne_perplexity=30).build()
        # Act
        df = p.df_features_train_scaled
        df_null_frame_values = df.loc[df['frame'].isnull()]
        actual_num_of_rows_with_null_frame_values = len(df_null_frame_values)
        # Assert
        err = f"""

{df_null_frame_values[['frame', 'data_source']].head(20)}

"""

        self.assertEqual(expected_num_of_rows_with_null_frame_values, actual_num_of_rows_with_null_frame_values, err)

    ### Adding new training data sources ###
    def test__pipeline_adding_train_data_file_source__should____(self):
        """"""
        # Arrange
        data_source_file_path = csv__train_data__file_path__TRAINING_DATA
        pipe = default_pipeline_class(get_unique_pipe_name())
        num_of_sources_before_addition: int = len(pipe.training_data_sources)
        num_of_sources_should_be_this_after_addition = num_of_sources_before_addition + 1

        # Act
        p = pipe.add_train_data_source(data_source_file_path)
        num_of_sources_actually_this_after_addition: int = len(p.training_data_sources)

        # Assert

        err_msg = f"""
    list_of_sources_before_addition = {num_of_sources_before_addition}
    num_of_sources_should_be_this_after_addition = {num_of_sources_should_be_this_after_addition}

    list_of_sources_after_addition = {num_of_sources_actually_this_after_addition}
    """
        self.assertEqual(num_of_sources_should_be_this_after_addition, num_of_sources_actually_this_after_addition,
                         err_msg)

    def test__pipeline_adding_train_data_file_source__shouldBeZeroToStart(self):
        """"""
        # Arrange
        p = default_pipeline_class(get_unique_pipe_name())
        expected_amount_of_sources = 0

        # Act
        actual_amount_of_dataframes = len(p.training_data_sources)

        # Assert
        err_msg = f"""
    expected_amount_of_dataframes = {expected_amount_of_sources}

    actual_amount_of_dataframes = {actual_amount_of_dataframes}
    """
        self.assertEqual(expected_amount_of_sources, actual_amount_of_dataframes, err_msg)

    def test__pipeline_add_train_data__(self):  # TODO: add should/when
        # Arrange
        p = default_pipeline_class('Test_65465465465asddsfasdfde34asdf')
        num_sources_before_adding_any = len(p.training_data_sources)

        # Act
        p = p.add_train_data_source(csv__train_data__file_path__TRAINING_DATA)
        num_sources_after_adding_sources = len(p.training_data_sources)

        is_equal = num_sources_before_adding_any + 1 == num_sources_after_adding_sources
        # Assert
        err_msg = f"""

    """
        self.assertTrue(is_equal, err_msg)

    ### Removing train data sources ###
    pass

    ### Removing predict data sources ###
    def test__remove_train_data_source__shouldRemoveSource__whenSourceIsPresent(self):
        # Arrange
        csv_file_path__this_test = csv__train_data__file_path__TRAINING_DATA
        p = dibs.base_pipeline.BasePipeline(get_unique_pipe_name())
        p = p.add_train_data_source(csv_file_path__this_test)
        num_sources_before_remove = len(p.training_data_sources)
        expected_num_sources_after = num_sources_before_remove - 1
        # Act
        p = p.remove_train_data_source(dibs.config.get_data_source_from_file_path(csv_file_path__this_test))
        actual_num_sources_after_remove = len(p.training_data_sources)

        # Assert
        self.assertEqual(expected_num_sources_after, actual_num_sources_after_remove)

    def test__remove_train_data_source__shouldChangeNothing__whenSourceNotPresent(self):
        # Arrange
        not_a_real_data_source = 'NotARealDataSource'
        p = dibs.base_pipeline.BasePipeline(get_unique_pipe_name())
        p = p.add_train_data_source(csv__train_data__file_path__TRAINING_DATA)
        expected_num_sources = len(p.training_data_sources)
        # Act
        p = p.remove_train_data_source(not_a_real_data_source)
        num_sources_after = len(p.training_data_sources)
        # Assert
        self.assertEqual(expected_num_sources, num_sources_after)

    ### Test/train splitting
    def test___scale_training_data_and_add_train_test_split__shouldOnlyProduceValuesTrueOrFalse__whenUsedNormally(self):
        # Arrange
        expected = valid_values_for_testtrain_col = {True, False}
        p = get_unique_pipeline_loaded_with_data()

        # Act
        p = p._scale_training_data_and_add_train_test_split()
        actual = unique_testrain_values = set(p.df_features_train_scaled[p.test_col_name].values)

        # Assert
        err = f"""

"""
        self.assertEqual(expected, actual, err)

    ### Label assignments ###
    def test__get_assignment_label__shouldReturnEmptyString__whenLabelNotSet(self):
        """
        Test to see if output is None if no assignment label found
        """
        # Arrange
        p = default_pipeline_class(get_unique_pipe_name())
        expected_label = ''
        # Act
        actual_label = p.get_assignment_label(0)
        # Assert
        self.assertEqual(expected_label, actual_label)

    def test__set_label__shouldUpdateAssignment__whenUsed(self):
        # Arrange
        p = default_pipeline_class(get_unique_pipe_name())
        assignment, input_label = 1, 'Behaviour1'

        # Act
        p = p.set_label(assignment, input_label)

        actual_label = p.get_assignment_label(assignment)
        # Assert
        self.assertEqual(input_label, actual_label)

    @skip  # TODO: low/med: fix test for specificity
    def test__updatingAssignment__shouldSaveLabel__whenSavedAndRereadIn(self):
        # Arrange
        # out_path =
        name = get_unique_pipe_name()
        p_write = default_pipeline_class(name)
        assignment, input_label = 12, 'Behaviour12'

        # Act
        p_write = p_write.set_label(assignment, input_label)
        p_write.save_to_folder()

        p_read = dibs.read_pipeline(
            os.path.join(
                dibs.config.OUTPUT_PATH,
                dibs.pipeline.generate_pipeline_filename(name),
            ))

        actual_label = p_read.get_assignment_label(assignment)
        # Assert
        err = f"""
    Expected label: {input_label}

    Actual label: {actual_label}


    """  # All labels map: {p_read._map_assignment_to_behaviour}
        self.assertEqual(input_label, actual_label, err)

    ### Param adds, changes, checks ###
    def test__set_params__average_over_n_frames(self):
        property_of_note = 'average_over_n_frames'
        old_value = 10
        expected_value = new_value = 20
        # Arrange
        p = default_pipeline_class(get_unique_pipe_name())
        assert old_value != new_value
        setattr(p, property_of_note, old_value)

        new_params = {property_of_note: new_value, }
        # Act
        p = p.set_params(**new_params)
        actual_value = getattr(p, property_of_note)
        # Assert
        err = f"""
    Expected {property_of_note} value: {expected_value}
    Actual   {property_of_note} value: {actual_value}"""
        self.assertEqual(expected_value, actual_value, err)

    def test__set_params__cross_validation_k(self):
        property_of_note = 'cross_validation_k'
        old_value = 10
        expected_value = new_value = 20
        # Arrange
        p = default_pipeline_class(get_unique_pipe_name())
        assert old_value != new_value
        setattr(p, property_of_note, old_value)

        new_params = {property_of_note: new_value, }
        # Act
        p = p.set_params(**new_params)
        actual_value = getattr(p, property_of_note)
        # Assert
        err = f"""
        Expected {property_of_note} value: {expected_value}
        Actual   {property_of_note} value: {actual_value}"""
        self.assertEqual(expected_value, actual_value, err)

    def test__set_params__video_fps(self):
        # Arrange
        property_of_note = 'video_fps'
        old_value = 100.
        expected_value = new_value = 30.3

        p = default_pipeline_class(get_unique_pipe_name())
        setattr(p, property_of_note, old_value)
        new_params = {property_of_note: new_value, }
        # Act
        p = p.set_params(**new_params)
        actual_value = getattr(p, property_of_note)
        # Assert
        err = f"""
    Expected {property_of_note} value: {expected_value}
    Actual   {property_of_note} value: {actual_value}"""
        self.assertEqual(expected_value, actual_value, err)

    def test__set_params__classifier_type__svmtorandomforest(self):
        property_of_note = 'classifier_type'
        old_value = 'SVM'
        expected_value = new_value = 'RANDOMFOREST'
        # Arrange
        p = default_pipeline_class(get_unique_pipe_name())
        setattr(p, property_of_note, old_value)

        new_params = {property_of_note: new_value, }
        # Act
        p = p.set_params(**new_params)
        actual_value = getattr(p, property_of_note)
        # Assert
        err = f"""
    Expected {property_of_note} value: {expected_value}
    Actual   {property_of_note} value: {actual_value}"""
        self.assertEqual(expected_value, actual_value, err)

    def test__set_params__classifier_type__rftosvm(self):
        property_of_note = 'classifier_type'
        old_value = 'RANDOMFOREST'
        expected_value = new_value = 'SVM'
        # Arrange
        p = default_pipeline_class(get_unique_pipe_name())
        setattr(p, property_of_note, old_value)

        new_params = {property_of_note: new_value, }
        # Act
        p = p.set_params(**new_params)
        actual_value = getattr(p, property_of_note)
        # Assert
        err = f"""
    Expected {property_of_note} value: {expected_value}
    Actual   {property_of_note} value: {actual_value}"""
        self.assertEqual(expected_value, actual_value, err)

    ### Scaling data ###
    @skip('Test needs to be reevaluated')  # TODO: med/high
    def test__scale_data__shouldReturnDataFrameWithSameColumnNames__afterScalingData(self):
        # Note: this function takes a while to run
        # Arrange
        p = default_pipeline_class(get_unique_pipe_name()).add_train_data_source(csv__train_data__file_path__TRAINING_DATA).build(True)

        # Act
        p = p._scale_training_data_and_add_train_test_split()

        unscaled_features_cols: Set[str] = set(p.df_features_train.columns)
        scaled_features_cols: Set[str] = set(p.df_features_train_scaled.columns)

        # Assert
        err_message = f"""
    Cols were found in one but not the other.

    unscaled_features_cols = {unscaled_features_cols}
    scaled_features_cols = {scaled_features_cols}

    Symmetric diff = {unscaled_features_cols.symmetric_difference(scaled_features_cols)}

    """.strip()
        self.assertEqual(unscaled_features_cols, scaled_features_cols, err_message)

    # GMM
    def test__set_params__gmm_n_components(self):
        property_of_note = 'gmm_n_components'
        old_value = 10
        expected_value = new_value = 20
        # Arrange
        p = default_pipeline_class(get_unique_pipe_name())
        assert old_value != new_value
        setattr(p, property_of_note, old_value)

        new_params = {property_of_note: new_value, }
        # Act
        p = p.set_params(**new_params)
        actual_value = getattr(p, property_of_note)
        # Assert
        err = f"""
    Expected {property_of_note} value: {expected_value}
    Actual   {property_of_note} value: {actual_value}"""
        self.assertEqual(expected_value, actual_value, err)

    def test__set_params__gmm_n_init(self):
        property_of_note = 'gmm_n_init'
        old_value = 10
        expected_value = new_value = 20
        # Arrange
        p = default_pipeline_class(get_unique_pipe_name())
        setattr(p, property_of_note, old_value)

        new_params = {property_of_note: new_value, }
        # Act
        p = p.set_params(**new_params)
        actual_value = getattr(p, property_of_note)
        # Assert
        err = f"""
    Expected {property_of_note} value: {expected_value}
    Actual   {property_of_note} value: {actual_value}"""
        self.assertEqual(expected_value, actual_value, err)
    
    def test__set_params__gmm_max_iter(self):
        property_of_note = 'gmm_max_iter'
        old_value = 100
        expected_value = new_value = 50
        # Arrange
        p = default_pipeline_class(get_unique_pipe_name())
        setattr(p, property_of_note, old_value)

        new_params = {property_of_note: new_value, }
        # Act
        p = p.set_params(**new_params)
        actual_value = getattr(p, property_of_note)
        # Assert
        err = f"""
    Expected {property_of_note} value: {expected_value}
    Actual   {property_of_note} value: {actual_value}"""
        self.assertEqual(expected_value, actual_value, err)

    def test__set_params__gmm_gmm_tol(self):
        property_of_note = 'gmm_tol'
        old_value = 100.
        expected_value = new_value = 50.
        # Arrange
        p = default_pipeline_class(get_unique_pipe_name())
        setattr(p, property_of_note, old_value)

        new_params = {property_of_note: new_value, }
        # Act
        p = p.set_params(**new_params)
        actual_value = getattr(p, property_of_note)
        # Assert
        err = f"""
    Expected {property_of_note} value: {expected_value}
    Actual   {property_of_note} value: {actual_value}"""
        self.assertEqual(expected_value, actual_value, err)

    def test__set_params__gmm_reg_covar(self):
        property_of_note = 'gmm_reg_covar'
        old_value = 100
        expected_value = new_value = 50
        # Arrange
        p = default_pipeline_class(get_unique_pipe_name())
        setattr(p, property_of_note, old_value)

        new_params = {property_of_note: new_value, }
        # Act
        p = p.set_params(**new_params)
        actual_value = getattr(p, property_of_note)
        # Assert
        err = f"""
    Expected {property_of_note} value: {expected_value}
    Actual   {property_of_note} value: {actual_value}"""
        self.assertEqual(expected_value, actual_value, err)

    def test__gmm_recolour__should(self):
        # Arrange
        p = get_unique_pipeline_loaded_with_data()
        old_gmm_n_components = 11
        p = p.set_params(gmm_n_components=old_gmm_n_components).build()
        expected_gmm_n_components = new_gmm_n_components = 7
        # Act
        p = p.recolor_gmm_and_retrain_classifier(new_gmm_n_components).build()
        actual_gmm_n_components = p.gmm_n_components
        # Assert
        self.assertEqual(expected_gmm_n_components, actual_gmm_n_components)

    # TSNE
    def test__set_params__tsne_perplexity(self):
        property_of_note = 'tsne_perplexity'
        old_value = 10
        expected_value = new_value = 20
        # Arrange
        p = default_pipeline_class(get_unique_pipe_name(), **{property_of_note: old_value})
        assert old_value != new_value
        # setattr(p, property_of_note, old_value)

        new_params = {property_of_note: new_value, }
        # Act
        p = p.set_params(**new_params)
        actual_value = getattr(p, property_of_note)
        # Assert
        err = f"""
    Expected {property_of_note} value: {expected_value}
    Actual   {property_of_note} value: {actual_value}"""
        self.assertEqual(expected_value, actual_value, err)

    def test__set_params__shouldMakePerplexityRelativeToNumFeatures__whenSpecifiedToBeRelativeToNumFeatures(self):
        # Arrange
        property_of_note = 'tsne_perplexity'
        old_value = 10
        new_value = 'lambda self: len(self.all_features) * 1.0'
        p = get_unique_pipeline_loaded_with_data().set_params(**{property_of_note: old_value}).build()
        expected_value = len(p.all_engineered_features) * 1.0
        assert old_value != new_value
        # logger.debug(f'Preplexity: {p.tsne_perplexity} / num features: {len(p.all_features)}')
        new_params = {property_of_note: new_value, }
        # Act
        p = p.set_params(**new_params)
        actual_value = getattr(p, property_of_note)
        # Assert
        err = f"""
    Expected {property_of_note} value: {expected_value}
    Actual   {property_of_note} value: {actual_value}"""
        self.assertEqual(expected_value, actual_value, err)

    def test__set_params__shouldMakePerplexityRelativeToNumFeatures__whenSpecifiedToBeRelativeToNumDataPoints(self):
        # TODO
        property_of_note = 'tsne_perplexity'
        old_value = 10
        attr = 'num_training_data_points'

        new_value = f'lambda self: self.{attr} * 1.0'
        p = get_unique_pipeline_loaded_with_data().set_params(**{property_of_note: old_value}).build()
        assert hasattr(p, attr)
        expected_value = getattr(p, attr) * 1.0
        # Arrange
        assert old_value != new_value
        # logger.debug(f'Perplexity: {p.tsne_perplexity} / num features: {len(p.all_features)}')
        new_params = {property_of_note: new_value, }
        # Act
        p = p.set_params(**new_params)
        actual_value = getattr(p, property_of_note)
        # Assert
        err = f"""
    Expected {property_of_note} value: {expected_value}
    Actual   {property_of_note} value: {actual_value}"""
        self.assertEqual(expected_value, actual_value, err)

    def test__set_params__tsne_learning_rate(self):
        property_of_note = 'tsne_learning_rate'
        old_value = 10
        expected_value = new_value = 20
        # Arrange
        p = default_pipeline_class(get_unique_pipe_name())
        assert old_value != new_value
        setattr(p, property_of_note, old_value)

        new_params = {property_of_note: new_value, }
        # Act
        p = p.set_params(**new_params)
        actual_value = getattr(p, property_of_note)
        # Assert
        err = f"""
    Expected {property_of_note} value: {expected_value}
    Actual   {property_of_note} value: {actual_value}"""
        self.assertEqual(expected_value, actual_value, err)

    def test__set_params__tsne_early_exaggeration(self):
        property_of_note = 'tsne_early_exaggeration'
        old_value = 10
        expected_value = new_value = 20
        # Arrange
        p = default_pipeline_class(get_unique_pipe_name())
        assert old_value != new_value
        setattr(p, property_of_note, old_value)

        new_params = {property_of_note: new_value, }
        # Act
        p = p.set_params(**new_params)
        actual_value = getattr(p, property_of_note)
        # Assert
        err = f"""
    Expected {property_of_note} value: {expected_value}
    Actual   {property_of_note} value: {actual_value}"""
        self.assertEqual(expected_value, actual_value, err)

    def test__set_params__tsne_n_iter(self):
        property_of_note = 'tsne_n_iter'
        old_value = 100
        expected_value = new_value = 50
        # Arrange
        p = default_pipeline_class(get_unique_pipe_name())
        setattr(p, property_of_note, old_value)

        new_params = {property_of_note: new_value, }
        # Act
        p = p.set_params(**new_params)
        actual_value = getattr(p, property_of_note)
        # Assert
        err = f"""
    Expected {property_of_note} value: {expected_value}
    Actual   {property_of_note} value: {actual_value}"""
        self.assertEqual(expected_value, actual_value, err)

    def test__set_params__tsne_n_components(self):
        property_of_note = 'tsne_n_components'
        old_value = 10
        expected_value = new_value = 20
        # Arrange
        p = default_pipeline_class(get_unique_pipe_name())
        assert old_value != new_value
        setattr(p, property_of_note, old_value)

        new_params = {property_of_note: new_value, }
        # Act
        p = p.set_params(**new_params)
        actual_value = getattr(p, property_of_note)
        # Assert
        err = f"""
    Expected {property_of_note} value: {expected_value}
    Actual   {property_of_note} value: {actual_value}
"""
        self.assertEqual(expected_value, actual_value, err)

    def test__set_tsne_perplexity_as_fraction_of_training_data__shouldRunFine__whenFractionValid(self):
        p = get_unique_pipeline_loaded_with_data()
        p = p.set_tsne_perplexity_as_fraction_of_training_data(0.2).build()
        err = f"""
"""
        self.assertTrue(True, err)

    def test__set_tsne_perplexity_as_fraction_of_training_data__shouldProduceError__whenFractionAt0(self):
        p = get_unique_pipeline_loaded_with_data()
        err = f"""
"""
        new_perp_frac = 0.
        expected_exception = ValueError
        self.assertRaises(expected_exception, p.set_tsne_perplexity_as_fraction_of_training_data, new_perp_frac)

    def test__set_tsne_perplexity_as_fraction_of_training_data__shouldProduceError__whenFractionBelow0(self):
        p = get_unique_pipeline_loaded_with_data()
        err = f"""
"""
        new_perp_frac = -1.
        expected_exception = ValueError
        self.assertRaises(expected_exception, p.set_tsne_perplexity_as_fraction_of_training_data, new_perp_frac)
    pass

    def test__set_tsne_perplexity_as_fraction_of_training_data__shouldProduceError__whenFractionAbove1(self):
        p = get_unique_pipeline_loaded_with_data()
        err = f"""
    """
        new_perp_frac = 1.0001
        expected_exception = ValueError
        self.assertRaises(expected_exception, p.set_tsne_perplexity_as_fraction_of_training_data, new_perp_frac)

    # SVM
    def test__set_params__svm_c(self):
        property_of_note = 'svm_c'
        old_value = 10
        expected_value = new_value = 20
        # Arrange
        p = default_pipeline_class(get_unique_pipe_name())
        assert old_value != new_value
        setattr(p, property_of_note, old_value)

        new_params = {property_of_note: new_value, }
        # Act
        p = p.set_params(**new_params)
        actual_value = getattr(p, property_of_note)
        # Assert
        err = f"""
    Expected {property_of_note} value: {expected_value}
    Actual   {property_of_note} value: {actual_value}"""
        self.assertEqual(expected_value, actual_value, err)

    def test__set_params__svm_gamma(self):
        property_of_note = 'svm_gamma'
        old_value = 10
        expected_value = new_value = 20
        # Arrange
        p = default_pipeline_class(get_unique_pipe_name())
        assert old_value != new_value
        setattr(p, property_of_note, old_value)

        new_params = {property_of_note: new_value, }
        # Act
        p = p.set_params(**new_params)
        actual_value = getattr(p, property_of_note)
        # Assert
        err = f"""
    Expected {property_of_note} value: {expected_value}
    Actual   {property_of_note} value: {actual_value}"""
        self.assertEqual(expected_value, actual_value, err)

    # RANDOM FOREST
    def test__set_params__rf_n_estimators(self):
        property_of_note = 'rf_n_estimators'
        old_value = 100
        expected_value = new_value = 50
        # Arrange
        p = default_pipeline_class(get_unique_pipe_name())
        setattr(p, property_of_note, old_value)

        new_params = {property_of_note: new_value, }
        # Act
        p = p.set_params(**new_params)
        actual_value = getattr(p, property_of_note)
        # Assert
        err = f"""
    Expected {property_of_note} value: {expected_value}
    Actual   {property_of_note} value: {actual_value}"""
        self.assertEqual(expected_value, actual_value, err)

    def test__set_params__shouldKeepDefaultsWhileChangingSpecifiedVars__whenOptionalArgForReadingInConfigVarsNotTrue(self):
        # Arrange
        p = default_pipeline_class(get_unique_pipe_name())
        default_gmm_n_components = p.gmm_n_components

        # Act
        p = p.set_params(tsne_n_components=5)
        gmm_n_components_after_set_param = p.gmm_n_components

        # Assert
        err_msg = f"""
TODO: {get_current_function()}
"""
        self.assertEqual(default_gmm_n_components, gmm_n_components_after_set_param, err_msg)

    def test__set_params__shouldDetectVars__whenNewPipelineInstantiatedAsSuch(self):
        # Arrange
        cv = expected_cv = 5

        # Act
        p = default_pipeline_class(get_unique_pipe_name(), cross_validation_k=cv)
        actual_cv = p.cross_validation_k

        # Assert
        err = f"""Error: cv cross val did not get read-in correctly. TODO: elaborate. """.strip()
        self.assertEqual(expected_cv, actual_cv, err)

    # Accuracy scoring
    def test__test_train_splitting__shouldHaveProportionOfRecords__asSpecifiedInConfig(self):
        # Arrange
        round_by = 5
        expected = dibs.config.HOLDOUT_PERCENT
        p = get_unique_pipeline_loaded_with_data()
        p = p.build()

        # Act
        data = p.df_features_train_scaled
        test_rows, total_rows = len(data.loc[data[p.test_col_name]]), len(data)
        actual = test_rows / total_rows
        err = f"""
---------------------------------------------
Expected = {expected}
Actual   = {actual}
--- DEBUG DATA ------------------------------
Total TEST data rows: {test_rows}
Total ALL  data rows: {total_rows}
---------------------------------------------
"""
        # Assert
        return self.assertAlmostEqual(expected, actual, round_by, err)

    # Plotting
    def test__plot_clusters_by_assignments__shouldPullDataJustFine__whenUsedNormally(self):
        # Arrange
        p = get_unique_pipeline_loaded_with_data().build()
        # Act
        fig, ax = p.plot_clusters_by_assignments()
        # Assert
        self.assertTrue(True)

    ### End-to-end tests ###
    def test___DefaultPipelineListedAbove___buliding_pipeline_start_to_finish(self):
        # Arrange
        p = get_unique_pipeline_loaded_with_data()

        # Act
        built_ok = False
        err = f'Build failed. Error: '
        try:
            p = p.build(True, True)
            built_ok = True
        except BaseException as e:
            err += repr(e)
            raise e
        # Assert
        self.assertTrue(built_ok, err)

    ### Tests that need to be finished, confirmed, then moved to appropriate section ###
    @skip('Test needs to be worked on, finish')
    def test__add_train_data_AND_build__shouldHaveSameNumRowsInRawDataAsBuiltData__whenRawDataBuilt(self):
        """
        After adding just 1 train data source,

        *** NOTE: This test usually takes a while since it builds the entire model as part of the test ***
        """
        # Arrange
        p = default_pipeline_class('asdfasdfdfs44444')
        p = p.add_train_data_source(csv__train_data__file_path__TRAINING_DATA)
        original_number_of_data_rows = len(dibs.io.read_csv(csv__train_data__file_path__TRAINING_DATA))

        # Act
        p = p.build()
        actual_total_rows_after_feature_engineering = len(p.df_features_train)

        # Assert
        err_msg = f'TODO: err msg'
        self.assertEqual(original_number_of_data_rows, actual_total_rows_after_feature_engineering, err_msg)

    @skip('Review if test completely built')
    def test__pipeline_adding_train_data_file_source__shouldAddParticularFileTo____when____(self):
        """"""
        # Arrange
        p = default_pipeline_class(get_unique_pipe_name())
        data_source_file_path = csv__train_data__file_path__TRAINING_DATA

        # Act
        p = p.add_train_data_source(data_source_file_path)
        is_path_now_in_list_of_paths = data_source_file_path in p.train_data_files_paths
        # Assert

        err_msg = f"""
p.train_data_files_paths = {p.train_data_files_paths}
""".strip()
        self.assertTrue(is_path_now_in_list_of_paths, err_msg)

    ### Templates ###
    @skip('Test stub')
    def test__stub7(self):
        """

        """
        # Arrange

        # Act
        is_equal = 1 + 1 == 2
        # Assert
        self.assertTrue(is_equal)

    @skip('Test stub')
    def test__stub__pipeline_instantiate(self):
        # Arrange
        p = default_pipeline_class(get_unique_pipe_name())
        data_source_file_path = csv__train_data__file_path__TRAINING_DATA
        p = p.add_train_data_source(data_source_file_path)

        # Act
        built_ok = True
        err = f"""

"""
        try:
            p = p.build(True, True)
        except BaseException as e:

            pass
        # Assert
        self.assertTrue(built_ok, err)
