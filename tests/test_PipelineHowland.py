"""
Create tests specifically for the PipelineHowland class
"""
from unittest import TestCase, skip
import os
import random

import dibs

csv__train_data__file_path__TRAINING_DATA = dibs.config.TEST_FILE__PipelineMimic__CSV__TRAIN_DATA_FILE_PATH
csv__train_data__file_path__PREDICT_DATA = dibs.config.TEST_FILE__PipelineMimic__CSV__TRAIN_DATA_FILE_PATH
assert os.path.isfile(csv__train_data__file_path__TRAINING_DATA)

default_pipeline_class = dibs.pipeline.PipelineHowland


########################################################################################################################

def get_unique_pipe_name() -> str:
    name = f'Pipeline__{dibs.logging_enhanced.get_caller_function()}__{random.randint(0, 100_000_000)}__{dibs.config.runtime_timestr}'
    return name


def get_unique_pipeline_loaded_with_data() -> dibs.base_pipeline.BasePipeline:
    p = default_pipeline_class(get_unique_pipe_name())
    data_source_file_path = csv__train_data__file_path__TRAINING_DATA
    p = p.add_train_data_source(data_source_file_path)
    p = p.add_predict_data_source(csv__train_data__file_path__PREDICT_DATA)

    return p


########################################################################################################################


class TestPipelineMimic(TestCase):

    def test__build__shouldRunFine__whenUsingDefaults(self):
        # Arrange
        p = default_pipeline_class(get_unique_pipe_name())
        err = f"""Sanity Check: Something bad happened and cross val is not right"""
        p = p.add_train_data_source(csv__train_data__file_path__TRAINING_DATA)
        # Act
        p = p.build()

        # Assert
        self.assertTrue(True)

    def test__build__shouldBuildFine__whenBhtsneIsSpecified(self):
        # Arrange
        gmm_n_components, cv = 2, 2  # Set gmm clusters low so that it can still work with 10 rows of data
        p = default_pipeline_class(get_unique_pipe_name(),
                                   cross_validation_k=cv,
                                   gmm_n_components=gmm_n_components,
                                   )
        err = f"""Sanity Check: Something bad happened and cross val is not right"""
        self.assertEqual(cv, p.cross_validation_k, err)
        p = p.add_train_data_source(csv__train_data__file_path__TRAINING_DATA)
        p.cross_validation_n_jobs = 1  # Reduce CPU load. Optional.
        p.tsne_implementation = 'BHTSNE'
        # Act
        err = f"""
        Columns: {list(p.df_features_train_scaled.columns)}

        """.strip()
        try:
            p = p.build()
        except Exception as e:
            print(err)
            raise e
        # Assert

        self.assertTrue(True, err)

    def test__build__shouldBuildFine__whenOpentsneIsSpecified(self):
        # Arrange
        gmm_n_components, cv = 2, 2  # Set gmm clusters low so that runtime isn't long
        tsne_n_iter = 500
        p = default_pipeline_class(get_unique_pipe_name(),
                                   cross_validation_k=cv,
                                   gmm_n_components=gmm_n_components,
                                   tsne_n_iter=tsne_n_iter
                                   )
        p.cross_validation_n_jobs = 1  # Reduce CPU load. Optional.
        err = f"""Sanity Check: Something bad happened and cross val is not right"""
        self.assertEqual(cv, p.cross_validation_k, err)
        p = p.add_train_data_source(csv__train_data__file_path__TRAINING_DATA)
        p.tsne_implementation = 'OPENTSNE'
        # Act
        p = p.build()
        # Assert
        self.assertTrue(True)

    def test__build__shouldBuildFine__whenSklearnIsSpecified(self):
        """
        When GMM N components/CV-K are badly balanced, you will get the
        following error: ValueError: The number of classes has to be greater than one; got 1 class (CV)
        """
        # Arrange
        gmm_n_components, cv = 3, 2  # Set gmm clusters low so that runtime isn't long
        p = default_pipeline_class(get_unique_pipe_name(), cross_validation_k=cv, gmm_n_components=gmm_n_components)
        p.cross_validation_n_jobs = 1  # Reduce CPU load. Optional.
        err = f"""Sanity Check: Something bad happened and cross val is not right"""
        self.assertEqual(cv, p.cross_validation_k, err)
        p = p.add_train_data_source(csv__train_data__file_path__TRAINING_DATA)
        p.tsne_implementation = 'SKLEARN'
        # Act
        p = p.build()
        # Assert
        self.assertTrue(True)

    def test__build__shouldBuildFine__whenSetParamsForAlmostEverything__example1(self):
        """
        This is a kitchen sink test from legacy
        """
        # Arrange
        gmm_n_components, cv = 2, 2  # Set gmm clusters low so that runtime isn't long
        p = default_pipeline_class(get_unique_pipe_name(), cross_validation_k=cv, gmm_n_components=gmm_n_components)
        p.cross_validation_n_jobs = 1  # Reduce CPU load. Optional.
        err = f"""Sanity Check: Something bad happened and cross val is not right"""
        self.assertEqual(cv, p.cross_validation_k, err)
        p = p.add_train_data_source(csv__train_data__file_path__TRAINING_DATA)

        select_classifier = 'SVM'

        select_rf_n_estimators = 100
        video_fps = 30.1
        average_over_n_frames = 4
        slider_gmm_n_components = 10
        input_k_fold_cross_val = 2
        input_tsne_perplexity = 12
        input_tsne_learning_rate = 13
        input_tsne_early_exaggeration = 16.
        input_tsne_n_iter = 250
        input_tsne_n_components = 3
        input_gmm_reg_covar = 1.
        input_gmm_tolerance = 0.001
        input_gmm_max_iter = 300
        input_gmm_n_init = 20
        input_svm_c = 1.
        input_svm_gamma = 2.

        model_vars = {
            # General opts
            'classifier_type': select_classifier,
            'rf_n_estimators': select_rf_n_estimators,
            'input_videos_fps': video_fps,
            'average_over_n_frames': average_over_n_frames,

            'gmm_n_components': slider_gmm_n_components,
            'cross_validation_k': input_k_fold_cross_val,

            # Advanced opts
            'tsne_perplexity': float(input_tsne_perplexity),
            'tsne_learning_rate': float(input_tsne_learning_rate),
            'tsne_early_exaggeration': input_tsne_early_exaggeration,
            'tsne_n_iter': input_tsne_n_iter,
            'tsne_n_components': input_tsne_n_components,

            'gmm_reg_covar': input_gmm_reg_covar,
            'gmm_tol': input_gmm_tolerance,
            'gmm_max_iter': input_gmm_max_iter,
            'gmm_n_init': input_gmm_n_init,

            'svm_c': input_svm_c,
            'svm_gamma': input_svm_gamma,

        }

        p = p.set_params(**model_vars)

        # Act
        p = p.build()

        pass


