"""
Create tests specifically for the PipelineMimic object
"""
from unittest import TestCase, skip
import os
import random

import dibs

csv__train_data__file_path__TRAINING_DATA = dibs.config.TEST_FILE__PipelineMimic__CSV__TRAIN_DATA_FILE_PATH
csv__train_data__file_path__PREDICT_DATA = dibs.config.TEST_FILE__PipelineMimic__CSV__TRAIN_DATA_FILE_PATH
assert os.path.isfile(csv__train_data__file_path__TRAINING_DATA)


def get_unique_pipe_name() -> str:
    name = f'Pipeline__{dibs.logging_enhanced.get_caller_function()}__' \
           f'{random.randint(0, 100_000_000)}__{dibs.config.runtime_timestr}'
    return name


class TestPipelineMimic(TestCase):

    def test__build__shouldRunFine__whenUsingDefaults(self):
        # Arrange
        p = dibs.pipeline.PipelineMimic(get_unique_pipe_name())
        err = f"""Sanity Check: Something bad happened and cross val is not right"""
        p = p.add_train_data_source(csv__train_data__file_path__TRAINING_DATA)
        # Act
        p = p.build()

        # Assert
        self.assertTrue(True)

    def test__build__shouldBuildFine__whenBhtsneIsSpecified(self):
        # Arrange
        gmm_n_components, cv = 2, 3  # Set gmm clusters low so that it can still work with 10 rows of data
        p = dibs.pipeline.PipelineMimic(get_unique_pipe_name(),
                                        cross_validation_k=cv,
                                        gmm_n_components=gmm_n_components,
                                        # tsne_n_jobs=1,
                                        )
        err = f"""Sanity Check: Something bad happened and cross val is not right"""
        self.assertEqual(cv, p.cross_validation_k, err)
        p = p.add_train_data_source(csv__train_data__file_path__TRAINING_DATA)
        p.cross_validation_n_jobs = 1  # Reduce CPU load. Optional.
        p.tsne_implementation = 'BHTSNE'
        # Act

        p = p.build()

        # Assert
        err = f"""



"""
        self.assertTrue(True)

    def test__build__shouldBuildFine__whenOpentsneIsSpecified(self):
        # Arrange
        gmm_n_components, cv = 2, 3  # Set gmm clusters low so that runtime isn't long
        p = dibs.pipeline.PipelineMimic(get_unique_pipe_name(),
                                        cross_validation_k=cv,
                                        gmm_n_components=gmm_n_components,
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
        # Arrange
        gmm_n_components, cv = 2, 3  # Set gmm clusters low so that runtime isn't long
        p = dibs.pipeline.PipelineMimic(get_unique_pipe_name(),
                                        cross_validation_k=cv,
                                        gmm_n_components=gmm_n_components,
                                        )
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
        # Arrange
        gmm_n_components, cv = 2, 3  # Set gmm clusters low so that runtime isn't long
        p = dibs.pipeline.PipelineMimic(f'TestPipeline_{random.randint(0, 100_000_000)}',
                                        cross_validation_k=cv,
                                        gmm_n_components=gmm_n_components,
                                        )
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

        # TODO: HIGH: make sure that model parameters are put into Pipeline before build() is called.
        p = p.set_params(**model_vars)

        # Act
        p = p.build()

        pass


