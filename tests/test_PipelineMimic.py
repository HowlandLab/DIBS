"""
Create tests specifically for the PipelineMimic object
"""
from unittest import TestCase, skip
import os
import random

import dibs

csv__train_data__file_path = dibs.config.DEFAULT_PIPELINE__MIMIC__CSV_TEST_FILE_PATH
assert os.path.isfile(csv__train_data__file_path)


class TestPipelineMimic(TestCase):

    def test__build__shouldRunFine__whenUsingDefaults(self):
        # Arrange
        gmm_n_components, cv = 2, 3  # Set gmm clusters low so that it can still work with 10 rows of data
        p = dibs.pipeline.PipelineMimic(f'TestPipeline_{random.randint(0, 100_000_000)}',
                                        cross_validation_k=cv,
                                        gmm_n_components=gmm_n_components,
                                        )
        err = f"""Sanity Check: Something bad happened and cross val is not right"""
        self.assertEqual(cv, p.cross_validation_k, err)
        p = p.add_train_data_source(csv__train_data__file_path)
        p.cross_validation_n_jobs = 1  # Reduce CPU load. Optional.
        # Act
        p = p.build()

        # Assert
        err = f"""



"""
        self.assertTrue(True)

    def test__build__shouldBuildFine__whenBhtsneIsSpecified(self):
        # Arrange
        gmm_n_components, cv = 2, 3  # Set gmm clusters low so that it can still work with 10 rows of data
        p = dibs.pipeline.PipelineMimic(f'TestPipeline_{random.randint(0, 100_000_000)}',
                                        cross_validation_k=cv,
                                        gmm_n_components=gmm_n_components,

                                        )
        err = f"""Sanity Check: Something bad happened and cross val is not right"""
        self.assertEqual(cv, p.cross_validation_k, err)
        p = p.add_train_data_source(csv__train_data__file_path)
        p.cross_validation_n_jobs = 1  # Reduce CPU load. Optional.
        p.tsne_implementation = 'bhtsne'
        # Act

        p = p.build()

        # Assert
        err = f"""



"""
        self.assertTrue(True)

    def test__build__shouldBuildFine__whenOpentsneIsSpecified(self):
        # Arrange
        gmm_n_components, cv = 2, 3  # Set gmm clusters low so that runtime isn't long
        p = dibs.pipeline.PipelineMimic(f'TestPipeline_{random.randint(0, 100_000_000)}',
                                        cross_validation_k=cv,
                                        gmm_n_components=gmm_n_components,
                                        )
        p.cross_validation_n_jobs = 1  # Reduce CPU load. Optional.
        err = f"""Sanity Check: Something bad happened and cross val is not right"""
        self.assertEqual(cv, p.cross_validation_k, err)
        p = p.add_train_data_source(csv__train_data__file_path)
        # Act
        p = p.build()

        # Assert
        self.assertTrue(True)

    @skip
    def test__stub(self):

        pass


