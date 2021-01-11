# TODO: high: Note: pypi.org/dibs is TAKEN
# -*- coding: utf-8 -*-
# Initial setup instructions from: https://queirozf.com/entries/package-a-python-project-and-make-it-available-via-pip-install-simple-example

import setuptools

with open('README.md', 'r') as readme_file:
    long_description = readme_file.read()

setuptools.setup(
    name='EXAMPLE-PACKAGE-_yourusernamehere_', # This is likely the name that will be called on `pip install _`, so make it count # TODO: HIGH: change module name
    version='0.0.1',  # TODO: HIGH: change initial version as necessary
    url='https://github.com/HowlandLab/DIBS',
    author='Example Author',  # TODO: HIGH: add author
    author_email='johnhowlandlab@gmail.com',
    description='description goes here',  # TODO: HIGH: add (short) description
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='',  # TODO: HIGH: select license
    # packages=setuptools.find_packages(include=['SomePackage']),
    # TODO: Use the __init__ script inside 'dibs' to 'import' all modules/functions to be included in the library.
    #       For example see deeplabcut, or any other python libraries source code.
    install_requires=[  # TODO: HIGH: re-evaluate necessary minimum versions of packages in requirements.txt
        'Cython',
        # 'numpy>=1.1',
        # 'matplotlib>=1.5',  # TODO: re-evaluate matplotlib being version above 3.1. ??? Recall problems plotting on Linux and needing matplotlib to be a certain version
        'bhtsne',
        'ffmpeg',
        'hdbscan',
        'joblib',
        'matplotlib>=3.0.3',
        'networkx',
        'numpy>=1.16.4',
        'pandas',
        'psutil',
        'opencv-python',
        'opentsne',
        'seaborn',
        'scikit-learn',
        'streamlit',
        'tables',
        'tqdm',
        'umap-learn',
        # etc...
    ],
    setup_requires=['pytest-runner', ],  # TODO: change this
    tests_require=['pytest', ],  # TODO: change this
    test_suite='unittest',  # TODO: check this
    classifiers=[  # https://pypi.org/classifiers/  #TODO: HIGH
        'Development Status :: 1 - Planning',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        # 'Intended Audience :: Education',
        # 'Intended Audience :: Science/Research',
        # 'License :: OSI Approved :: GNU General Public License (GPL)',
        # 'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        # 'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        # 'Topic :: Scientific/Engineering :: Image Recognition',
        # 'Topic :: Scientific/Engineering :: Artificial Intelligence',
        ],
)
