# DIBS: Data-Informed Behavioural Segmentation

## Summary
*DIBS* is a package built on top of DeepLabCut (commonly recognized as "DLC") that aims to 
automate and standardize animal behaviour recognition for video analysis.

Often times, human analysts are asked to gauge and record animal behaviours for an experiment,
but scoring between researchers can vary greatly due to differences in unavoidable subjective perspective. 
*DIBS* aims to reduce the human-error element 
from the equation and instead uses a data-centric approach to cluster similar 
behaviours together and then standardize the behaviour recognition process. By having just one set 
of "rules" (more on this later) that DIBS uses to recognize distinct behaviours, it reduces instances of 
researcher error due to things like overlooked behaviours or split decisions on short-duration behaviours. 
On top of all this, *DIBS* comes replete with a web app option that allows for
non-technically savvy people to utilize the package without any prior programming knowledge.
Just follow the package installation instructions, the usage guide, and enjoy!

## Installation & Setup

- Ensure that you have Anaconda installed
  - You can ensure you have conda installed by running: `conda --version`
- Run the following command to automate creation of the environment: `conda env create -n dibs -f env_windows.yml`

- Run the following command to create your environment: `conda create -n dibs && conda activate dibs`
- Copy and paste all of the commands from [CONDA_ENV_COMMANDS.txt](./CONDA_ENV_COMMANDS.txt) into your 
  command line one at a time to download all necessary packages


## Usage

### Bare package use option

(WIP)

For clear examples on how best to use the DIBS Pipeline, open a local instance of `jupyter notebook`
and open the [Try-Me! notebook](./notebooks/TryMe.ipynb).


### Web app option: Streamlit

To run normally, run: `streamlit run main.py streamlit`

To run the Streamlit app with an existing Pipeline file, run:

  - `streamlit run main.py streamlit -- -p '/absolute/path/to/existing.pipeline'`
    - This works with Linux and Windows systems so long as the path is absolute, and
    - **Ensure that the path to the pipeline is in single quotes** so that it is evaluated as-is (or else you 
    could have problems with backslashes being evaluated and other weird character quirks)
      

### Configuring runtime and default variables

A user may configure these variables at [config.ini](./config.ini).


_

To see the FAQ, read here: [FAQ.md](./FAQ.md)

## Tests

To run tests, run: `python -m unittest discover DIBS/tests`

Note: you may see much more debug information during each test's pipeline build than you 
were expecting if your config.ini variable for **[TESTING] [STREAM_LOG_LEVEL]** is too 
general. This can be quite cluttered to the uninitiated. Changing the level of output from 
DEBUG to WARNING will greatly reduce the amount of output.



### 