# DIBS: Data-Informed Behavioural Segmentation



## Installation & Setup

- Ensure that you have Anaconda installed
  - You can ensure you have conda installed by running: `conda --version`
- Run the following command to automate creation of the environment: `conda env create -n dibs -f env_windows.yml`

- Run the following command to create your environment: `conda create -n dibs && conda activate dibs`
- Copy and paste all of the commands from [CONDA_ENV_COMMANDS.txt](./CONDA_ENV_COMMANDS.txt) into your 
  command line one at a time to download all necessary packages


## Usage

### Streamlit

To run normally, run: `streamlit run main.py streamlit`

To run the Streamlit app with an existing Pipeline file, run:

  - `streamlit run main.py streamlit -- -p '/absolute/path/to/existing.pipeline'`
    - This works with Linux and Windows systems so long as the path is absolute, and
    - **Ensure that the path to the pipeline is in single quotes** so that it is evaluated as-is (or else you 
    could have problems with backslashes being evaluated and other weird character quirks)
      

To see the FAQ, see this: [FAQ.md](./FAQ.md)

### Tests

To run tests, run: `python -m unittest discover DIBS/tests`

### Tunable parameters
- #TODO:
- SVM
  - C
  - gamma