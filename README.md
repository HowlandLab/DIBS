# DIBS
(WIP) Data-Informed Behavioural Segmentation



## Installation & Setup
- Ensure that you have Anaconda installed
- Run the following command: `conda env create -n dibs -f environment.yml`


## Usage

### Streamlit

To run normally, run: `streamlit run main.py streamlit`

To run the Streamlit app with an existing Pipeline file, run:

  - `streamlit run main.py streamlit -- -p '/path/to/existing.pipeline'`
    - This works with Linux and Windows systems so long as the path is absolute
    - **Ensure that the path to the pipeline is in single quotes** so that it is evaluated as-is (or else you 
    could have problems with backslashes and other weird characters)
      



