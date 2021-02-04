# Include at top of Notebook header to inject DIBS project path into sys.path
import os
import sys
DIBS_project_path = os.path.dirname(os.path.dirname(os.path.abspath(__name__)))
if DIBS_project_path not in sys.path:
    sys.path.insert(0, DIBS_project_path)
# from dibs.pipeline import *  # This line is required for Streamlit to load Pipeline objects. Do not delete. For a more robust solution, consider: https://rebeccabilbro.github.io/module-main-has-no-attribute/
