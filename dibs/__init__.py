"""
TODO: med: Add in description of this package
"""

__version__ = '0.0.1'  # TODO: review this version #
__author__ = 'Howland Lab'  # TODO: review author(s)
__credits__ = 'John Howland Lab at University of Saskatchewan'


# General imports for packages
from . import (
    app,
    base_pipeline,
    check_arg,
    config,
    feature_engineering,
    io,
    logging_enhanced,
    statistics,
    streamlit_app,
    streamlit_session_state,
    videoprocessing,
    visuals,
)

# User-facing IO API
from .io import (
    read_csv,
    read_pipeline,
)

## Edgar package exsample of autoimporting modules
# from os.path import dirname, basename, isfile
# import glob
# from .edgar import Edgar
# from .txtml import TXTML
# from .company import Company
# from .xbrl import XBRL, XBRLElement
# from .document import Document, Documents
#
# __version__ = "5.4.1"
#
# modules = glob.glob(dirname(__file__)+"/*.py")
# __all__ = [ basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]
