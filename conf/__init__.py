import os

# Folder dir 
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
LOG_DIR = os.path.join(ROOT_DIR, 'log')
LIB_DIR = os.path.join(ROOT_DIR, 'lib')
FIGURE_DIR = os.path.join(ROOT_DIR, 'figs')
SUBMISSION_DIR = os.path.join(ROOT_DIR, 'submission')
TRAINED_MODEL_DIR = os.path.join(ROOT_DIR, 'trained_model')

# Font for plot graph
# FONT = FontProperties(fname=os.path.join(conf.LIB_DIR, 'simsun.ttc'))