import os

# Folder dir 
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'sample_data')
LOG_DIR = os.path.join(ROOT_DIR, 'log')
LIB_DIR = os.path.join(ROOT_DIR, 'lib')
FIGURE_DIR = os.path.join(ROOT_DIR, 'figs')
SUBMISSION_DIR = os.path.join(ROOT_DIR, 'submission')
TRAINED_MODEL_DIR = os.path.join(ROOT_DIR, 'trained_model')
PRED_DATA_PARENT_DIR = os.path.join(ROOT_DIR, 'tcdata')
PRED_DATA_DIR = os.path.join(PRED_DATA_PARENT_DIR, 'disk_sample_smart_log_round2')

# Font for plot graph
# FONT = FontProperties(fname=os.path.join(conf.LIB_DIR, 'simsun.ttc'))
