"""
Only used when training & evaluating model on local
"""

import datetime
import os

date = datetime.datetime.now()
date = date.strftime("%y%m%d")

DATASET_PATH = os.getcwd()
NSML_NFS_OUTPUT = ""
SESSION_NAME = "FED_LNL_model__{}".format(date)
IS_ON_NSML = False
GPU_NUM = 0


def report(**args):
    # print(args)
    pass


def save(name):
    pass

