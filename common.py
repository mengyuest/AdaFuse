from os.path import join as ospj

from os.path import expanduser
ROOT_DIR = ospj(expanduser("~"), "scratch")  # TODO change this to your root path // root dir
DATA_PATH = ospj(ROOT_DIR, "datasets")       # TODO change this to your data path // dataset path
EXPS_PATH = ospj(ROOT_DIR, "logs_tsm")       # TODO change this to your logs path // saving logs


def inner_set_manual_data_path(data_path, exps_path):
    if data_path is not None:
        global DATA_PATH
        DATA_PATH = data_path

    if exps_path is not None:
        global EXPS_PATH
        EXPS_PATH = exps_path


def set_manual_data_path(data_path, exps_path):
    inner_set_manual_data_path(data_path, exps_path)

    global STHV1_FRAMES
    STHV1_FRAMES = ospj(DATA_PATH, "sthv1", "20bn-something-something-v1")

    global STHV2_FRAMES
    STHV2_FRAMES = ospj(DATA_PATH, "something2something-v2", "frames")

    global MINIK_FRAMES
    MINIK_FRAMES = ospj(DATA_PATH, "kinetics-qf")

    global JESTER_FRAMES
    JESTER_FRAMES = ospj(DATA_PATH, "jester", "20bn-jester-v1")