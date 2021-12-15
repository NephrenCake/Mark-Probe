# -- coding: utf-8 --
import os
import logging
import time
import sys


def get_logger(save_dir, exp_name):

    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s : %(message)s"))

    file_name = os.path.join(sys.path[0], save_dir, exp_name,
                             f"{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}.log")
    fh = logging.FileHandler(filename=file_name, encoding="utf-8", mode="a")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s : %(message)s"))

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger