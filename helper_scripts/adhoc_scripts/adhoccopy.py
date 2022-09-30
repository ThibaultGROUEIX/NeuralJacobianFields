from ast import arg
import glob
import json
import os
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
import queue
import sys
import numpy as np
import torch
from time import time, sleep
import multiprocessing as mp
import igl
import shutil
from scipy.sparse import save_npz
import threading
import collections
import warnings
from datetime import datetime
from pathlib import Path

from tqdm import tqdm
sys.path.append(Path(__file__).parent.parent.as_posix())
import SourceToTargetsFile
from MeshProcessor import MeshProcessor
from joblib import Parallel, delayed
from easydict import EasyDict
from os.path import join


if __name__ == '__main__':
    # Parallel(n_jobs=-1)(delayed(shutil.move)(f"/sensei-fs/users/groueix/db_train_params/{i:08d}_gender.npy", f"/sensei-fs/users/groueix/db_train/{i:08d}_gender.npy") for i in tqdm(range(100000)) )
    # Parallel(n_jobs=-1)(delayed(shutil.copy)(f"/sensei-fs/users/groueix/db_train_params/{i:08d}_gender.npy", f"/sensei-fs/users/groueix/db_train/{i:08d}_gender.npy") for i in tqdm(range(170000, 600000)) )
    for i in tqdm(range(170001, 600000)):
        shutil.move(f"/sensei-fs/users/groueix/db_train_params/{i:08d}_shape.npy", f"/sensei-fs/users/groueix/db_train/{i:08d}_shape.npy")
        shutil.move(f"/sensei-fs/users/groueix/db_train_params/{i:08d}_pose.npy", f"/sensei-fs/users/groueix/db_train/{i:08d}_pose.npy")
        shutil.move(f"/sensei-fs/users/groueix/db_train_params/{i:08d}_gender.npy", f"/sensei-fs/users/groueix/db_train/{i:08d}_gender.npy")
