from ast import arg
import glob
import json
import os
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
import queue
import sys
from weakref import WeakSet
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
from scipy.sparse import diags,coo_matrix

from tqdm import tqdm
sys.path.append(Path(__file__).parent.parent.as_posix())
import SourceToTargetsFile
from MeshProcessor import MeshProcessor
from joblib import Parallel, delayed
from easydict import EasyDict
from os.path import join
class TrainingDataWriter:
    def __init__(self, args):
        self.args = args
        # self.files = files
        self.files_to_copy = None

    def initialize(self, path):
        flag = self.args.skipexisting
        self.args.skipexisting = False
        a= self.read_single_mesh(path, 0)
        directory = Path(a.mesh_name[:-4]).parent
        prefix = Path(a.mesh_name[:-4]).parts[-1]
        save_list = sorted(directory.glob(f"{prefix}_*"))
        save_list = [item.parts[-1][len(prefix):]  for item in save_list]
        self.files_to_copy = save_list
        self.args.skipexisting = flag
        return 

    def process_one_sample(self, path, index, processing):
        if (a := self.read_single_mesh(path, index, processing)) is not None:
            b= self.analyze_single_mesh(a, processing)
            c= self.writing(b)

    def read_single_mesh(self, mesh_name, mesh_index, processing=True):
        # mesh_name = self.files[mesh_index]

        fname = os.path.split(mesh_name)[-1]
        fname = fname.split('.')[0]

        directory = os.path.join(self.args.save_dir, fname)
        if self.args.skipexisting and os.path.exists(directory):
            file_list = ['vertices.npy', 'faces.npy']
            if processing:
                file_list.extend(["samples.npy", "samples_normals.npy", "samples_wks.npy", "centroids_wks.npy",
                                    "centroids_and_normals.npy", 'lap_perm_c.npy', 'lap_perm_r.npy', 'w.npy', 
                                    'new_grad.npz', 'new_rhs.npz', 'lap_L.npz', 'lap_U.npz'])
            for sf in file_list:
                tf = os.path.join(directory, sf)
                if not os.path.exists(tf):
                    print(f'missing file  {tf} in existing directory, recreating that directory')
                    break
            else:  # all files accounted for
                # print(f'skipping {directory}, already exists in destination (instructed to skip and not overwrite)')
                return None

        if mesh_name.endswith('off'):
            vertices, faces, _ = igl.read_off(mesh_name)
        elif mesh_name.endswith('obj'):
            vertices, _, _, faces, _, _ = igl.read_obj(mesh_name)
        return EasyDict({'mesh_name': mesh_name, 'vertices': vertices, 'faces': faces, 'output_directory': directory, 'mesh_index': mesh_index})
        
    def analyze_single_mesh(self, mesh_dict, processing=True):
        os.environ["OPENBLAS_NUM_THREADS"] = "1"  # this doesn't seem to help but why not try
        processor = MeshProcessor(mesh_dict.vertices, mesh_dict.faces, self.args.ttype,
                                     self.args.dir, from_file=True, 
                                     load_wks_centroids=not self.args.no_compute_wks,
                                     load_wks_samples=not self.args.no_compute_wks,
                                     compute_splu=not self.args.no_compute_splu)
        if processing:
            processor.get_samples()
            processor.get_centroids()
            processor.get_differential_operators()
        out_np, out_npz = processor.get_writeable()
        mesh_dict['np'] = out_np
        mesh_dict['npz'] = out_npz
        return mesh_dict

    def copy_single_dir(self, mesh_name, copy_list, target_path):
        mesh_name = mesh_name[0:-4]
        for file, target in zip(copy_list, target_path):
            if os.path.exists(target):
                continue
            try:
                shutil.copy2(file, target)
            except:
                shutil.copy2(file.replace("db_train", "db_train_params"), target)


    def writing(self, mesh_dict):
        # getting all files
        Path(mesh_dict.output_directory).mkdir(exist_ok=True)
        np_saves = mesh_dict['np'].keys()
        for item in np_saves:
            np.save(os.path.join(mesh_dict.output_directory, item), mesh_dict['np'][item])

        mesh_dict['npz']  = self.convert_to_coo(mesh_dict['npz'])
        npz_saves = mesh_dict['npz'].keys()
        for item in npz_saves:
            save_npz(os.path.join(mesh_dict.output_directory, item), mesh_dict['npz'][item])


        copy_list = [mesh_dict.mesh_name[:-4] + key for key in self.files_to_copy]
        target_path = [join(mesh_dict.output_directory,  key[1:]) for key in self.files_to_copy]
        self.copy_single_dir(mesh_dict.mesh_name, copy_list, target_path)

    def convert_to_coo(self, npz_saves):
        for i,key in enumerate(npz_saves):
            if not isinstance(npz_saves[key], coo_matrix):
                try:
                    npz_saves[key] = npz_saves[key].tocoo()
                except Exception:
                    npz_saves[key] = npz_saves[key].to_coo()
        return npz_saves

    def run(self, files, sources):
        # Investigate this
        self.initialize(files[0])
        if not self.args.parallel:
            [self.process_one_sample(files[i],i, processing=(Path(files[i]).parts[-1][:-4] in sources))  for i in tqdm(range(self.args.fastforward, len(files)))]
        else:
            print("RUNNING IN PARALLEL")
            Parallel(n_jobs=-1)(delayed(self.process_one_sample)(files[i], i, processing= (Path(files[i]).parts[-1][:-4] in sources))  for i in tqdm(range(self.args.fastforward, len(files))))

        # print(f"CPU {self.args.cpus}")
        # if self.__original_pairs is not None:
        #     pairs = []
        #     for pair in self.__original_pairs:
        #         new_pair = [None, None]
        #         for i in range(2):
        #             if pair[i] not in self.__files_to_dirs:
        #                 break
        #             new_pair[i] = os.path.basename(os.path.normpath(self.__files_to_dirs[pair[i]]))
        #         if new_pair[0] is None or new_pair[
        #             1] is None:  # these did not exist as processed, so there was an issue, and we skip them
        #             continue
        #         pairs.append(new_pair)
        #     SourceToTargetsFile.write(os.path.join(self.target_dir, 'data.json'), pairs)


def main(args):
    print( f"running on {args.cpus} cpus, processing {args.dir}, writing into {args.save_dir}, {'including' if not args.skiptargets else 'skipping'} targets, starting from file #{args.fastforward},  {'overwritting' if not args.skipexisting else 'skipping'} targets that already exist in destination")
    assert os.path.exists(args.dir), f"Source directory '{args.dir}' doesn't exist"
    if args.save_dir is not None:
        if not os.path.exists(args.save_dir):
            print(f"CREATING THE OUTPUT DIRECTORY {args.save_dir}")
            os.mkdir(args.save_dir)
        else:
            warnings.warn(
                f'TrainingDataWriter: target directory {args.save_dir} already exists, gonna blindly overwrite (or add) files in that directory')

    files = glob.glob(os.path.join(args.dir, '*.off'))
    if len(files) == 0:
        files = glob.glob(os.path.join(args.dir, '*.obj'))
    assert len(files) > 0
    files.sort()
    jfile = "data.json"
    data_file = os.path.join(args.dir, jfile)
    shutil.copy(data_file, os.path.join(args.save_dir, jfile))
    if os.path.isfile(data_file):
        with open(data_file) as file:
            pairs = json.load(file)
            pairs = pairs["pairs"]
    else:
        pairs = None
    sources_to_processed = [u for (u,v) in pairs]
    if not args.skiptargets:
        sources_to_processed = sources_to_processed + [v for (u,v) in pairs]

    t = TrainingDataWriter(args=args)
    t.run(files, sources_to_processed)
    print("xxxxxxx ...NOW MAIN IS REALLY EXITING :P (LAST PRINT BEFORE RETURNING) xxxxxxxxx")

def main_from_args():
    # if len(sys.argv) == 1:
    #     sys.argv.append('C:/final_for_star')
    #     sys.argv.append('-c')
    #     sys.argv.append('2')

    print("starting...")
    # dir_name = 'data/10k_surface_patches'

    import argparse

    dcpu = min(mp.cpu_count() - 2, 24)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', nargs='?', default='data/faust')
    parser.add_argument("--skipexisting", help="dont process and write meshes that already exist in target",
                        action="store_true")
    parser.add_argument('-v', "--verbose", help="verbose reporting of timings", action="store_true")
    parser.add_argument('-s', "--save_dir",
                        help=f"destination dir to write to (content may be overwritten). Defaults to <input_dir>_processed",
                        type=str)
    parser.add_argument('-c', "--cpus",
                        help=f"number of cpus to use (default on this machine is {dcpu}, out of {mp.cpu_count()})",
                        type=int, default=dcpu)
    parser.add_argument('-ff', '--fastforward',
                        help='start from the given index of the file (sorted by lexigoraphical order)', type=int,
                        default=0)
    parser.add_argument("--skiptargets",
                        help="If specified,  don't perform analysis (like laplacian) on meshes that only act as target according to the json file. This option exists for speed considerations in cases where you know that a mesh will NEVER be used as a source.",
                        action="store_true")
    parser.add_argument("--singlethreadblas",
                        help="If specified,  run blas in single thread",
                        action="store_true")
    parser.add_argument("--copy_missing", help="If specified,  run blas in single thread",
                        action="store_true")
    parser.add_argument("--make_fake_json", help="create a json file that maps each mesh to itself",
                        action="store_true")
    parser.add_argument("--parallel", help="Run processing in parrallel with joblib",
                        action="store_true")
    parser.add_argument("--no_compute_splu", help="Avoid computing splu and storing the results",
                        action="store_true")
    parser.add_argument("--no_compute_wks", help="Avoid computing splu and storing the results",
                        action="store_true")
    args = parser.parse_args()
    args = EasyDict(args.__dict__)

    if args.save_dir is None:
        dir_to_write_to = args.dir
        if dir_to_write_to[-1] == '/' or dir_to_write_to[-1] == '\\':
            dir_to_write_to = dir_to_write_to[0:-1]
        dir_to_write_to += "_processed/"
        args.save_dir = dir_to_write_to

    args.EOF = False
    args.ttype = torch.float
    # args.skipexisting = False

    print(args)
    main(args)


if __name__ == '__main__':
    main_from_args()
