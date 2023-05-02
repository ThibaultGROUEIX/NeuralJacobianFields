import random
from re import I
import sys

from torch.utils.data import Dataset
import numpy as np
import os
import glob
from BatchOfTargets import BatchOfTargets
from SourceMesh import SourceMesh
import time
from os.path import join
from pathlib import Path
from MeshProcessor import WaveKernelSignatureError
import logging
logging.basicConfig(level=logging.DEBUG, filename='./exception.log')

class DeformationDataset(Dataset):
    '''
    Main dataset. Each sample in it is a source <---> target pair. Note this dataset return a custom Batch class object instead of a
    tensor, which is already batched.
    '''
    def num_meshes(self):
        return len(self.file_names)

    def num_pairs(self):
        return self.len_pairs

    def __init__(self, s_and_t,source_keys,target_keys,ttype, args, train=False, cpuonly=False):
        '''
        :param s_and_t: list of tuples, each tuple is two indices into the file_names, giving a source/target pair
        :param max_source_batch: max batch_size. Note batches cannot have more than one unique source mesh.
        :param batch_class: which class to use to create batch objects
        '''
        SHUFFLE_TARGETS_OF_SINGLE_SOURCE = True
        self.cpuonly=cpuonly
        self.ttype = ttype
        self.train = train
        self.unique_source = False

        self.len_pairs = len(s_and_t)
        if SHUFFLE_TARGETS_OF_SINGLE_SOURCE:
            random.shuffle(s_and_t)

        self.source_and_target = None
        # st =time.time()
        source_target = {}
        for val in s_and_t:
            s = val[0]
            t = val[1]
            if s not in source_target:
                source_target[s] = []
            source_target[s].append(t)
        # print(f"ellapsed {time.time() -st}")
        if len(source_target) == 1 and not (args.init == "isometric" and args.ninit > 1):
            # This flag will avoid reloading the source at each iteration, since the source is always the same.
            self.unique_source = True
        self.source_and_target = []
        # TODO: duplicate source for however many initializations we need
        for s,ts in source_target.items():
            chunks = np.split(ts, np.arange(args.targets_per_batch,len(ts),args.targets_per_batch))
            if args.init == "isometric":
                for _ in range(args.ninit):
                    for chunk in chunks:
                        self.source_and_target.append((s,chunk))
            else:
                for chunk in chunks:
                    self.source_and_target.append((s,chunk))

        self.source_keys = source_keys
        self.s_and_t = s_and_t
        self.target_keys = target_keys
        self.args = args
        self.directory = self.args.root_dir_train if self.train else self.args.root_dir_test
        self.directory = "" if self.directory is None else self.directory
        self.source = None
        self.target = None
        self.star_is_initialized = False

        # Store point dimension
        # if args.layer_normalization == "FLATTEN":
        #     self.point_dim = args.flat_channels
        # if args.fft:
        #     self.point_dim = 6 + 2 * args.fft_dim
        # else:
        #     self.point_dim = 6

    def initialize_star(self):
        # DEFINE GENERATOR FUNCTION FOR HUMANS
        from dataset_generation.human_db.star_generator import StarGenerator
        # We use a list here because we might want to have several in order.
        self.human_db_list = []
        self.human_db_list.append(StarGenerator(gpu=False, add_gaussian_noise_pose = False, random_pose = False, random_shape=True))
        self.human_db_list.append(StarGenerator(gpu=False, add_gaussian_noise_pose = False, random_pose = False, random_shape=True, bent_human_pose=True, from_generator=self.human_db_list[0]))
        self.human_db_list.append(StarGenerator(gpu=False, add_gaussian_noise_pose = True, random_pose = False, random_shape=True, from_generator=self.human_db_list[0]))
        self.probas = [0.5, 0.25, 0.25]
        self.star_is_initialized = True



    def __len__(self):
        if self.source_and_target is None:
            return self.len_pairs
        return len(self.source_and_target)

    def get_scales(self):
        random_scale = self.args.random_scale
        scale = {True: 1, False: 1}
        # self.pair_inds = pair_inds

        if random_scale == 'target':
            scale[False] = 1.15 - random.random() * 0.15 * 2
        elif random_scale == 'source':
            scale[True] = 1.15 - random.random() * 0.15 * 2
        elif random_scale == 'both':
            scale[True] = 1.15 - random.random() * 0.15 * 2
            scale[False] = 1.15 - random.random() * 0.15 * 2
        elif random_scale == 'same':
            scale_val = 1.15 - random.random() * 0.15 * 2
            scale[False] = scale_val
            scale[True] = scale_val
        else:
            assert random_scale == 'none', print(f'got incorrect scale argument: {random_scale}')
        return scale


    def check_if_common_faces_is_saved(self, path, faces):
         if not path.is_file():
            np.save(path.as_posix(), faces)

    def get_random_generator(self, source=0):
        # Generate one human in random pose and shape
        if not self.star_is_initialized:
            self.initialize_star()

        from dataset_generation.human_db.star_generator import gender_to_int
        # Generate humans on the fly
        dataset_choice = np.random.choice(3,1,self.probas)[0]
        points_s, pose_s, shape_s, trans_s, gender_s = next(self.human_db_list[dataset_choice])
        folder_params_s = join(self.directory, source)
        os.makedirs(folder_params_s, exist_ok=True)
        np.save(f"{folder_params_s}/pose.npy", pose_s.cpu().numpy())
        np.save(f"{folder_params_s}/shape.npy", shape_s.cpu().numpy())
        np.save(f"{folder_params_s}/gender.npy", np.array(gender_to_int(gender_s)))
        np.save(f"{folder_params_s}/vertices.npy", points_s.cpu().numpy())
        if  self.args.store_faces_per_sample:
            np.save(f"{folder_params_s}/faces.npy", self.human_db_list[dataset_choice].faces)
        else:
            path_common = Path(folder_params_s).parent / "faces.npy"
            path = Path(folder_params_s) / "faces.npy"
            self.check_if_common_faces_is_saved(path_common, self.human_db_list[dataset_choice].faces)
            try:
                path.symlink_to(path_common)
            except FileExistsError:
                pass
            # make a symbolic link
        return points_s.cpu().numpy(), np.expand_dims(self.human_db_list[dataset_choice].faces, 0), dataset_choice,gender_s,pose_s,shape_s,trans_s

    def get_item_default(self,ind, verbose=False):
        source = None
        target = None
        # Single source single target
        source_index ,target_index = self.source_and_target[ind]
        for i,target in enumerate(target_index):
            if Path(target).suffix in [ '.obj' , '.off', '.ply']:
                # NOTE: Below caches the vertices/faces + creates the directory
                self.obj_to_npy(Path(join(self.directory, target)), ind)
                target_index[i] = target[:-4]

        if Path(source_index).suffix  in [ '.obj' , '.off', '.ply']:
            # NOTE: Below caches the vertices/faces + creates the directory
            self.obj_to_npy(Path(join(self.directory, source_index)), ind)
            source_index = source_index[:-4]

        # print(source_index, target_index)
        scales = self.get_scales()

        # ==================================================================
        # LOAD SOURCE
        if self.source is not None and self.unique_source:
            source = self.source
        else:
            source = SourceMesh(source_index, join(self.directory, 'cache', f"{source_index}_{ind}"), self.source_keys, scales[True], self.ttype, use_wks = not self.args.no_wks,
                                random_centering=(self.train and self.args.random_centering),  cpuonly=self.cpuonly, init=self.args.init,
                                initjinput = self.args.initjinput, fft=self.args.fft, fft_dim=self.args.fft_dim,
                                flatten=self.args.layer_normalization == "FLATTEN")
            source.load()
            self.source = source

        # ==================================================================
        # LOAD TARGET
        target = BatchOfTargets(target_index, [join(self.directory, 'cache', target_index[i]) for i in range(len(target_index))], self.target_keys, scales[False], self.ttype)
        target.load()
        return source, target

    def check_if_files_exist(self, paths):
        exist = True
        for path in paths:
            exist = (exist and path.is_file())
        return exist

    def obj_to_npy(self, path, ind):
        from meshing.io import PolygonSoup
        from meshing.mesh import Mesh
        # NOTE: All mesh data should be saved into 'cache'
        directory_name, basename = os.path.split(os.path.join(os.path.splitext(path)[0]))
        directory = os.path.join(directory_name, "cache", f"{basename}_{ind}")

        if not os.path.exists(join(directory , "vertices.npy")) and not  os.path.exists(join(directory, "faces.npy")):
            os.makedirs(directory, exist_ok=True)
            soup = PolygonSoup.from_obj(path)
            mesh = Mesh(soup.vertices, soup.indices)
            np.save(join(directory , "vertices.npy"), mesh.vertices)
            np.save(join(directory , "faces.npy"), mesh.faces)

    def __getitem__(self,ind, verbose=False):
        start = time.time()
        if self.args.experiment_type == "DEFAULT":
            if self.args.dataset_fail_safe:
                try:
                    data_sample = self.get_item_default(ind)
                except:
                    print(f"Data sample {ind} was not loaded properly")
                    logging.exception("Oops in Dataloader:")
                    data_sample = self.get_item_default(0)
            else:
                data_sample = self.get_item_default(ind)
        if verbose:
            print(f"DATALOADER : loaded sample in {time.time() - start}")

        return data_sample
    def get_point_dim(self):
        return self[0][0].get_point_dim()

    @staticmethod
    def dataset_from_files(source:str, targets:list = [None], source_keys=None, target_keys=None, ttype=None, args=None, cpuonly=False):
        pairs = [[source, target] for target in targets]
        dataset =  DeformationDataset(s_and_t=pairs,source_keys=source_keys,target_keys=target_keys,ttype=ttype,args=args, cpuonly=cpuonly)
        dataset[0]
        return dataset
