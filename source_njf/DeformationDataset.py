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
        if len(source_target) == 1:
            # This flag will avoid reloading the source at each iteration, since the source is always the same.
            self.unique_source = True
        self.source_and_target = []
        for s,ts in source_target.items():
            chunks = np.split(ts, np.arange(args.targets_per_batch,len(ts),args.targets_per_batch))
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

    def get_tpose_generator(self, target=0, dataset_choice=0, gender_s=0, pose_s=0, shape_s=0, trans_s=0):
        if not self.star_is_initialized:
            self.initialize_star()
        
        from dataset_generation.human_db.star_generator import gender_to_int

        points_t, pose_t, shape_t, trans_t, gender_t = self.human_db_list[dataset_choice].star_forward_in_tpose(gender=gender_s, pose=pose_s*0,shape=shape_s, trans=trans_s)
        folder_params_t = join(self.directory, target)
        os.makedirs(folder_params_t, exist_ok=True)
        np.save(f"{folder_params_t}/pose.npy", pose_t.cpu().numpy())
        np.save(f"{folder_params_t}/shape.npy", shape_t.cpu().numpy())
        np.save(f"{folder_params_t}/gender.npy", np.array(gender_to_int(gender_t)))
        np.save(f"{folder_params_t}/vertices.npy", points_t.cpu().numpy())
        if  self.args.store_faces_per_sample:
            np.save(f"{folder_params_t}/faces.npy", self.human_db_list[dataset_choice].faces)
        else:
            path_common = Path(folder_params_t).parent / "faces.npy"
            path = Path(folder_params_t) / "faces.npy"
            self.check_if_common_faces_is_saved(path_common, self.human_db_list[dataset_choice].faces)
            try:
                # This is critical because the link might already exist which throws an error.
                path.symlink_to(path_common)
            except FileExistsError:
                pass
        return  points_t.cpu().numpy(), np.expand_dims(self.human_db_list[dataset_choice].faces, 0), dataset_choice,gender_s,pose_s,shape_s,trans_s


    def get_item_tpose(self,ind, verbose=False):
        source = None
        target = None
        # Single source single target
        source_index ,target_index = self.s_and_t[ind]
        if Path(source_index).suffix == '.obj':
            self.obj_to_npy(Path(join(self.directory, source_index)))
            source_index = source_index[:-4]

        print(source_index, target_index)
        scales = self.get_scales()

        # ==================================================================
        # LOAD SOURCE
        source = SourceMesh(source_index, join(self.directory, source_index), self.source_keys, scales[True], self.ttype, use_wks= not self.args.no_wks, random_centering=(self.train and self.args.random_centering),  cpuonly=self.cpuonly)
        
        if (Path(join(self.directory, source_index))/"vertices.npy").is_file():
            print(f"loading source {source_index}")
            # Assume data has been generated
            source.load()
        else:

            try:
                print(f"creating sources {source_index}")
                points_s, faces_s, dataset_choice,gender_s,pose_s,shape_s,trans_s = self.get_random_generator(source.source_dir)
                source.load(points_s, faces_s)
            except WaveKernelSignatureError:
                print(f"creating sources {source_index}, {target_index} again after WKS computation error")
                points_s, faces_s,  dataset_choice,gender_s,pose_s,shape_s,trans_s = self.get_random_generator(source.source_dir)
                source.load(points_s, faces_s)
        
        # ==================================================================
        # LOAD TARGET
        if target_index is not None:
            # Don't do this for some inference, like Tposing
            target = BatchOfTargets(target_index, [join(self.directory, target_index)], self.target_keys, scales[False], self.ttype)
            
            if (Path(join(self.directory, target_index))/"vertices.npy").is_file():
                print(f"loading target {target_index}")
                target.load()
            else:
                try:
                    print(f"creating targets {target_index}")
                    source = SourceMesh(source_index, join(self.directory, source_index), self.source_keys, scales[True], self.ttype, use_wks= not self.args.no_wks, random_centering=(self.train and self.args.random_centering),  cpuonly=self.cpuonly)
                    points_s, faces_s,  dataset_choice,gender_s,pose_s,shape_s,trans_s = self.get_random_generator(source.source_dir)
                    source.load(points_s, faces_s)
                    points_t, faces_t, _,_,_,_,_ = self.get_tpose_generator(target.target_dirs[0], dataset_choice,gender_s,pose_s,shape_s,trans_s)
                    target.load(points_t, faces_t)
                except WaveKernelSignatureError:
                    print(f"creating targets {target_index} again after WKS computation error")
                    source = SourceMesh(source_index, join(self.directory, source_index), self.source_keys, scales[True], self.ttype, use_wks= not self.args.no_wks, random_centering=(self.train and self.args.random_centering),  cpuonly=self.cpuonly)
                    points_s, faces_s,  dataset_choice,gender_s,pose_s,shape_s,trans_s = self.get_random_generator(source.source_dir)
                    source.load(points_s, faces_s)
                    points_t, faces_t, _,_,_,_,_  = self.get_tpose_generator(target.target_dirs[0], dataset_choice,gender_s,pose_s,shape_s,trans_s)
                    target.load(points_t, faces_t)

        return source, target


    def get_item_register_template(self,ind, verbose=False):
        source = None
        target = None
        # Single source single target
        source_index ,target_index = self.source_and_target[ind]
        for i,target in enumerate(target_index):
            if Path(target).suffix == '.obj':
                self.obj_to_npy(Path(join(self.directory, target)))
                target_index[i] = target[:-4]

        
        if Path(source_index).suffix == '.obj':
            self.obj_to_npy(Path(join(self.directory, source_index)))
            source_index = source_index[:-4]

        # print(source_index, target_index)
        scales = self.get_scales()

        # ==================================================================
        # LOAD SOURCE
        if self.source is None:
            source = SourceMesh(source_index, join(self.directory, source_index), self.source_keys, scales[True], self.ttype, use_wks = not self.args.no_wks, random_centering=(self.train and self.args.random_centering),  cpuonly=self.cpuonly)
            source.load()
            self.source = source
        else:
            source = self.source

        # ==================================================================
        # LOAD TARGET
        # Don't do this for some inference, like Tposing
        target = BatchOfTargets(target_index, [join(self.directory, target_index[i]) for i in range(len(target_index))], self.target_keys, scales[False], self.ttype)
        
        try:
            #  self.check_if_files_exist([Path(join(self.directory, target_index[i]))/"vertices.npy" for i in range(len(target_index))]):
            # print(f"loading {target_index}")
            target.load()

        except Exception:

            done = False
            count = 0
            while (not done) and count < 5:
                try:
                    # The bug that can occur here is a single matrix bug in the eigen decomposition.
                    # it is critical to reinitialize BatchOfTargets because the mesh_processor attribute does not get updated if V and F change, and is responsible for the bug
                    target = BatchOfTargets(target_index, [join(self.directory, target_index[i]) for i in range(len(target_index))], self.target_keys, scales[False], self.ttype)
                    # WARNING : THIS ONLY CREATES A SINGLE TARGET -- needs to be adapted for single source multiple targets
                    print(f"creating {target_index}")
                    points_t = []
                    faces_t = []
                    for i in range(len(target.target_dirs)):
                        points_t_i, faces_t_i, _,_,_,_,_ = self.get_random_generator(target.target_dirs[i])
                        points_t.append(points_t_i)
                        faces_t.append(faces_t_i)

                    points_t = np.vstack(points_t)
                    faces_t = np.vstack(faces_t)
                    target.load(points_t, faces_t)
                    done = True

                except WaveKernelSignatureError:
                    import traceback
                    traceback.print_exc()
                    print(WaveKernelSignatureError)
                    print("Failure")
                    count = count + 1
            
            if not done:
                print(f"creating {target_index}")
                points_t = []
                faces_t = []
                for i in range(len(target.target_dirs)):
                    points_t_i, faces_t_i, _,_,_,_,_ = self.get_random_generator(target.target_dirs[i])
                    points_t.append(points_t_i)
                    faces_t.append(faces_t_i)

                points_t = np.vstack(points_t)
                faces_t = np.vstack(faces_t)
                target.load(points_t, faces_t)
        return source, target

    def get_item_default(self,ind, verbose=False):
        source = None
        target = None
        # Single source single target
        source_index ,target_index = self.source_and_target[ind]
        for i,target in enumerate(target_index):
            if Path(target).suffix in [ '.obj' , '.off', '.ply']:
                self.obj_to_npy(Path(join(self.directory, target)))
                target_index[i] = target[:-4]

        
        if Path(source_index).suffix  in [ '.obj' , '.off', '.ply']:
            self.obj_to_npy(Path(join(self.directory, source_index)))
            source_index = source_index[:-4]

        # print(source_index, target_index)
        scales = self.get_scales()

        # ==================================================================
        # LOAD SOURCE
        if self.source is not None and self.unique_source:
            source = self.source
        else:
            source = SourceMesh(source_index, join(self.directory, source_index), self.source_keys, scales[True], self.ttype, use_wks = not self.args.no_wks, random_centering=(self.train and self.args.random_centering),  cpuonly=self.cpuonly)
            source.load()
            self.source = source

        # ==================================================================
        # LOAD TARGET
        target = BatchOfTargets(target_index, [join(self.directory, target_index[i]) for i in range(len(target_index))], self.target_keys, scales[False], self.ttype)
        target.load()
        return source, target



    def check_if_files_exist(self, paths):
        exist = True
        for path in paths:
            exist = (exist and path.is_file())
        return exist

    def obj_to_npy(self, path):
        import igl
        directory_name = path.as_posix()[:-4]
        if not os.path.exists(join(directory_name , "vertices.npy")) and not  os.path.exists(join(directory_name , "faces.npy")):
            os.makedirs(directory_name, exist_ok=True)
            mesh = igl.read_triangle_mesh(path.as_posix())
            np.save(join(directory_name , "vertices.npy"), mesh[0] )
            np.save(join(directory_name , "faces.npy"), mesh[1] )


    def __getitem__(self,ind, verbose=False):
        start = time.time()
        if self.args.experiment_type == "TPOSE":
            data_sample =  self.get_item_tpose(ind)
        elif self.args.experiment_type == "REGISTER_TEMPLATE":
            data_sample = self.get_item_register_template(ind)
        elif self.args.experiment_type == "DEFAULT":
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
