import sys
import time

import igl
import numpy as np
import pickle
import os
from smplpytorch.pytorch.smpl_layer import SMPL_Layer
import torch
from typing import Union
import threading
#import multiprocessing as mp
import queue
import math
class SMPLGenerator:
    """ This class acts as a generator to generate a smpl model from a db"""
    def __init__(self,POSES_PER_SHAPE) -> None:
        # Models
        self.__poses_per_shape = POSES_PER_SHAPE
        model_root = os.path.abspath('smplmodels')
        self.smpl_layer_neutral = SMPL_Layer(
            center_idx=0,
            gender='neutral',
            model_root=model_root)
        self.smpl_layer_male = SMPL_Layer(
            center_idx=0,
            gender='male',
            model_root=model_root)
        self.smpl_layer_female = SMPL_Layer(
            center_idx=0,
            gender='female',
            model_root=model_root)
        self.smpl_layer_male = self.smpl_layer_male.cuda()
        self.smpl_layer_female = self.smpl_layer_female.cuda()
        self.smpl_layer_neutral = self.smpl_layer_neutral.cuda()

        self.num_betas = 10
        self.batch_size = 1

        # Database
        self.database = np.load("smpl_data.npz")
        self.db_betas_male = self.database['maleshapes']
        self.db_betas_female = self.database['femaleshapes']
        self.db_poses = [i for i in self.database.keys() if "pose" in i]
        self.num_poses = sum(np.shape(self.database[i])[0] for i in self.db_poses)

        print(f"Number of video sequences {len(self.db_poses)}")
        print('Number of poses ' + str(self.num_poses))
        print('Number of male betas ' + str(np.shape(self.db_betas_male)[0]))
        print('Number of female betas ' + str(np.shape(self.db_betas_female)[0]))

    def random_human_shape_from_database(self):
        gender = np.random.randint(2)
        if gender == 0:
            gender = "female"
            beta_id = np.random.randint(len(self.db_betas_female))
            beta = self.db_betas_female[beta_id]
        else:
            gender = "male"
            beta_id = np.random.randint(len(self.db_betas_male))
            beta = self.db_betas_male[beta_id]
        return gender,beta

    def random_human_shape_from_gaussian(self):
        gender = np.random.randint(2)
        beta = np.random.randn(10) * 4
        return gender,beta

    def sample_db(self,num_poses  = 32, add_gaussian_noise: bool = False):
        """ This function :
            * Shape : randomly samples a beta
            * Pose : randomy sample a sequence from cmu, then randomly sample a frame in that sequence
            Return pose, shape
        """
        # Randomly select the gender
        gender,beta = self.random_human_shape_from_gaussian()
        sequence_ids = np.random.randint(len(self.db_poses),size=num_poses)
        poses = []
        for i,sequence_id in enumerate(sequence_ids):
            poses_for_the_full_sequence = self.database[self.db_poses[sequence_ids]]
            frame_id = np.random.randint(len(poses_for_the_full_sequence))
            pose = poses_for_the_full_sequence[frame_id]

            if use_gaussian_noise:
                # Optionally use gaussian noise
                pose[0:3] = 0
                pose[3:] += 0.3 * np.random.randn(69)
            poses.append(torch.from_numpy(pose))
        return gender, poses, torch.from_numpy(beta)

    def model_forward(self, gender:str = "female", poses: Union[torch.Tensor, int] = 0, shape: Union[torch.Tensor, int] = 0, trans: Union[torch.Tensor, int] = 0, gpu: bool = True):
        """ run the SMPL model. default to canonical pose if no arguments are provided."""
        if isinstance(poses, int):
            poses = torch.cuda.FloatTensor(np.zeros((1, 72)))
        if isinstance(shape, int):
            shape = np.array([
                np.array([2.25176191, -3.7883464, 0.46747496, 3.89178988,
                          2.20098416, 0.26102114, -3.07428093, 0.55708514,
                          -3.94442258, -2.88552087])])
            if gpu:
                shape = torch.cuda.FloatTensor(shape)
            else:
                shape = torch.FloatTensor(shape)

        if isinstance(trans, int) and not gpu:
            trans = torch.FloatTensor(np.zeros((1, 3)))
        if isinstance(trans, int) and gpu:
            trans = torch.cuda.FloatTensor(np.zeros((1, 3)))
        # Put pose parameters on the correct device
        poses.insert(0,poses[0]*0)
        for i,pose in enumerate(poses):
            if gpu:
                pose = pose.cuda()
            device = pose.device
            pose = pose.float()

            # Put ALL parameters on the SAME device and fix dimension issues
            if pose.ndim == 1:
                pose = pose.unsqueeze(0)
            poses[i] = pose
        if shape.ndim == 1:
            shape = shape.unsqueeze(0)
        shape = shape.to(device).float()

        if trans.ndim == 1:
            trans = trans.unsqueeze(0).to(device)
        trans = trans.to(device).float()

        # Forward
        points_list = []
        NO_GLOBAL_ROT = True
        for pose in poses:
            if NO_GLOBAL_ROT:
                pose[:,0:3] = 0
            if gender == "female":
                points = self.smpl_layer_female.forward(pose, shape, trans)
            # Forward
            if gender == "male":
                points = self.smpl_layer_male.forward(pose, shape, trans)
            # Forward
            if gender == "neutral":
                points = self.smpl_layer_neutral.forward(pose, shape, trans)
            points_list.append(points)
        return points_list, poses, shape, trans, gender

    def __next__(self):
        return self.model_forward(*self.sample_db(self.__poses_per_shape))

    def __iter__(self):
        return self

    def __len__(self):
        """return number of poses"""
        return self.num_poses

    def __getitem__(self, index):
        """return the indexth pose"""
        # Find corresponding sequence
        i=-1
        cumsum_pose = 0
        while index < cumsum_pose:
            i+=1
            cumsum_pose += np.shape(self.database[self.db_poses[i]])[0]
        pose = self.database[self.db_poses[i]]
        # Find frame
        pose = pose[index-cumsum_pose]
        return pose

    def save_mesh(self, points: torch.Tensor, path: str) -> None:
        """ save output of model"""
        #import trimesh
        import igl
        #mesh = trimesh.Trimesh(points[0].cpu().squeeze().numpy(), self.smpl_layer_male.th_faces.cpu())
        #mesh.export(path)
        V = np.array(points[0].cpu().squeeze().float().numpy())
        F = np.array(self.smpl_layer_male.th_faces.cpu().numpy())
        igl.write_triangle_mesh(path,np.array(V.tolist()),np.array(F.tolist()) )

def produce(points_queue : queue.Queue,SHAPES,
             sg):


        next(sg)
        it = iter(sg)

        for i in range(SHAPES):

            points_list, poses, shape, trans, gender = next(it)
            npoints_list = []
            for points in points_list:
                p = points[0].cpu().squeeze().float().numpy()
                npoints_list.append(p)
            # out = {}
            # out['points_list'] = points_list
            # out['poses'] = poses
            # out['shape'] = shape
            # out['trans'] = trans
            # out['gender'] = gender
            nposes = []
            for pose in poses:
                nposes.append(pose.cpu().numpy())
                a = {'points_list':npoints_list,'poses' :nposes, 'shape':shape.cpu().numpy(), 'trans': trans.cpu().numpy(), 'gender':gender}
            points_queue.put(a)
        points_queue.put(-1)
def one_file_write(points_list,poses,shape,trans,gender,faces,directory,counter):
    for i in range(len(points_list)):
        points = points_list[i]
        pose = poses[i]
        igl.write_obj(os.path.join(directory, f"{counter:07}.obj"), points, faces)
        np.save(os.path.join(directory, f"{counter:07}_pose"), pose)
        np.save(os.path.join(directory, f"{counter:07}_shape"), shape)
        np.save(os.path.join(directory, f"{counter:07}_trans"), trans)
        np.save(os.path.join(directory, f"{counter:07}_gender"), gender)
        counter += 1
def write(faces,directory,q:queue.Queue,N_WRITERS,SOURCE_TPOSE,TARGET_TPOSE,
    SOURCES = 16):
    counter = 0
    pairs = []
    all_groups = []
    last_counter = 0
    thread_pool = []
    while(True):
        stime = time.time()
        group_inds = []
        a= q.get()
        if a == -1:
            N_WRITERS -= 1
            if N_WRITERS == 0:
                break
            continue
        points_list = a['points_list']
        poses = a['poses']
        shape = a['shape']
        trans = a['trans']
        gender = a['gender']
        t = threading.Thread(target=one_file_write,args=[points_list,poses,shape,trans,gender,faces,directory,counter])
        t.start()
        thread_pool.append(t)
        MAX_POOL_SIZE = 4
        while(len(thread_pool)>MAX_POOL_SIZE):
            for i in range(len(thread_pool)):
                if not thread_pool[i].is_alive():
                    del thread_pool[i]
                    break
        for i in range(len(points_list)):
            # points = points_list[i]
            # pose = poses[i]
            # igl.write_obj( os.path.join(directory, f"{counter:07}.obj"),points,faces)
            # np.save(os.path.join(directory, f"{counter:07}_pose"), pose)
            # np.save(os.path.join(directory, f"{counter:07}_shape"), shape)
            # np.save(os.path.join(directory, f"{counter:07}_trans"), trans)
            # np.save(os.path.join(directory, f"{counter:07}_gender"), gender)
            group_inds.append(counter)
            counter += 1
        all_groups.append(group_inds)
        if not SOURCE_TPOSE and TARGET_TPOSE:
            for j in range(1, len(group_inds)):
                pair = (group_inds[j], group_inds[0])
                pairs.append(pair)
        elif SOURCE_TPOSE and not TARGET_TPOSE:
            for j in range(1, len(group_inds)):
                pair = (group_inds[0], group_inds[j])
                pairs.append(pair)
        elif not SOURCE_TPOSE and not TARGET_TPOSE:
            for j in range(1, max(SOURCES, len(group_inds))):
                for k in range(1, len(group_inds)):
                    if j == k:
                        continue
                    pair = (group_inds[j], group_inds[k])
                    pairs.append(pair)
        ttime = time.time() - stime
        avg_meshes = math.floor((counter-last_counter)/ttime)
        last_counter = counter
        print(f'iter: {counter}, writing {avg_meshes:.2f} meshes per sec')
    import json
    data = {'pairs': pairs, 'groups': all_groups}
    with open(os.path.join(directory, 'data.json'), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    for t in thread_pool:
        t.join()      
def new_main(target_dir,SOURCE_TPOSE,TARGET_TPOSE,
        SHAPES = 3000,
    SOURCES = 16,
    POSES_PER_SHAPE = 32):
    directory = target_dir
    if not os.path.exists(directory):
        os.mkdir(directory)
    if not os.path.exists(directory):
        os.mkdir(directory)
    sg = SMPLGenerator(POSES_PER_SHAPE)
    next(sg)
    faces = sg.smpl_layer_male.th_faces.cpu().numpy()
    N = 1
    points_queue = queue.Queue(N*10)

    write_thread = threading.Thread(target=write,
                                    args=[faces, directory, points_queue,N, SOURCE_TPOSE, TARGET_TPOSE, SOURCES])
    write_thread.start()
    processes = []
    for i in range(N):

        p = threading.Thread(target=produce,args=(points_queue,math.ceil(SHAPES/N),sg))
        processes.append(p)
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    write_thread.join()
    while(not points_queue.empty()):
        points_queue.get()

if __name__ == '__main__':
    new_main('data/smpl_big',False,True)