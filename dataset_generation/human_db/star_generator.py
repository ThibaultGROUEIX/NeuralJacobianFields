import sys
import numpy as np
import pickle
import os
from os.path import join
from pathlib import Path
sys.path.append(join(Path(__file__).parent, "STAR"))
from star.pytorch.star import STAR
import torch
from typing import Union
from pathlib import Path
from torch.utils.data import IterableDataset
from tqdm import tqdm
import copy

class StarGenerator(IterableDataset):
    """ This class acts as a generator to generate a star model from a db"""
    def __init__(self, add_gaussian_noise_pose=True, random_pose=False, random_shape=True, bent_human_pose=False, gpu=True, from_generator = None) -> None:
        self.num_betas=10
        self.batch_size=1
        if from_generator is not None:
            # Use pointers to avoid reloading the same memory chunk for different versions of this dataset            self.star_layer_neutral = from_generator.star_layer_neutral
            self.star_layer_neutral = from_generator.star_layer_neutral
            self.star_layer_male = from_generator.star_layer_male
            self.star_layer_female = from_generator.star_layer_female
            self.database = from_generator.database
            self.db_betas_male = from_generator.db_betas_male
            self.db_betas_female = from_generator.db_betas_female
            self.db_poses = from_generator.db_poses
            self.num_poses = from_generator.num_poses
            self.faces  = from_generator.faces 
        else:
            self.star_layer_neutral = STAR(
                gender='neutral',
                num_betas=self.num_betas)
            self.star_layer_male = STAR(
                gender='male',
                num_betas=self.num_betas)
            self.star_layer_female = STAR(
                gender='female',
                num_betas=self.num_betas)

            # self.database = np.load(join(Path(__file__).parent, "surreal_database", "smpl_data.npz"))
            self.database = dict(np.load(join(Path(__file__).parent, "surreal_database", "smpl_data.npz")))
            self.db_betas_male = self.database['maleshapes']
            self.db_betas_female = self.database['femaleshapes']
            self.db_poses = [i for i in self.database.keys() if "pose" in i]
            self.num_poses = sum(np.shape(self.database[i])[0] for i in self.db_poses)
            self.faces = np.array(self.star_layer_male.faces.cpu().numpy())

        self.add_gaussian_noise_pose = add_gaussian_noise_pose
        self.random_pose = random_pose
        self.random_shape = random_shape
        self.bent_human_pose = bent_human_pose
        self.gpu = gpu
        if not self.gpu:
            self.star_layer_neutral = self.star_layer_neutral.to('cpu')
            self.star_layer_male = self.star_layer_male.to('cpu')
            self.star_layer_female = self.star_layer_female.to('cpu')

        # print(f"Number of video sequences {len(self.db_poses)}")
        print('Number of poses ' + str(self.num_poses))
        # print('Number of male betas ' + str(np.shape(self.db_betas_male)[0]))
        # print('Number of female betas ' + str(np.shape(self.db_betas_female)[0]))

    def random_human_shape_from_database(self):
        gender = np.random.randint(3)
        if gender == 0:
            gender = "female"
            beta_id = np.random.randint(len(self.db_betas_female))
            beta = self.db_betas_female[beta_id]
        elif gender == 1:
            gender = "neutral"
            beta_id = np.random.randint(len(self.db_betas_male))
            beta = self.db_betas_male[beta_id]
        else:
            gender = "male"
            beta_id = np.random.randint(len(self.db_betas_male))
            beta = self.db_betas_male[beta_id]
        return gender,beta

    def random_human_shape_from_gaussian(self):
        gender = np.random.randint(3)
        if gender == 0:
            gender = "female"
        elif gender == 1:
            gender = "neutral"
        else:
            gender = "male"
        beta = np.random.rand(10) * 8 - 4
        return gender,beta

    
    def sample_db(self,num_poses  = 32):
        """ This function :
            * Shape : randomly samples a beta
            * Pose : randomy sample a sequence from cmu, then randomly sample a frame in that sequence
            Return pose, shape
        """
        # Randomly select the gender
        gender,beta = self.random_human_shape_from_gaussian()
        sequence_id = np.random.randint(len(self.db_poses),size=1)
        poses_for_the_full_sequence = self.database[self.db_poses[sequence_id[0]]]
        frame_id = np.random.randint(len(poses_for_the_full_sequence))
        if torch.utils.data.get_worker_info() is not None:
            workers_info = torch.utils.data.get_worker_info()
            id_worker = workers_info.id
            num_worker = workers_info.num_workers
            seed = workers_info.seed
        else:
            id_worker = 0
            num_worker = "main"
            seed = "main"

        print(f"new human  : {gender} {sequence_id} {frame_id} from worker {id_worker}/{num_worker} with seed {seed}")
        pose = poses_for_the_full_sequence[frame_id]

        if self.add_gaussian_noise_pose:
            # Optionally use gaussian noise
            pose[0:3] = 0
            pose[3:] += 0.1 * np.random.randn(69)

        if self.random_pose:
            pose[3:] =  np.random.randn(69)

        if self.random_shape:
            beta =   np.random.rand(10)*8 - 4

        if self.bent_human_pose:
            pose = self.generate_benthuman(pose)
        return gender, pose, torch.from_numpy(beta)


    def generate_benthuman(self, pose):
        ## Assign random pose parameters except for certain ones to have random bent humans
        pose[:] = pose
        a = np.random.randn(12)
        pose[1] = 0
        pose[2] = 0
        pose[3] = -1.0 + 0.1*a[0]
        pose[4] = 0 + 0.1*a[1]
        pose[5] = 0 + 0.1*a[2]
        pose[6] = -1.0 + 0.1*a[0]
        pose[7] = 0 + 0.1*a[3]
        pose[8] = 0 + 0.1*a[4]
        pose[9] = 0.9 + 0.1*a[6]
        pose[0] = - (-0.8 + 0.1*a[0] )
        pose[18] = 0.2 + 0.1*a[7]
        pose[43] = 1.5 + 0.1*a[8]
        pose[40] = -1.5 + 0.1*a[9]
        pose[44] = -0.15 
        pose[41] = 0.15
        pose[48:54] = 0
        return pose


    def star_forward(self,gender:str = "female", pose: Union[torch.Tensor, int] = 0, shape: Union[torch.Tensor, int] = 0, trans: Union[torch.Tensor, int] = 0):
        """ run the STAR model. default to canonical pose if no arguments are provided."""
        if isinstance(pose, int) and self.gpu:
            pose = torch.cuda.FloatTensor(np.zeros((1,72)))
        elif isinstance(pose, int) and not self.gpu:
            pose = torch.FloatTensor(np.zeros((1,72)))
        elif isinstance(pose, np.ndarray):
            pose = torch.from_numpy(pose)

        if isinstance(shape, int):
            shape = np.array([
                        np.array([ 2.25176191, -3.7883464, 0.46747496, 3.89178988,
                                2.20098416, 0.26102114, -3.07428093, 0.55708514,
                                -3.94442258, -2.88552087])])
            if self.gpu:
                shape = torch.cuda.FloatTensor(shape)
            else:
                shape = torch.FloatTensor(shape)
        
        if isinstance(trans, int) and not self.gpu:
            trans = torch.FloatTensor(np.zeros((1,3)))
        if isinstance(trans, int) and self.gpu:
            trans = torch.cuda.FloatTensor(np.zeros((1, 3)))

        # Put pose parameters on the correct device
        if self.gpu:
            pose = pose.cuda()
        else:
            pose = pose.cpu()

        device = pose.device
        pose = pose.float()

        # Put ALL parameters on the SAME device and fix dimension issues
        if pose.ndim == 1:
            pose = pose.unsqueeze(0)
        if shape.ndim == 1:
            shape = shape.unsqueeze(0)
        shape = shape.to(device).float()


        if trans.ndim == 1:
            trans = trans.unsqueeze(0).to(device)
        trans = trans.to(device).float()
        NO_GLOBAL_ROT = True

        # Forward
        if NO_GLOBAL_ROT:
            pose[:,0:3] = 0

        with torch.no_grad():
            if gender == "female":
                points = self.star_layer_female.forward(pose, shape, trans)
            # Forward
            if gender == "male":
                points = self.star_layer_male.forward(pose, shape, trans)
            # Forward
            if gender == "neutral":
                points = self.star_layer_neutral.forward(pose, shape, trans)

        # print(gender)
        return points, pose, shape, trans, gender

    def __next__(self):
        return self.star_forward(*self.sample_db())

    def __iter__(self):
        return self

    def save_mesh(self, points: torch.Tensor, path: str) -> None:
        """ save output of model"""
        import igl
        V = np.array(points[0].cpu().squeeze().float().numpy())
        F = self.faces
        igl.write_triangle_mesh(path,np.array(V.tolist()),np.array(F.tolist()) )
    
    def star_forward_in_tpose(self,gender:str = "female", pose: Union[torch.Tensor, int] = 0, shape: Union[torch.Tensor, int] = 0, trans: Union[torch.Tensor, int] = 0):
        pose = 0 * pose
        return self.star_forward(gender=gender, pose=pose,shape=shape, trans=trans)
        
    def __len__(self):
        """return number of poses"""
        return self.num_poses



def save_mesh(i):
    sg.save_mesh(next(sg)[0], f"db/5_{i}.obj")


def save_parallel():
    from joblib import delayed, Parallel
    global sg 
    sg = StarGenerator(add_gaussian_noise_pose = False, random_pose = False, random_shape=True)
    Parallel(n_jobs=2)(delayed(save_mesh)(i) for i in tqdm(range(100)))


def gender_to_int(gender):
    if gender == "female":
        return 0
    if gender == "male":
        return 1
    if gender == "neutral":
        return 2

if __name__ == '__main__':
    counter = 0 

    db_test = False

    if db_test:
        factor = 10
        folder = "db_test"
        folder_params = "db_test_params"
    else:
        factor = 1000
        folder = "db_train"
        folder_params = "db_train_params"

    os.makedirs(f"/sensei-fs/users/groueix/{folder_params}", exist_ok=True)
    os.makedirs(f"/sensei-fs/users/groueix/{folder}", exist_ok=True)


    def save(sg, counter):

        if os.path.exists(f"/sensei-fs/users/groueix/{folder}/{counter:08d}.obj") and os.path.exists(f"/sensei-fs/users/groueix/{folder}/{(counter+1):08d}.obj") and os.path.exists(f"/sensei-fs/users/groueix/{folder_params}/{counter:08d}_pose.npy") and os.path.exists(f"/sensei-fs/users/groueix/{folder_params}/{(counter+1):08d}_pose.npy") and os.path.exists(f"/sensei-fs/users/groueix/{folder_params}/{counter:08d}_shape.npy") and os.path.exists(f"/sensei-fs/users/groueix/{folder_params}/{(counter+1):08d}_shape.npy") and os.path.exists(f"/sensei-fs/users/groueix/{folder_params}/{counter:08d}_gender.npy") and os.path.exists(f"/sensei-fs/users/groueix/{folder_params}/{(counter+1):08d}_gender.npy"):
            counter = counter+ 2
            return counter
        else:
            print(f"DOES NOT EXIST {counter}")
            # counter = counter+ 2
            # return counter

            points, pose, shape, trans, gender = next(sg)
            np.save(f"/sensei-fs/users/groueix/{folder_params}/{counter:08d}_pose.npy", pose.cpu().numpy())
            np.save(f"/sensei-fs/users/groueix/{folder_params}/{counter:08d}_shape.npy", shape.cpu().numpy())
            np.save(f"/sensei-fs/users/groueix/{folder_params}/{counter:08d}_gender.npy", np.array(gender_to_int(gender)))
            sg.save_mesh(points, f"/sensei-fs/users/groueix/{folder}/{counter:08d}.obj")
            counter+=1
            points, pose, shape, trans, gender = sg.star_forward_in_tpose(gender=gender, pose=pose*0,shape=shape, trans=trans)
            np.save(f"/sensei-fs/users/groueix/{folder_params}/{counter:08d}_pose.npy", pose.cpu().numpy())
            np.save(f"/sensei-fs/users/groueix/{folder_params}/{counter:08d}_shape.npy", shape.cpu().numpy())
            np.save(f"/sensei-fs/users/groueix/{folder_params}/{counter:08d}_gender.npy", np.array(gender_to_int(gender)))
            sg.save_mesh(points, f"/sensei-fs/users/groueix/{folder}/{counter:08d}.obj")
            counter+=1
            return counter

    sg = StarGenerator(add_gaussian_noise_pose = False, random_pose = False, random_shape=True)
    for _ in tqdm(range(150*factor)):
        counter = save(sg, counter)
    sg = StarGenerator(add_gaussian_noise_pose = False, random_pose = False, random_shape=True, bent_human_pose=True)
    for _ in tqdm(range(75*factor)):
        counter = save(sg, counter)
    sg = StarGenerator(add_gaussian_noise_pose = True, random_pose = False, random_shape=True)
    for _ in tqdm(range(75*factor)):
        counter = save(sg, counter)


    # # Catastrophic non realistic
    # sg = StarGenerator(add_gaussian_noise_pose = False, random_pose = True, random_shape=True)
    # for i in range(20):
    #     sg.save_mesh(next(sg)[0], f"/sensei-fs/users/groueix/db/1_{counter}.obj")

