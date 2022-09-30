import os

import numpy as np
import torch
from easydict import EasyDict

import MeshProcessor


class BatchOfTargets:
    '''
    datastructure for the target mappings, in batched form (one source can be mapped to multiple targets)
    '''

    def __init__(self, target_inds, target_dirs, extra_target_fields, scale, ttype):
        self.target_dirs = target_dirs
        self.target_inds = target_inds

        self.target_tensors = None
        self.center_target = True
        self.__target_global_translation_to_original = 0

        self.__extra_keys = extra_target_fields
        self.__loaded_data = {}
        self.ttype = ttype
        self.__random_scale = scale
        self.__target_global_translation_to_original = 0
        self.__target_mesh_centroid = None
        self.mesh_processors = {}

    def change_loaded_data(self,key, value):
        assert(key in self.__loaded_data), "Thibault: I told you so. Do you find this useful now?"
        if isinstance(value, np.ndarray):
            value  = torch.from_numpy(value)
        self.__loaded_data[key] = value.to(self.__loaded_data[key].device)


    def get_target_vertices(self):
        return self.target_tensors.vertices

    def get_loaded_data(self, key: str):
        return self.__loaded_data.get(key)

    def to(self, device):
        for attr in self.target_tensors.__dict__.keys():
            t = getattr(self.target_tensors, attr)
            setattr(self.target_tensors, attr, t.to(device))
        for key in self.__loaded_data.keys():
            self.__loaded_data[key] = self.__loaded_data[key].to(device)
        return self

    def load(self, V=None, F=None):
        self.target_tensors = EasyDict()
        self.V = V
        self.F = F
        self.target_tensors.vertices = self._load_tensor_batch("vertices", 'npy')

        for attr in self.target_tensors.__dict__.keys():
            t = getattr(self.target_tensors, attr)
            setattr(self.target_tensors, attr, torch.from_numpy(t).type(self.ttype))
        self.__target_mesh_centroid = torch.mean(self.target_tensors.vertices, axis=1).unsqueeze(1)

        if self.center_target:
            c = self.__target_mesh_centroid
            # self.target_tensors.samples -= c
            self.target_tensors.vertices -= c
            self.__target_global_translation_to_original = c
        scale = self.__random_scale
        self.target_tensors.vertices *= scale

        for key in self.__extra_keys:
            data = torch.from_numpy(self._load_tensor_batch(key)).type(self.ttype)
            if key == 'samples':
                if self.center_target:
                    data -= self.__target_mesh_centroid
                scale = self.__random_scale
                data *= scale
            self.__loaded_data[key] = data

    def _load_tensor_batch(self, name, file_type='npy'):
        tensors = []
        for i, f in enumerate(self.target_dirs):
            if f not in self.mesh_processors:
                use_wks = 'samples_wks' in self.__extra_keys
                if self.V is None:
                    if os.path.isdir(f):
                        self.mesh_processors[f] = MeshProcessor.MeshProcessor.meshprocessor_from_directory(f, self.ttype,  load_wks_samples=use_wks, load_wks_centroids=use_wks)
                    else:
                        self.mesh_processors[f] = MeshProcessor.MeshProcessor.meshprocessor_from_file(f, self.ttype,  load_wks_samples=use_wks, load_wks_centroids=use_wks)
                else:
                    self.mesh_processors[f] = MeshProcessor.MeshProcessor.meshprocessor_from_array(self.V[i], self.F[i], f, self.ttype,  load_wks_samples=use_wks, load_wks_centroids=use_wks)

            processor = self.mesh_processors[f]
            tensor = processor.get_data(name, file_type)
            tensors.append(tensor)
        return np.stack(tensors, axis=0)

    def pin_memory(self):
        # for attr in self.target_tensors.__dict__.keys():
        #     getattr(self.target_tensors, attr).pin_memory()

        # for key in self.__loaded_data.keys():
        #     self.__loaded_data[key].pin_memory()
        return self

    def get_vertices(self):
        return self.target_tensors.vertices