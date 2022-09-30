import os

import numpy
import torch
import igl
import MeshProcessor
WKS_DIM = MeshProcessor.WKS_DIM
WKS_FACTOR = 1000
import numpy as np
import sys
import random
import time
class SourceMesh:
    '''
    datastructure for the source mesh to be mapped
    '''

    def __init__(self, source_ind, source_dir, extra_source_fields,
                 random_scale, ttype, use_wks=False, random_centering=False,
                cpuonly=False):
        self.__use_wks = use_wks
        self.source_ind = source_ind
        self.source_dir = source_dir
        self.centroids_and_normals = None
        self.center_source = True
        self.poisson = None
        self.__source_global_translation_to_original = 0
        self.__extra_keys = extra_source_fields
        self.__loaded_data = {}
        self.__ttype = ttype
        self.__random_scale = random_scale
        self.random_centering = random_centering
        self.source_mesh_centroid = None
        self.mesh_processor = None
        self.cpuonly = cpuonly

    def get_vertices(self):
        return self.source_vertices

    def get_global_translation_to_original(self):
        return self.__source_global_translation_to_original

    def vertices_from_jacobians(self, d):
        return self.poisson.solve_poisson(d)

    def jacobians_from_vertices(self, v):
        return self.poisson.jacobians_from_vertices(v)

    def restrict_jacobians(self, J):
        return self.poisson.restrict_jacobians(J)

    def get_loaded_data(self, key: str):

        return self.__loaded_data.get(key)

    def get_source_triangles(self):
        # if self.__source_triangles is None:
        #     self.__source_triangles = np.load(os.path.join(self.source_dir, 'faces.npy'))
        return self.mesh_processor.get_faces()

    def to(self, device):
        self.poisson = self.poisson.to(device)
        self.centroids_and_normals = self.centroids_and_normals.to(device)
        for key in self.__loaded_data.keys():
            self.__loaded_data[key] = self.__loaded_data[key].to(device)
        return self

    def __init_from_mesh_data(self):
        assert self.mesh_processor is not None
        self.mesh_processor.prepare_differential_operators_for_use(self.__ttype) #call 1
        self.source_vertices = torch.from_numpy(self.mesh_processor.get_vertices()).type(
            self.__ttype)
        if self.__random_scale != 1:
            print("Diff ops and WKS need to be multiplied accordingly. Not implemented for now")
            sys.exit()
        self.source_vertices *= self.__random_scale

        bb = igl.bounding_box(self.source_vertices.numpy())[0]
        diag = igl.bounding_box_diagonal(self.source_vertices.numpy())

        # self.source_mesh_centroid = torch.mean(self.source_vertices, axis=0)
        self.source_mesh_centroid =  (bb[0] + bb[-1])/2
        if self.random_centering:
            # centering augmentation
            self.source_mesh_centroid =  self.source_mesh_centroid + [(2*random.random() - 1)*diag*0.2, (2*random.random() - 1)*diag*0.2, (2*random.random() - 1)*diag*0.2]
        # self.source_mesh_centroid =  (bb[0] + bb[-1])/2 - np.array([-0.00033245, -0.2910367 ,  0.02100835])

        # Load input to NJF MLP
        # start = time.time()
        centroids = self.mesh_processor.get_centroids()
        centroid_points_and_normals = centroids.points_and_normals
        if self.__use_wks:
            wks = WKS_FACTOR * centroids.wks
            centroid_points_and_normals = numpy.hstack((centroid_points_and_normals, wks))
        self.centroids_and_normals = torch.from_numpy(
            centroid_points_and_normals).type(self.__ttype)
        if self.center_source:
            c = self.source_mesh_centroid
            self.centroids_and_normals[:, 0:3] -= c
            self.source_vertices -= c
            self.__source_global_translation_to_original = c
        self.poisson = self.mesh_processor.diff_ops.poisson_solver


        # Essentially here we load pointnet data and apply the same preprocessing
        for key in self.__extra_keys:
            data = self.mesh_processor.get_data(key)
            # if data is None:  # not found in mesh data so try loading from disk
            #     data = np.load(os.path.join(self.source_dir, key + ".npy"))
            data = torch.from_numpy(data)
            if key == 'samples':
                if self.center_source:
                    data -= self.get_mesh_centroid()
                scale = self.__random_scale
                data *= scale
            data = data.unsqueeze(0).type(self.__ttype)
            
            self.__loaded_data[key] = data
        # print("Ellapsed load source mesh ", time.time() - start)

    def load(self, source_v=None, source_f=None):
        # mesh_data = SourceMeshData.SourceMeshData.meshprocessor_from_file(self.source_dir)
        if source_v is not None and source_f is not None:
            self.mesh_processor = MeshProcessor.MeshProcessor.meshprocessor_from_array(source_v,source_f, self.source_dir, self.__ttype, cpuonly=self.cpuonly, load_wks_samples=self.__use_wks, load_wks_centroids=self.__use_wks)
        else:
            if os.path.isdir(self.source_dir):
                self.mesh_processor = MeshProcessor.MeshProcessor.meshprocessor_from_directory(self.source_dir, self.__ttype, cpuonly=self.cpuonly, load_wks_samples=self.__use_wks, load_wks_centroids=self.__use_wks)
            else:
                self.mesh_processor = MeshProcessor.MeshProcessor.meshprocessor_from_file(self.source_dir, self.__ttype, cpuonly=self.cpuonly, load_wks_samples=self.__use_wks, load_wks_centroids=self.__use_wks)
        self.__init_from_mesh_data()

    def get_point_dim(self):
        return self.centroids_and_normals.shape[1]

    def get_centroids_and_normals(self):
        return self.centroids_and_normals

    def get_mesh_centroid(self):
        return self.source_mesh_centroid

    def pin_memory(self):
        # self.poisson.pin_memory()
        # self.centroids_and_normals.pin_memory()
        # self.source_vertices.pin_memory()
        # for key in self.__loaded_data.keys():
        #     self.__loaded_data[key].pin_memory()
        return self


