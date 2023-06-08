import os

import numpy
import numpy as np
import torch
import igl
import MeshProcessor
WKS_DIM = MeshProcessor.WKS_DIM
WKS_FACTOR = 1000
import numpy as np
import sys
import random
import time
from utils import FourierFeatureTransform

class SourceMesh:
    '''
    datastructure for the source mesh to be mapped
    '''

    def __init__(self, source_ind, source_dir, extra_source_fields,
                 random_scale, ttype, use_wks=False, random_centering=False,
                cpuonly=False, init=False, fft=False, fft_dim=256, flatten=False,
                initjinput=False, debug=False, top_k_eig=50):
        self.__use_wks = use_wks
        self.source_ind = source_ind
        # NOTE: This is the CACHE DIRECTORY
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
        self.init = init
        self.initjinput = initjinput
        self.flatten = flatten
        self.debug = debug
        self.top_k_eig = top_k_eig

        self.fft = None
        if fft:
            # Compute input channels
            if self.__use_wks:
                n_input = 106
            else:
                n_input = 6

            # Initialize fourier features transform
            self.fft = FourierFeatureTransform(n_input, fft_dim)

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

    ### PRECOMPUTATION HAPPENS HERE ###
    def __init_from_mesh_data(self, new_init=False):
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

        self.source_mesh_centroid =  (bb[0] + bb[-1])/2
        if self.random_centering:
            # centering augmentation
            self.source_mesh_centroid =  self.source_mesh_centroid + [(2*random.random() - 1)*diag*0.2, (2*random.random() - 1)*diag*0.2, (2*random.random() - 1)*diag*0.2]

        # Load input to NJF MLP
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

        if self.fft is not None:
            self.centroids_and_normals = self.fft(self.centroids_and_normals)

        self.poisson = self.mesh_processor.diff_ops.poisson_solver

        # First check if initialization cached
        # Precompute Tutte if set
        if self.init == "tutte":
            if os.path.exists(os.path.join(self.source_dir, "tuttefuv.pt")) and \
                os.path.exists(os.path.join(self.source_dir, "tutteuv.pt")) and \
                os.path.exists(os.path.join(self.source_dir, "tuttej.pt")) and \
                os.path.exists(os.path.join(self.source_dir, "tuttetranslate.pt")):
                self.tuttefuv = torch.load(os.path.join(self.source_dir, "tuttefuv.pt"))
                self.tutteuv = torch.load(os.path.join(self.source_dir, "tutteuv.pt"))
                self.tuttej = torch.load(os.path.join(self.source_dir, "tuttej.pt"))
                self.tuttetranslate = torch.load(os.path.join(self.source_dir, "tuttetranslate.pt"))
            else:
                from utils import tutte_embedding, get_local_tris

                vertices = self.source_vertices
                device = vertices.device
                faces = self.get_source_triangles()
                fverts = vertices[faces].transpose(1,2)

                # TODO: GENERATE RANDOM CUTS IF SET

                self.tutteuv = torch.from_numpy(tutte_embedding(vertices.detach().cpu().numpy(), self.get_source_triangles(), fixclosed=True)).unsqueeze(0) # 1 x F x 2

                # Convert Tutte to 3-dim
                self.tutteuv = torch.cat([self.tutteuv, torch.zeros(self.tutteuv.shape[0], self.tutteuv.shape[1], 1)], dim=-1)

                # Get Jacobians
                self.tuttej = self.jacobians_from_vertices(self.tutteuv) #  F x 3 x 3

                # DEBUG: make sure we can get back the original UVs up to global translation
                pred_V = torch.einsum("abc,acd->abd", (self.tuttej[0,:,:2,:], fverts)).transpose(1,2)
                checktutte = self.tutteuv[0,faces,:2]
                diff = pred_V - checktutte
                diff -= torch.mean(diff, dim=1, keepdim=True) # Removes effect of per-triangle clobal translation
                torch.testing.assert_allclose(diff.float(), torch.zeros(diff.shape), rtol=1e-4, atol=1e-4)

                ## Save the global translations
                self.tuttetranslate = (checktutte - pred_V)[:,:,:2]
                self.tuttefuv = self.tutteuv[:,faces,:2] # B x F x 3 x 2

                # Cache everything
                torch.save(self.tuttefuv, os.path.join(self.source_dir, "tuttefuv.pt"))
                torch.save(self.tutteuv, os.path.join(self.source_dir, "tutteuv.pt"))
                torch.save(self.tuttej, os.path.join(self.source_dir, "tuttej.pt"))
                torch.save(self.tuttetranslate, os.path.join(self.source_dir, "tuttetranslate.pt"))

            ## Store in loaded data so it gets mapped to device
            # Remove extraneous dimension
            self.__loaded_data['tuttefuv'] = self.tuttefuv
            self.__loaded_data['tutteuv'] = self.tutteuv
            self.__loaded_data['tuttej'] = self.tuttej
            self.__loaded_data['tuttetranslate'] = self.tuttetranslate

            if self.initjinput:
                self.centroids_and_normals = torch.cat([self.centroids_and_normals, self.tuttej.reshape(len(self.centroids_and_normals), -1)], dim=1)

            # self.__loaded_data['localj'] = self.localj

            # Debugging: plot the initial embedding
            # import matplotlib.pyplot as plt
            # fig, axs = plt.subplots(figsize=(6, 4))
            # # plot ours
            # axs.triplot(self.tutteuv[0,:,0], self.tutteuv[0,:,1], self.get_source_triangles(), linewidth=0.5)
            # plt.axis('off')
            # plt.savefig(f"scratch/{self.source_ind}.png")
            # plt.close(fig)
            # plt.cla()
        elif self.init == "isometric":
            if os.path.exists(os.path.join(self.source_dir, "isofuv.pt")) and \
                os.path.exists(os.path.join(self.source_dir, "isoj.pt")) and \
                os.path.exists(os.path.join(self.source_dir, "isotranslate.pt")) and \
                not new_init:

                self.isofuv = torch.load(os.path.join(self.source_dir, "isofuv.pt"))
                self.isoj = torch.load(os.path.join(self.source_dir, "isoj.pt"))
                self.isotranslate = torch.load(os.path.join(self.source_dir, "isotranslate.pt"))
            else:
                from utils import get_local_tris

                vertices = self.source_vertices
                device = vertices.device
                faces = self.get_source_triangles()
                fverts = vertices[faces]

                # Random choice of local basis
                if new_init:
                    # Global rotation of initialization
                    if new_init == "global":
                        local_tris = get_local_tris(vertices, faces, basis=None) # F x 3 x 2
                        theta = np.random.uniform(low=0, high=2 * np.pi, size=1)
                        rotationmat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                        local_tris = (np.matmul(rotationmat.reshape(1, 2, 2), local_tris.transpose(2,1))).transpose(2,1) # F x 3 x 2

                    # Sample random basis per triangle
                    if new_init == "basis":
                        basistype = np.random.choice(6, size=len(faces))
                        local_tris = get_local_tris(vertices, faces, basis=basistype) # F x 3 x 2

                    # Randomly sample rotations
                    if new_init == "rot":
                        local_tris = get_local_tris(vertices, faces, basis=None) # F x 3 x 2
                        thetas = np.random.uniform(low=0, high=2 * np.pi, size=len(local_tris))
                        rotations = np.array([[[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]] for theta in thetas])
                        local_tris = (np.matmul(rotations, local_tris.transpose(2,1))).transpose(2,1) # F x 3 x 2

                else:
                    local_tris = get_local_tris(vertices, faces) # F x 3 x 2

                # TODO: TRY PROJECTING TO PRINCIPAL CURVATURE BASIS INSTEAD
                # TODO: random init -> 6 dfft local bases for each triangle, just sample from them (ab => [1,0], ba => [-1,0], ...)

                # Randomly sample displacements
                # NOTE: might need to alter the scales here
                self.isofuv = local_tris + np.random.uniform(size=(local_tris.shape[0], 1, local_tris.shape[2])) # F x 3 x 2

                # Unit testing: face areas should be same as in 3D
                from meshing.mesh import Mesh
                from meshing.analysis import computeFaceAreas

                mesh = Mesh(vertices.detach().cpu().numpy(), faces)
                computeFaceAreas(mesh)
                fareas3d = mesh.fareas
                fareas2d = 0.5 * np.abs(torch.linalg.det(torch.cat([torch.ones((len(self.isofuv), 1, 3)).float(), self.isofuv.transpose(2,1)], dim=1)).numpy())

                np.testing.assert_allclose(fareas3d, fareas2d, err_msg="Isometric embedding: all triangle areas should be same!")

                #### Get jacobians using gradient operator per triangle
                from igl import grad

                # Gradient operator (F*3 x V)
                # NOTE: need to compute this separately per triangle b/c of soup
                from scipy.sparse import bmat
                G = []
                for i in range(len(faces)):
                    G.append(grad(fverts[i].detach().cpu().numpy(), np.arange(3).reshape(1,3)))
                # Create massive sparse block diagonal matrix
                G = bmat([[None for _ in range(i)] + [G[i]] + [None for _ in range(i+1, len(G))] for i in range(len(G))])

                # Convert local tris to soup
                isosoup = self.isofuv.reshape(-1, 2).detach().cpu().numpy() # V x 2

                # Get Jacobians
                self.isoj = torch.from_numpy((G @ isosoup).reshape(local_tris.shape)).transpose(2,1)

                ## Debugging: make sure we can get back the original UVs up to global translation
                pred_V = torch.einsum("abc,acd->abd", (fverts, self.isoj.transpose(2,1)))
                diff = pred_V - self.isofuv
                diff -= torch.mean(diff, dim=1, keepdim=True) # Removes effect of per-triangle global translation
                torch.testing.assert_allclose(diff.float(), torch.zeros(diff.shape), rtol=1e-4, atol=1e-4)

                ## Save the global translations
                self.isotranslate = self.isofuv - pred_V

                # Cache everything
                torch.save(self.isofuv, os.path.join(self.source_dir, "isofuv.pt"))
                torch.save(self.isoj, os.path.join(self.source_dir, "isoj.pt"))
                torch.save(self.isotranslate, os.path.join(self.source_dir, "isotranslate.pt"))

            ## Store in loaded data so it gets mapped to device
            # NOTE: need to transpose isoj to interpret as 2x3
            self.__loaded_data['isofuv'] = self.isofuv
            self.__loaded_data['isoj'] = self.isoj
            self.__loaded_data['isotranslate'] = self.isotranslate

            if self.initjinput:
                self.centroids_and_normals = torch.cat([self.centroids_and_normals, self.isoj.reshape(len(self.centroids_and_normals), -1)], dim=1)

            # self.__loaded_data['localj'] = self.localj

            # Debugging: plot the initial embedding
            # import matplotlib.pyplot as plt
            # fig, axs = plt.subplots(figsize=(6, 4))
            # # plot ours
            # axs.triplot(self.tutteuv[0,:,0], self.tutteuv[0,:,1], self.get_source_triangles(), linewidth=0.5)
            # plt.axis('off')
            # plt.savefig(f"scratch/{self.source_ind}.png")
            # plt.close(fig)
            # plt.cla()

        # TODO: OBVIOUSLY THIS WONT WORK WITH LEARNING -- NEED INPUT TO BE FUNCTION OF THE SAMPLED INITIALIZATION
        # Initialize random flat vector if set
        # if self.flatten == "random":
        #     self.flat_vector = torch.rand(1, len(self.mesh_processor.faces) * 9) * 100

        # if self.flatten == "xyz":
        #     # Initialize with all triangle centroid positions
        #     self.flat_vector = self.centroids_and_normals[:,:3].reshape(1, -1)

        # Use initialization jacobians as input
        if self.flatten == "input":
            self.flat_vector = torch.cat([self.isoj, torch.zeros((self.isoj.shape[0], 1, 3))], dim=1).reshape(1, -1)
            # nchannels = self.centroids_and_normals.shape[1]
            # gsize = int(np.ceil(nchannels/9))
            # newchannels = []
            # for i in range(9):
            #     newchannels.append(torch.sum(self.centroids_and_normals[:,i*gsize:(i+1)*gsize], dim=1))
            # self.flat_vector = torch.stack(newchannels, dim=1).reshape(1, -1)

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

    def load(self, source_v=None, source_f=None, new_init=False):
        if source_v is not None and source_f is not None:
            self.mesh_processor = MeshProcessor.MeshProcessor.meshprocessor_from_array(source_v,source_f, self.source_dir, self.__ttype,
                                                                                       cpuonly=self.cpuonly, load_wks_samples=self.__use_wks,
                                                                                       load_wks_centroids=self.__use_wks,
                                                                                       top_k_eig=self.top_k_eig)
        else:
            if os.path.isdir(self.source_dir):
                self.mesh_processor = MeshProcessor.MeshProcessor.meshprocessor_from_directory(self.source_dir, self.__ttype,
                                                                                               cpuonly=self.cpuonly,
                                                                                               load_wks_samples=self.__use_wks,
                                                                                               load_wks_centroids=self.__use_wks,
                                                                                               top_k_eig=self.top_k_eig)
            else:
                self.mesh_processor = MeshProcessor.MeshProcessor.meshprocessor_from_file(self.source_dir, self.__ttype,
                                                                                          cpuonly=self.cpuonly,
                                                                                          load_wks_samples=self.__use_wks,
                                                                                          load_wks_centroids=self.__use_wks,
                                                                                          top_k_eig=self.top_k_eig)
        self.__init_from_mesh_data(new_init)

    def get_point_dim(self):
        if self.flatten:
            return self.flat_vector.shape[1]
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