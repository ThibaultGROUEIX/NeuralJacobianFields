#defines the network and the train loop
import warnings

import igl
import matplotlib
import numpy as np
import numpy.random
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import DeformationEncoder
from losses import UVLoss, symmetricdirichlet
from results_saving_scripts import save_mesh_with_uv
from DeformationDataset import DeformationDataset
from torch.utils.data import DataLoader
from torch import nn
import torch
import PerCentroidBatchMaker
import MeshProcessor
from utils import dclamp, ZeroNanGrad, order_edges_and_cut

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment
from signal import SIGUSR1

from meshing.mesh import Mesh
from meshing.analysis import computeFacetoEdges

USE_CUPY = True
has_gpu = "cpu"
if USE_CUPY and torch.cuda.is_available():
    import cupy
    has_gpu="gpu"

import math
from pytorch_lightning.loggers import TensorBoardLogger
from results_saving_scripts.plot_uv import plot_uv, export_views
from pathlib import Path
from results_saving_scripts import paper_stats
import json

FREQUENCY = 10 # frequency of logguing - every FREQUENCY iteration step
UNIT_TEST_POISSON_SOLVE = True

class MyNet(pl.LightningModule):
    '''
    the network
    '''
    def __init__(self, encoder, code_dim, args, point_dim=6, verbose=False, face_dim=0):
        print("********** Some Network info...")
        print(f"********** code dim: {code_dim}")
        print(f"********** centroid dim: {point_dim}")
        super().__init__()
        self.args = args
        self.arch = self.args.arch

        # HACK: lossgt
        if not hasattr(args, "lossgt"):
            args.lossgt = False

        self.lossfcn = UVLoss(args, self.device)

        # NOTE: code dim refers to the pointnet encoding. Point_dim is centroid position (also potentially fourier features)
        layer_normalization = self.get_layer_normalization_type()
        if self.args.softpoisson:
            assert face_dim > 0, f"face_dim must be > 0 for soft poisson. face_dim: {face_dim}."

            # TODO: INPUT DIM IS DIFFERENT FOR DIFFUSIONNET (NEED VERTEX FEATURES)
            # NOTE: Dimension of output depends on max of # edge weights being predicted across dataset
            output_dim = 9 + face_dim
            input_dim = point_dim + code_dim

            ## DiffusionNet vertex encoder
            if self.arch == "diffusionnet":
                from diffusionnet import DiffusionNet
                vertexdim = self.args.vertexdim
                self.vertexencoder = DiffusionNet(C_in=input_dim, C_out=self.args.vertexdim, C_width=128, N_block=4, outputs_at='vertices',
                                                  with_gradient_features=True, with_gradient_rotations=True)
                face_decoder_dim = 9
                edge_decoder_dim = 1
                self.edge_decoder = nn.Sequential(nn.Linear(self.args.vertexdim, 128),
                                            nn.ReLU(),
                                            nn.Linear(128, 128),
                                            nn.ReLU(),
                                            nn.Linear(128, 128),
                                            nn.ReLU(),
                                            nn.Linear(128, edge_decoder_dim),
                                            )

                # Initialize edge weights to 0
                self.edge_decoder[-1].bias.data.zero_()
                self.edge_decoder[-1].weight.data.zero_()

                self.face_decoder = nn.Sequential(nn.Linear(self.args.vertexdim, 128),
                                            nn.GroupNorm(num_groups=4, num_channels=128), # , eps=0.0001 I have considered increasing this value in case we have channels from pointnet with the same values.
                                            nn.ReLU(),
                                            nn.Linear(128, 128),
                                            nn.GroupNorm(num_groups=4, num_channels=128), # , eps=0.0001 I have considered increasing this value in case we have channels from pointnet with the same values.
                                            nn.ReLU(),
                                            nn.Linear(128, 128),
                                            nn.GroupNorm(num_groups=4, num_channels=128),
                                            nn.ReLU(),
                                            nn.Linear(128, face_decoder_dim),
                                            )

                self.__IDENTITY_INIT = self.args.identity
                if self.__IDENTITY_INIT:
                    self.face_decoder[-1].bias.data.zero_()
                    self.face_decoder[-1].weight.data.zero_()

            ## TODO
            elif self.arch == "attention":
                pass

            elif self.arch == "mlp":
                self.face_decoder = nn.Sequential(nn.Linear(input_dim, 128),
                                                    nn.GroupNorm(num_groups=4, num_channels=128), # , eps=0.0001 I have considered increasing this value in case we have channels from pointnet with the same values.
                                                    nn.ReLU(),
                                                    nn.Linear(128, 128),
                                                    nn.GroupNorm(num_groups=4, num_channels=128), # , eps=0.0001 I have considered increasing this value in case we have channels from pointnet with the same values.
                                                    nn.ReLU(),
                                                    nn.Linear(128, 128),
                                                    nn.GroupNorm(num_groups=4, num_channels=128),
                                                    nn.ReLU(),
                                                    nn.Linear(128, 128),
                                                    nn.GroupNorm(num_groups=4, num_channels=128),
                                                    nn.ReLU(),
                                                    nn.Linear(128, output_dim))
                # Initialize the last layer sparsely
                torch.nn.init.sparse_(self.face_decoder[-1].weight.data[9:], 0.2)

                self.__IDENTITY_INIT = self.args.identity
                if self.__IDENTITY_INIT:
                    self.face_decoder[-1].bias.data[:].zero_()
                    self.face_decoder[-1].weight.data[:9].zero_()

            else:
                raise NotImplementedError(f"Unknown architecture: {self.arch}.")

        elif hasattr(self.args, "dense") and self.args.dense:
            print("==== We are predicting FLAT vectors! ==== ")
            if self.args.dense == "xyz":
                channels = (point_dim + code_dim) * 3
            else:
                channels = point_dim + code_dim
            self.face_decoder = nn.Sequential(nn.Linear(point_dim + code_dim, channels),
                                                    nn.LayerNorm(normalized_shape=channels),
                                                    nn.ReLU(),
                                                    nn.Linear(channels, channels),
                                                    nn.LayerNorm(normalized_shape=channels),
                                                    nn.ReLU(),
                                                    nn.Linear(channels, channels),
                                                    nn.LayerNorm(normalized_shape=channels),
                                                    nn.ReLU(),
                                                    nn.Linear(channels, channels),
                                                    nn.LayerNorm(normalized_shape=channels),
                                                    nn.ReLU(),
                                                    nn.Linear(channels, channels),
                                                    nn.LayerNorm(normalized_shape=channels),
                                                    nn.ReLU(),
                                                    nn.Linear(channels, channels),
                                                    )
        elif layer_normalization == "IDENTITY":
            # print("Using IDENTITY (no normalization) in face_decoder!")
            self.face_decoder = nn.Sequential(nn.Linear(point_dim + code_dim, 128),
                                                  nn.Identity(),
                                                  nn.ReLU(),
                                                  nn.Linear(128, 128),
                                                  nn.Identity(),
                                                  nn.ReLU(),
                                                  nn.Linear(128, 128),
                                                  nn.Identity(),
                                                  nn.ReLU(),
                                                  nn.Linear(128, 128),
                                                  nn.Identity(),
                                                  nn.ReLU(),
                                                  nn.Linear(128, 9))
        elif layer_normalization == "BATCHNORM":
            # print("Using BATCHNORM in face_decoder!")
            self.face_decoder = nn.Sequential(nn.Linear(point_dim + code_dim, 128),
                                                  nn.BatchNorm1d(128),
                                                  nn.ReLU(),
                                                  nn.Linear(128, 128),
                                                  nn.BatchNorm1d(128),
                                                  nn.ReLU(),
                                                  nn.Linear(128, 128),
                                                  nn.BatchNorm1d(128),
                                                  nn.ReLU(),
                                                  nn.Linear(128, 128),
                                                  nn.BatchNorm1d(128),
                                                  nn.ReLU(),
                                                  nn.Linear(128, 9))
        elif layer_normalization == "GROUPNORM_CONV":
            # print("Using GROUPNORM2 in face_decoder!")
            self.face_decoder = nn.Sequential(nn.Conv1d(point_dim + code_dim, 128, 1),
                                                  nn.GroupNorm(num_groups=4, num_channels=128),
                                                  nn.ReLU(),
                                                  nn.Conv1d(128, 128, 1),
                                                  nn.GroupNorm(num_groups=4, num_channels=128),
                                                  nn.ReLU(),
                                                  nn.Conv1d(128, 128, 1),
                                                  nn.GroupNorm(num_groups=4, num_channels=128),
                                                  nn.ReLU(),
                                                  nn.Conv1d(128, 128, 1),
                                                  nn.GroupNorm(num_groups=4, num_channels=128),
                                                  nn.ReLU(),
                                                  nn.Conv1d(128, 9, 1))
        elif layer_normalization == "GROUPNORM":
            # print("Using GROUPNORM in face_decoder!")
            self.face_decoder = nn.Sequential(nn.Linear(point_dim + code_dim, 128),
                                                  nn.GroupNorm(num_groups=4, num_channels=128), # , eps=0.0001 I have considered increasing this value in case we have channels from pointnet with the same values.
                                                  nn.ReLU(),
                                                  nn.Linear(128, 128),
                                                  nn.GroupNorm(num_groups=4, num_channels=128), # , eps=0.0001 I have considered increasing this value in case we have channels from pointnet with the same values.
                                                  nn.ReLU(),
                                                  nn.Linear(128, 128),
                                                  nn.GroupNorm(num_groups=4, num_channels=128),
                                                  nn.ReLU(),
                                                  nn.Linear(128, 128),
                                                  nn.GroupNorm(num_groups=4, num_channels=128),
                                                  nn.ReLU(),
                                                  nn.Linear(128, 9))
        elif layer_normalization == "LAYERNORM":
            # print("Using LAYERNORM in face_decoder!")
            self.face_decoder = nn.Sequential(nn.Linear(point_dim + code_dim, 128),
                                                  nn.LayerNorm(normalized_shape=128),
                                                  nn.ReLU(),
                                                  nn.Linear(128, 128),
                                                  nn.LayerNorm(normalized_shape=128),
                                                  nn.ReLU(),
                                                  nn.Linear(128, 128),
                                                  nn.LayerNorm(normalized_shape=128),
                                                  nn.ReLU(),
                                                  nn.Linear(128, 128),
                                                  nn.LayerNorm(normalized_shape=128),
                                                  nn.ReLU(),
                                                  nn.Linear(128, 9))
        else:
            raise Exception("unknown normalization method")

        self.__global_trans = self.args.globaltrans
        if self.__global_trans:
            self.global_decoder = nn.Sequential(nn.Linear(code_dim, 128),
                                                nn.ReLU(),
                                                nn.Linear(128, 64),
                                                nn.ReLU(),
                                                nn.Linear(64, 64),
                                                nn.ReLU(),
                                                nn.Linear(64, 64),
                                                nn.ReLU(),
                                                nn.Linear(64, 9))

        self.encoder = encoder
        self.point_dim = point_dim
        self.code_dim = code_dim
        self.verbose = verbose
        self.mse = nn.MSELoss()
        self.save_hyperparameters()
        self.log_validate = True
        self.lr = args.lr
        self.val_step_iter = 0
        self.__test_stats = paper_stats.PaperStats()

        # For iterative reweighting
        self.stitchweights = {}

    ##################
    # inference code below
    ##################
    def forward(self, x, source=None):
        '''
		The MLP applied to a (batch) of global code concatenated to a centroid (z|c)
		:param x: B x (|z|+|c|) batch of (z|c) vectors
		:return: B x 9 batch of 9 values that are the 3x3 matrix predictions for each input vector
		'''
        if self.arch == 'mlp':
            return self.face_decoder(x)
        elif self.arch == 'diffusionnet':
            return self.vertexencoder(x, source.get_loaded_data('mass'), L=source.get_loaded_data('L'),
                                      evals=source.get_loaded_data('evals'), evecs=source.get_loaded_data('evecs'),
                                      gradX=source.get_loaded_data('gradX'), gradY=source.get_loaded_data('gradY'),
                                      faces=source.get_loaded_data('faces'))

    def predict_jacobians(self, source, target):
        '''
		given a batch class, predict jacobians
		:param single_source_batch: batch object
		:return: BxTx3x3 a tensor of 3x3 jacobians, per T tris, per B targets in batch
		'''
        # extract the encoding of the source and target
        if self.args.noencoder:
            codes = None
        else:
            codes = self.extract_code(source, target)

        # get the network predictions, a BxTx3x3 tensor of 3x3 jacobians, per T tri, per B target in batch
        return self.predict_jacobians_from_codes(codes, source)

    def predict_jacobians_from_codes(self, codes, source):
        '''
		predict jacobians w.r.t give global codes and the batch
		:param codes: codes for each source/target in batch
		:param single_source_batch: the batch
		:return:BxTx3x3 a tensor of 3x3 jacobians, per T tris, per B targets in batch
		'''
        if self.args.dense:
            stacked = source.flat_vector
        elif codes is None:
            stacked = source.get_input_features()
        else:
            # take all encodings z_i of targets, and all centroids c_j of triangles, and create a cartesian product of the two as a 2D tensor so each sample in it is a vector with rows (z_i|c_j)
            net_input = PerCentroidBatchMaker.PerCentroidBatchMaker(codes, source.get_input_features(), args=self.args)
            stacked = net_input.to_stacked()
            if self.args.layer_normalization != "GROUPNORM2":
                stacked = net_input.prep_for_linear_layer(stacked)
            else:
                stacked = net_input.prep_for_conv1d(stacked)

        # feed the 2D tensor through the network, and get a 3x3 matrix for each (z_i|c_j)
        res = self.forward(stacked, source)

        ##### SOFT POISSON #####
        if self.arch == 'diffusionnet':
            # Construct edge and face codes from vertex codes
            vertexcodes = res
            faces = source.get_source_triangles()
            edge_vpairs_nobound = source.ogedge_vpairs_nobound
            facecodes = torch.mean(vertexcodes[faces,:], dim=1)
            edgecodes = torch.mean(vertexcodes[edge_vpairs_nobound,:], dim=1)

            res = self.face_decoder(facecodes)
            edgevals = self.edge_decoder(edgecodes).squeeze()

            # Compute edge weights
            if self.args.spweight == "sigmoid":
                weights = -torch.sigmoid(edgevals + source.initweights.to(edgevals.device))
            # Softmax the weights
            elif self.args.spweight == "softmax":
                weights = torch.softmax(edgevals + source.initweights.to(edgevals.device), dim=0)
                weights *= len(weights) # Normalize by number of weights
                weights = -weights

        # TODO: take SA-computed face latents => cosine similarity => dclamp(-2, 2) => add 1 div 2 (0 - 1 scaling)
        #                                     => face decoder MLP => jacobians
        elif self.arch == 'attention':
            pass

        elif self.arch == 'mlp':
            # NOTE: soft poisson weights will be dot product of "face latents" predicted per triangle for corresponding faces
            facelatents = res[:, 9:]
            facepairs = source.facepairs_nobound

            if self.args.spweight == "sigmoid":
                facedot = torch.sum(facelatents[facepairs[:,0]] * facelatents[facepairs[:,1]], dim=1)
                facesim = torch.sigmoid(facedot + source.initweights.to(facedot.device))
            elif self.args.spweight == "seamless":
                facedot = torch.sum(facelatents[facepairs[:,0]] * facelatents[facepairs[:,1]], dim=1)
                facesim = (facedot + source.initweights.to(facedot.device))**2/((facedot + source.initweights.to(facedot.device))**2 + self.args.seamlessdelta)
            elif self.args.spweight == "cosine":
                facesim = dclamp(torch.nn.functional.cosine_similarity(facelatents[facepairs[:,0]], facelatents[facepairs[:,1]]) \
                                + source.initweights.to(facelatents.device), 1e-7, 1)
            elif self.args.spweight == "softmax":
                facedot = torch.sum(facelatents[facepairs[:,0]] * facelatents[facepairs[:,1]], dim=1)
                facesim = torch.softmax(facedot + source.initweights.to(facedot.device))
                facesim *= len(facesim)
            else:
                raise Exception(f"Unknown soft poisson weight type: {self.args.spweight}.")
            weights = -facesim

            if self.args.softpoisson == "edges":
                assert len(weights) == len(source.edge_vpairs), f"weights len: {len(weights)}, edge vpairs len: {len(source.edge_vpairs)}"
            else:
                assert len(weights) == len(source.valid_pairs), f"weights len: {len(weights)}, valid pairs len: {len(source.valid_pairs)}"

            res = res[:, :9]

            if self.args.optweight:
                weights = -dclamp(self.trainer.optimizers[0].param_groups[1]['params'][0] + source.initweights.to(facelatents.device), 1e-7, 1)


        # No global codes
        if self.args.dense or codes is None:
            ret = res.reshape(1, source.mesh_processor.faces.shape[0], 3, 3)

            if self.__IDENTITY_INIT:
                for i in range(0, 3):
                    ret[:, :, i, i] += 1

            return ret, weights

        ## TODO: REFACTOR BELOW IF GLOBAL CODES ##
        # because of stacking the result is a 9-entry vec for each (z_i|c_j), now let's turn it to a batch x tris x 9 tensor
        pred_J = net_input.back_to_non_stacked(res)
        # and now reshape 9 to 3x3
        ret = pred_J.reshape(pred_J.shape[0], pred_J.shape[1], 3, 3)
        # if we apply a global transformation
        if self.__global_trans:
            glob = self.global_decoder(codes)
            glob = glob.reshape(glob.shape[0], 3, 3).unsqueeze(1)
            ret = torch.matmul(glob, ret)
        # if we chose to have the identity as the result when the prediction is 0,
        if self.__IDENTITY_INIT:
            for i in range(0, 3):
                ret[:, :, i, i] += 1

        return ret

    def extract_code(self, source, target):
        '''
		given a batch, extract the global code w.r.t the source and targets, using the set encoders
		:param batch: the batch object
		:return: Bx|z| batch of codes z
		'''
        return self.encoder.encode_deformation(source, target)

    def hardpoisson(self, source, jacobians, vertices, faces, cut_values, vertex_pairs, threshold=0.5):
        """ cut_values: E*2 x 1 array of values to evaluate against threshold to determine cuts"""
        assert len(vertex_pairs)  == len(cut_values), f"vertex pairs len: {len(vertex_pairs)}, cut_values len: {len(cut_values)}"
        # Compute new topology based off the weights (under threshold => have edge/no cut)
        seamlengths = []
        mesh = Mesh(vertices.detach().cpu().numpy(), faces.detach().cpu().numpy())

        ## Cutting strategy
        # Map weights => edge cuts
        # Construct edge cut list by connecting cut paths
        # Loop through edges and cut with prio:
        # (1) edge on boundary
        # (2) two edges which share a vertex
        # (3) worst case: have to cut internal edge while choosing random adjacent edge as e2_i
        from meshing.edit import EdgeCut

        # NOTE: If cut values are vertex separation, then either vertex split can result in edge being cut
        # Weights correspond to vertex pairs which refer to edges (edgeidxs)
        cut_e = np.unique(source.edges[np.where(cut_values > threshold)[0]]) # Dupes b/c vertex pair:edges is 2:1

        # Filter out bad edges (edges on boundary)
        bd_edges = []
        for candidate_e in cut_e:
            if mesh.topology.edges[candidate_e].onBoundary():
                bd_edges.append(candidate_e)
        if len(bd_edges) > 0:
            print(f"Hard Poisson: Found {len(bd_edges)} boundary edges in cut edges. Deleting ... ")
            cut_e = np.delete(cut_e, np.where(np.isin(cut_e, bd_edges)))

        ### Construct edge cut list (recursive tree search) & make cuts
        edge_list, vertex_cutlist, seamlengths = order_edges_and_cut(mesh, cut_e)

        # List of cutlists (edge indices)

        # Re-sort so that cutlists which start at boundary are first
        # Make sure that subsequent cutlists are prio if they start on boundaries
        # NOTE: OG cutlists will be edge indices bc the indexing doesn't change after each cut -- BUT the vertex indices will change!

        # Store indices of vertices which correspond to cut edges
        # TODO: Cut edges indexing is incorrect! Need to update for subsequent cuts.
        hp_cutedges = []
        while len(cut_e) > 0:
            eidx = None
            vidx = None
            # (1) One vertex on boundary
            for candidate_e in cut_e:
                if np.any([v.onBoundary() for v in mesh.topology.edges[candidate_e].two_vertices()]):
                    eidx = candidate_e
                    edge = mesh.topology.edges[candidate_e]
                    vidx = edge.halfedge.vertex.index if edge.halfedge.vertex.onBoundary() else edge.halfedge.twin.vertex.index
                    break
            if eidx is not None and vidx is not None:
                cute1 = [v.index for v in mesh.topology.edges[eidx].two_vertices()]
                cute2 = [cute1[0], len(mesh.topology.vertices)] if vidx != cute1[0] else [cute1[1], len(mesh.topology.vertices)]
                hp_cutedges.append(cute1)
                hp_cutedges.append(cute2)

                seamlengths.append(mesh.length(mesh.topology.edges[eidx]))
                EdgeCut(mesh, eidx, vidx, cutbdry=True, checktopo=False).apply()
                cut_e = np.delete(cut_e, np.where(cut_e == eidx))
                continue

            # (2) Two edges adjacent to each other
            eidx = None
            eidx2 = None
            for candidate_e in cut_e:
                for candidate_e2 in mesh.topology.edges[candidate_e].halfedge.vertex.adjacentEdges():
                    if candidate_e2.index in cut_e and candidate_e2.index != candidate_e:
                        eidx = candidate_e
                        eidx2 = candidate_e2.index
                        break

                if eidx is not None and eidx2 is not None:
                    break

                for candidate_e2 in mesh.topology.edges[candidate_e].halfedge.twin.vertex.adjacentEdges():
                    if candidate_e2.index in cut_e and candidate_e2.index != candidate_e:
                        eidx = candidate_e
                        eidx2 = candidate_e2.index
                        break

                if eidx is not None and eidx2 is not None:
                    break

            if eidx is not None and eidx2 is not None:
                cute1 = [v.index for v in mesh.topology.edges[eidx].two_vertices()]
                cute2 = [cute1[0], len(mesh.topology.vertices)] if vidx != cute1[0] else [cute1[1], len(mesh.topology.vertices)]
                hp_cutedges.append(cute1)
                hp_cutedges.append(cute2)

                cute1 = [v.index for v in mesh.topology.edges[eidx2].two_vertices()]
                cute2 = [cute1[0], len(mesh.topology.vertices)] if vidx != cute1[0] else [cute1[1], len(mesh.topology.vertices)]
                hp_cutedges.append(cute1)
                hp_cutedges.append(cute2)

                seamlengths.append(mesh.length(mesh.topology.edges[eidx]))
                seamlengths.append(mesh.length(mesh.topology.edges[eidx2]))

                vidx = list(set(mesh.topology.edges[eidx].two_vertices()).intersection(mesh.topology.edges[eidx2].two_vertices()))[0].index
                EdgeCut(mesh, eidx, vidx, cutbdry=True, e2_i=eidx2, checktopo=False).apply()
                cut_e = np.delete(cut_e, np.where(np.isin(cut_e, [eidx, eidx2])))
                continue

            # (3) Worst case: choose random adjacent second edge that's not on boundary and not going to create a disconnected component
            # TODO: CHECK FOR POSSIBLE DISCONNECTION
            eidx = cut_e[0]
            for candidate_e2 in mesh.topology.edges[eidx].halfedge.vertex.adjacentEdges():
                if candidate_e2.index != eidx and not candidate_e2.onBoundary():
                    eidx2 = candidate_e2.index
                    break

            if eidx is not None or eidx2 is not None:
                for candidate_e2 in mesh.topology.edges[eidx].halfedge.twin.vertex.adjacentEdges():
                    if candidate_e2.index != eidx and not candidate_e2.onBoundary():
                        eidx2 = candidate_e2.index
                        break

            if eidx is None or eidx2 is None:
                raise ValueError(f"Hard Poisson: no valid cuts available for edge cut list {cut_e}.")

            cute1 = [v.index for v in mesh.topology.edges[eidx].two_vertices()]
            cute2 = [cute1[0], len(mesh.topology.vertices)] if vidx != cute1[0] else [cute1[1], len(mesh.topology.vertices)]
            hp_cutedges.append(cute1)
            hp_cutedges.append(cute2)

            cute1 = [v.index for v in mesh.topology.edges[eidx2].two_vertices()]
            cute2 = [cute1[0], len(mesh.topology.vertices)] if vidx != cute1[0] else [cute1[1], len(mesh.topology.vertices)]
            hp_cutedges.append(cute1)
            hp_cutedges.append(cute2)

            seamlengths.append(mesh.length(mesh.topology.edges[eidx]))
            seamlengths.append(mesh.length(mesh.topology.edges[eidx2]))

            vidx = list(set(mesh.topology.edges[eidx].two_vertices()).intersection(mesh.topology.edges[eidx2].two_vertices()))[0].index

            EdgeCut(mesh, eidx, vidx, cutbdry=True, e2_i=eidx2, checktopo=False).apply()
            cut_e = np.delete(cut_e, np.where(np.isin(cut_e, [eidx, eidx2])))

        seamlengths = np.array(seamlengths)
        cutvs, cutfs, cutes = mesh.export_soup()
        meshprocessor = MeshProcessor.MeshProcessor.meshprocessor_from_array(cutvs, cutfs, source.source_dir, source._SourceMesh__ttype,
                                                                             cpuonly=source.cpuonly, load_wks_samples=source._SourceMesh__use_wks,
                                                                             load_wks_centroids=source._SourceMesh__use_wks,
                                                                             sparse=self.args.sparsepoisson)
        meshprocessor.prepare_temporary_differential_operators(source._SourceMesh__ttype)
        poissonsolver = meshprocessor.diff_ops.poisson_solver
        poissonsolver.to(self.device)

        harduv = poissonsolver.solve_poisson(jacobians).squeeze()

        return harduv, cutvs, cutfs, seamlengths, hp_cutedges

    #######################################
    # Pytorch Lightning Boilerplate code (training, logging, etc.)
    #######################################
    def training_step(self, source_batches, batch_id):
        # start = time.time()
        # torch.cuda.synchronize()

        # if self.args.debug:
        #     import pdb
        #     pdb.set_trace()

        batch_parts = self.my_step(source_batches, batch_id)

        # Log scheduled weights
        if self.args.stitchschedule:
            self.log('edgecut_weight', self.args.edgecut_weight, logger=True, batch_size=1)
        if self.args.sparse_schedule:
            self.log('sparsecuts_weight', self.args.sparsecuts_weight, logger=True, batch_size=1)

        loss = batch_parts["loss"]
        lossrecord = batch_parts["lossdict"]

        # if self.args.debug:
        #     import pdb
        #     pdb.set_trace()

        self.log("train_loss", loss, logger=True, prog_bar=True, batch_size=1, on_epoch=True, on_step=False)

        # Log losses
        for key, val in lossrecord[0].items():
            if "loss" in key:
                self.log(key, np.mean(val), logger=True, prog_bar=False, batch_size=1, on_epoch=True, on_step=False)

        if self.args.mem:
            # Check memory consumption
            # Get GPU memory usage
            t = torch.cuda.get_device_properties(0).total_memory
            r = torch.cuda.memory_reserved(0)
            a = torch.cuda.memory_allocated(0)
            m = torch.cuda.max_memory_allocated(0)
            f = r-a  # free inside reserved
            print(f"{a/1024**3:0.3f} GB allocated. \nGPU max memory alloc: {m/1024**3:0.3f} GB. \nGPU total memory: {t/1024**3:0.3f} GB.")

            # Get CPU RAM usage too
            import psutil
            print(f'RAM memory % used: {psutil.virtual_memory()[2]}')

        return loss

    def on_train_epoch_end(self):
        self.log("epoch", self.current_epoch)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=False, logger=True, batch_size=1)

    def test_step(self, batch, batch_idx):
        return self.my_step(batch, batch_idx)

    def get_gt_map(self, source, target, softpoisson=False):
        GT_V = target.get_vertices()

        # ground truth jacobians, restricted as well
        GT_J = source.jacobians_from_vertices(GT_V)
        GT_J_restricted = source.restrict_jacobians(GT_J)
        return GT_V, GT_J, GT_J_restricted

    def predict_map(self, source, target, initj=None):
        if self.args.softpoisson or self.args.optweight:
            pred_J, weights = self.predict_jacobians(source, target)
        else:
            pred_J = self.predict_jacobians(source, target)

        # Need initialization J to have batch dimension
        if initj is not None:
            if len(initj.shape) == 3:
                initj = initj.unsqueeze(0)

            pred_J = torch.cat([torch.einsum("abcd,abde->abce", (pred_J[:,:,:2,:2], initj[:,:,:2,:])),
                                torch.zeros(pred_J.shape[0], pred_J.shape[1], 1, pred_J.shape[3], device=self.device)],
                               dim=2) # B x F x 3 x 3

        if self.args.align_2D:  # if the target is in 2D the last column of J is 0
            pred_J[:, :,2,:] = 0 # NOTE: predJ shape is B x F x 3 x 3 (where first 3-size is interpreted as axes)

        # Update the soft poisson if set
        updatedlap = False
        if self.args.softpoisson or self.args.optweight:
            # Reorganize predicted weights to match soft poisson indexing
            sp_idxs = source.edge_vpairs.reshape(-1, 2) # E * 2 x 2 (last dimension compares corresponding vertices (two pairs per edge))
            sp_idxs = (torch.cat([sp_idxs[:,0],sp_idxs[:,1]]), torch.cat([sp_idxs[:,1],sp_idxs[:,0]]))

            sp_weights = weights.repeat_interleave(2)
            sp_weights = torch.cat([sp_weights, sp_weights])

            source.poisson.lap.update_vals(sp_weights, sp_idxs, source.poisson.lap_pinned,
                                            source.poisson.lap_pinned_rows, source.poisson.lap_pinned_cols)

            # if self.args.softpoisson == "edges":
            #     source.poisson.lap.update_vals(weights, source.valid_edge_pairs, source.poisson.lap_pinned,
            #                                 source.poisson.lap_pinned_rows, source.poisson.lap_pinned_cols)
            # elif self.args.softpoisson == "valid":
            #     source.poisson.lap.update_vals(weights, source.valid_pairs, source.poisson.lap_pinned,
            #                                 source.poisson.lap_pinned_rows, source.poisson.lap_pinned_cols)

            updatedlap = True

        pred_V = source.vertices_from_jacobians(pred_J, updatedlap = updatedlap)

        # Get back jacobians from predicted vertices
        # TODO: For now we use pred_J for the predicted jacobians, but we could also use the restriction.
        # My intuition is that using pred_J while spoofing restriction (by dropping the 3rd row) is better for NN since basis
        # restriction can result in pretty arbitrary transformation across neighboring triangles but TBD.

        pred_J_poiss = source.poisson.jacobians_from_vertices(pred_V)
        pred_J_restricted_poiss = source.restrict_jacobians(pred_J)

        if self.args.softpoisson or self.args.optweight:
            return pred_V, pred_J, pred_J_poiss, pred_J_restricted_poiss, -weights

        return pred_V, pred_J, pred_J_poiss, pred_J_restricted_poiss

    def check_map(self, source, target, GT_J, GT_V):
        pred_V = source.vertices_from_jacobians(GT_J)
        return torch.max(torch.absolute(pred_V - GT_V))


    def validation_step(self, batch, batch_idx):
        batch_parts = self.my_step(batch, batch_idx, validation=True)
        val_loss = batch_parts['loss'].item()
        self.log('val_loss', val_loss, logger=True, prog_bar=True, batch_size=1, on_epoch=True, on_step=False)

        self.val_step_iter += 1

        if torch.rand(1).item() > self.args.valrenderratio:
            return val_loss

        ### Visualizations

        # Log path
        source, target = batch
        sourcename = os.path.basename(source.source_dir)
        source_path = os.path.join(self.logger.save_dir, "renders", sourcename)
        save_path = os.path.join(self.logger.save_dir, "renders", sourcename, "frames")

        # Loss dict
        lossdict = batch_parts['lossdict']
        keepidxs = source.keepidxs

        # Construct mesh
        mesh = Mesh(batch_parts["source_V"].detach().cpu().numpy(), batch_parts["ogT"])

        # Log losses
        for key, val in lossdict[0].items():
            if "loss" in key:
                self.log(f"val_{key}", np.mean(val), logger=True, prog_bar=False, batch_size=1, on_epoch=True, on_step=False)

        if not os.path.exists(save_path):
            Path(save_path).mkdir(exist_ok=True, parents=True)

        # Save latest predictions
        np.save(os.path.join(self.logger.save_dir, f"latest_preduv.npy"), batch_parts['pred_V'].squeeze().detach().cpu().numpy())
        np.save(os.path.join(self.logger.save_dir, f"latest_predw.npy"), batch_parts['weights'])

        # if self.args.opttrans:
        #     np.save(os.path.join(self.logger.save_dir, f"latest_preduv.npy"), batch_parts['pred_V_opttrans'].squeeze().detach().cpu().numpy())

        np.save(os.path.join(self.logger.save_dir, f"latest_predt.npy"), batch_parts['T'])
        np.save(os.path.join(self.logger.save_dir, f"latest_predj.npy"), batch_parts["pred_J"].squeeze().detach().cpu().numpy())

        # if self.args.no_poisson:
        #     np.save(os.path.join(source_path, f"latest_poissonuv.npy"), batch_parts['poissonUV'].squeeze().detach().cpu().numpy())
        #     np.save(os.path.join(source_path, f"latest_poissont.npy"), batch_parts['ogT']) # NOTE: poisson T uses original triangles!

        val_loss = batch_parts['loss'].item()
        if self.args.xp_type == "uv":
            # Plot the histogram of weights
            if self.args.softpoisson or self.args.optweight:
                import matplotlib.pyplot as plt
                fig, axs = plt.subplots()
                # plot ours
                axs.set_title(f"Epoch {self.current_epoch:05}: SP Weights")
                axs.hist(batch_parts['weights'], bins=20)
                plt.savefig(os.path.join(save_path, f"weights_epoch_{self.current_epoch:05}_batch{batch_idx}.png"))
                plt.close(fig)
                plt.cla()
                self.logger.log_image(key='weights', images=[os.path.join(save_path, f"weights_epoch_{self.current_epoch:05}_batch{batch_idx}.png")],
                                      step=self.current_epoch)

            # NOTE: batch_parts['T'] = triangle soup indexing if no poisson solve
            # If recutting Tutte: then plot the original tutte uvs
            if (self.args.init in ["tutte", 'slim'] and self.args.ninit == -1) or \
                (self.current_epoch == 0 and self.args.init):
                source = batch[0]
                if self.args.init == "tutte":
                    uv = source.tutteuv
                    uvfs = source.cutfs
                elif self.args.init == "slim":
                    uv = source.slimuv
                    uvfs = source.cutfs
                else:
                    uv = source.isofuv.reshape(-1, 2)
                    uvfs = np.arange(len(uv)).reshape(-1, 3)

                plot_uv(save_path, f"{self.args.init} init epoch {self.current_epoch:05} batch {batch_idx}", uv.squeeze().detach().cpu().numpy(),
                        uvfs, losses=None, facecolors = np.arange(len(uvfs))/(len(uvfs)),
                        stitchweight=self.args.edgecut_weight)

                # Also plot the full boundary
                initfaces = batch_parts["ogT"]
                cutmesh = Mesh(source.cutvs, source.cutfs)
                totboundaries = []
                for key, bd in sorted(cutmesh.topology.boundaries.items()):
                    boundaryvs = []
                    for v in bd.adjacentVertices():
                        boundaryvs.append(cutmesh.vertices[v.index])
                    bidx = np.array([[i, i+1] for i in range(len(boundaryvs)-1)] + [[len(boundaryvs)-1, 0]])
                    totboundaries.append(np.array(boundaryvs)[bidx])
                if len(totboundaries) > 0:
                    totboundaries = np.concatenate(totboundaries, axis=0)
                    export_views(source.cutvs, source.cutfs, save_path, filename=f"boundary_mesh_{self.current_epoch:05}_batch{batch_idx}.png",
                                    plotname=f"Initial Mesh Boundary Batch {batch_idx}", cylinders=totboundaries,
                                    device="cpu", n_sample=30, width=200, height=200,
                                    vmin=0, vmax=1, shading=False)

            ### Compute cut edges based on stitching loss (on original topology and soup topology) ###
            if 'edgeseparation' not in lossdict[0].keys():
                edge_vpairs = source.edge_vpairs.detach().cpu().numpy()
                uvpairs = batch_parts["pred_V"].squeeze()[edge_vpairs] # E x 2 x 2 x 2
                uvpairs = uvpairs[keepidxs]

                edgeseparation = torch.sum(torch.nn.functional.l1_loss(uvpairs[:,:,0,:], uvpairs[:,:,1,:], reduction='none'), dim=2)
                edgeseparation = torch.mean(edgeseparation, dim=1).detach().cpu().numpy() # E x 1
            else:
                edgeseparation = lossdict[0]['edgeseparation']

            topo_cutedges_stitch = np.where(edgeseparation > self.args.cuteps)[0]
            topo_cutedgecolors_stitch = np.arange(len(edgeseparation))/(len(edgeseparation)-1)

            # Convert to soup edges
            # NOTE: We have to subset by keepidxs to get correct corr to edgeseparation
            soup_cutedges_stitch = []
            edge_vpairs = source.edge_vpairs.detach().cpu().numpy()[keepidxs]
            for cute in topo_cutedges_stitch:
                vpair = edge_vpairs[cute] # 2 x 2
                soup_cutedges_stitch.append(batch_parts["pred_V"].squeeze()[vpair[:,0]].detach().cpu().numpy())
                soup_cutedges_stitch.append(batch_parts["pred_V"].squeeze()[vpair[:,1]].detach().cpu().numpy())
            soup_cutedgecolors_stitch = np.repeat(topo_cutedges_stitch, 2)/(len(edgeseparation)-1)

            if len(soup_cutedges_stitch) > 0:
                soup_cutedges_stitch = np.stack(soup_cutedges_stitch, axis=0) # CutE * 2 x 2 x 2

            ### Compute cut edges based on weights (on original topology and soup topology) ###
            weights = batch_parts['weights']
            # assert len(weights) == len(edgeseparation), f"weights len: {len(weights)}, edgeseparation len: {len(edgeseparation)}"

            topo_cutedges_weight = np.where(weights < self.args.weightcuteps)[0]
            topo_cutedgecolors_weight = np.arange(len(weights))/(len(weights)-1)

            # Convert to soup edges
            soup_cutedges_weight = []
            edge_vpairs = source.edge_vpairs.detach().cpu().numpy()
            for cute in topo_cutedges_weight:
                vpair = edge_vpairs[cute] # 2 x 2
                soup_cutedges_weight.append(batch_parts["pred_V"].squeeze()[vpair[:,0]].detach().cpu().numpy())
                soup_cutedges_weight.append(batch_parts["pred_V"].squeeze()[vpair[:,1]].detach().cpu().numpy())
            soup_cutedgecolors_weight = np.repeat(topo_cutedges_weight, 2)/(len(weights)-1)

            if len(soup_cutedges_weight) > 0:
                soup_cutedges_weight = np.stack(soup_cutedges_weight, axis=0) # CutE * 2 x 2 x 2

            # edges = None
            # edgecolors = None
            # vertexseparation = lossdict[0]['vertexseparation']

            # edgestitchcheck = vertexseparation[source.edgeidxs].detach().cpu().numpy()
            # cutvidxs = np.where(edgestitchcheck > self.args.cuteps)[0]
            # cutsoup = [source.valid_edges_to_soup[vi] for vi in cutvidxs]

            # # NOTE: There will be duplicates in cutsoup b/c every edge has two valid edge pairs
            # # Remove the duplicates here
            # dedupcutsoup = []
            # dedupcutvidxs = []
            # for i in range(len(cutsoup)):
            #     souppair = cutsoup[i] # 2 x 2
            #     if souppair not in dedupcutsoup:
            #         dedupcutsoup.append(souppair)
            #         dedupcutvidxs.append(cutvidxs[i])
            # print(f"Original cutsoup #: {len(cutsoup)}. Number of dedup cut edges: {len(dedupcutsoup)}.")

            # Dedup'd edge colors and cut edges
            # NOTE: This ensures consistent coloring of each edge across different cuts
            # edgecolors = np.repeat(np.array(source.edgededupidxs)[dedupcutvidxs]/(np.max(source.edgededupidxs) - 1), 2)
            # edgecolors = np.repeat(dedupcutvidxs, 2)/(len(edgestitchcheck)-1)
            # edges = []
            # for souppair in dedupcutsoup:
            #     edges.append(batch_parts["pred_V"].squeeze()[list(souppair[0])].detach().cpu().numpy())
            #     edges.append(batch_parts["pred_V"].squeeze()[list(souppair[1])].detach().cpu().numpy())
            # if len(edges) > 0:
            #     edges = np.stack(edges, axis=0)

            # Compute flips
            from utils import get_flipped_triangles
            flipped = get_flipped_triangles(batch_parts["pred_V"].squeeze().detach().cpu().numpy(), batch_parts['T'].squeeze())
            flipvals = np.zeros(len(batch_parts['T'].squeeze()))
            flipvals[flipped] = 1
            lossdict[0]['fliploss'] = flipvals

            if len(batch_parts["pred_V"].shape) == 4:
                for idx in range(len(batch_parts["pred_V"])):
                    plot_uv(save_path, f"epoch {self.current_epoch:05} batch {idx:05}", batch_parts["pred_V"][idx].squeeze().detach().cpu().numpy(),
                            batch_parts["T"][idx].squeeze(), losses=lossdict[idx], cmin=0, cmax=2, dmin=0, dmax=1,
                            facecolors = np.arange(len(batch_parts["T"][idx].squeeze()))/(len(batch_parts["T"][idx].squeeze())),
                            edges = soup_cutedges_stitch, edgecolors = soup_cutedgecolors_stitch,
                            edgecorrespondences=source.edgecorrespondences, source=source,
                            keepidxs = keepidxs, stitchweight=self.args.edgecut_weight)
            else:
                plot_uv(save_path, f"epoch {self.current_epoch:05} batch {batch_idx}", batch_parts["pred_V"].squeeze().detach().cpu().numpy(),
                        batch_parts["T"].squeeze(), losses=lossdict[0], cmin=0, cmax=2, dmin=0, dmax=1,
                        facecolors = np.arange(len(batch_parts["T"].squeeze()))/(len(batch_parts["T"].squeeze())),
                        edges = soup_cutedges_stitch, edgecolors = soup_cutedgecolors_stitch,
                        edgecorrespondences=source.edgecorrespondences, source=source,
                        keepidxs = keepidxs, stitchweight=self.args.edgecut_weight)

            # Log the plotted imgs
            images = [os.path.join(save_path, f"epoch_{self.current_epoch:05}_batch_{batch_idx}.png")] + \
                        [os.path.join(save_path, f"{key}_epoch_{self.current_epoch:05}_batch_{batch_idx}.png") for key in lossdict[0].keys() if "loss" in key]

            if self.args.init in ['tutte', 'slim'] and self.args.ninit == -1:
                images = [os.path.join(save_path, f"{self.args.init}_init_epoch_{self.current_epoch:05}_batch_{batch_idx}.png")] + images
            elif self.current_epoch == 0 and self.args.init:
                images = [os.path.join(save_path, f"{self.args.init}_init_epoch_{self.current_epoch:05}_batch_{batch_idx}.png")] + images

            # Filter out all renders that dont exist
            images = [imgpath for imgpath in images if os.path.exists(imgpath)]
            self.logger.log_image(key='uvs', images=images, step=self.current_epoch)

            #### Compute and plot hard soft poisson ####
            # NOTE: We only look at valid edge pairs to cut, even if LEARNING is done over all valid pairs!
            # May also set weights based on the stitching loss instead
            if self.args.hardpoisson:
                if self.args.hardpoisson == "loss":
                    # NOTE: Edge stitch check is L2 distances between corresponding vertices!
                    edgeweights = edgeseparation
                    threshold = self.args.cuteps
                elif self.args.hardpoisson == "weight":
                    edgeweights = weights
                    threshold = self.args.weightcuteps

                hardpoisson_uv, cutvs, cutfs, seamlengths, hardpoisson_cuts = self.hardpoisson(source, batch_parts['pred_J'], batch_parts['source_V'],
                                                                                torch.from_numpy(batch_parts['ogT']).to(self.device), edgeweights,
                                                                                vertex_pairs = source.valid_edge_pairs,
                                                                                threshold = threshold)
                hardpoisson_uv = hardpoisson_uv[:, :2]

                # Compute seam length and distortion
                # NOTE: We only have ARAP implemented for now
                from source_njf.losses import arap
                distortionenergy = arap(torch.from_numpy(cutvs).float(), torch.from_numpy(cutfs).long(), hardpoisson_uv,
                                        device=self.device, renormalize=False,
                                        return_face_energy=True, timeit=False)

                # Visualize the edge cuts
                hp_ecolors, hp_edges = None, None
                hardpoisson_uv = hardpoisson_uv.squeeze().detach().cpu().numpy()

                plot_uv(save_path, f"hard poisson epoch {self.current_epoch:05} seam length {np.sum(seamlengths):04f}", hardpoisson_uv,
                            cutfs, losses={'distortionloss': distortionenergy.detach().cpu().numpy()},
                            edges = hp_edges, edgecolors=hp_ecolors, keepidxs = keepidxs,
                            facecolors = np.arange(len(cutfs))/(len(cutfs)), stitchweight=self.args.edgecut_weight)

                images = [os.path.join(save_path, f"hard_poisson_epoch_{self.current_epoch:05}_seam_length_{np.sum(seamlengths):04f}.png")] + \
                        [os.path.join(save_path, f"distortionloss_hard_poisson_epoch_{self.current_epoch:05}_seam_length_{np.sum(seamlengths):04f}.png")]
                self.logger.log_image(key='hard poisson', images=images, step=self.current_epoch)

            ### Losses on 3D surfaces
            ### NOTE: mesh is original mesh topology (not soup)

            # Plot edge cuts
            ogvs = batch_parts["source_V"].detach().cpu().numpy()
            ogfs = batch_parts["ogT"]

            topo_cutvpairs = source.ogedge_vpairs_nobound[topo_cutedges_stitch]
            cylinderpos = ogvs[topo_cutvpairs]
            cutlen = torch.sum(source.elens_nobound[topo_cutedges_stitch]).item()

            # Stitch cut
            if len(cylinderpos) > 0:
                export_views(ogvs, batch_parts["ogT"], save_path, filename=f"stitchcuts_mesh_{self.current_epoch:05}_batch{batch_idx}.png",
                            plotname=f"Total stitchcut len: {cutlen:0.5f}", cylinders=cylinderpos,
                            outline_width=0.01, cmap = plt.get_cmap('Reds_r'),
                            device="cpu", n_sample=30, width=200, height=200,
                            vmin=0, vmax=1, shading=False)

            # Weight cut
            topo_cutvpairs = source.ogedge_vpairs_nobound[topo_cutedges_weight]
            cylinderpos = ogvs[topo_cutvpairs]
            cutlen = torch.sum(source.elens_nobound[topo_cutedges_weight]).item()

            if len(cylinderpos) > 0:
                export_views(ogvs, batch_parts["ogT"], save_path, filename=f"weightcuts_mesh_{self.current_epoch:05}_batch{batch_idx}.png",
                            plotname=f"Total weight cut len: {cutlen:0.5f}", cylinders=cylinderpos,
                            outline_width=0.01, cmap = plt.get_cmap('Reds_r'),
                            device="cpu", n_sample=30, width=200, height=200,
                            vmin=0, vmax=1, shading=False)

            # Weights
            # NOTE: below results in 2x each cylinder but it's fine
            cylinderpos = ogvs[source.ogedge_vpairs_nobound.detach().cpu().numpy()]
            cylindervals = np.stack([weights, weights], axis=1) # E x 2

            export_views(ogvs, batch_parts["ogT"], save_path, filename=f"weights_mesh_{self.current_epoch:05}_batch{batch_idx}.png",
                        plotname=f"Edge Weights", cylinders=cylinderpos,
                        cylinder_scalars=cylindervals,
                        outline_width=0.01, cmap = plt.get_cmap('Reds_r'),
                        device="cpu", n_sample=30, width=200, height=200,
                        vmin=0, vmax=1, shading=False)

            images = [os.path.join(save_path, f"weights_mesh_{self.current_epoch:05}_batch{batch_idx}.png")]
            # if os.path.exists(os.path.join(save_path, f"weightcuts_mesh_{self.current_epoch:05}_batch{batch_idx}.png")):
            #     images.append(os.path.join(save_path, f"weightcuts_mesh_{self.current_epoch:05}_batch{batch_idx}.png"))
            if (self.args.init in ["tutte", "slim"] and self.args.ninit == -1) or \
                (self.current_epoch == 0 and self.args.init):
                boundary_path = os.path.join(save_path, f"boundary_mesh_{self.current_epoch:05}_batch{batch_idx}.png")
                if os.path.exists(boundary_path):
                    images = [boundary_path] + images
            self.logger.log_image(key='pred weight', images=images, step=self.current_epoch)

            edgecorrespondences = source.edgecorrespondences
            for key, val in lossdict[0].items():
                if "loss" in key: # Hacky way of avoiding aggregated values
                    if key == "vertexseploss":
                        ogvs = batch_parts["source_V"].detach().cpu().numpy()
                        valid_pairs = source.valid_pairs
                        separationloss = val

                        # Convert separation loss to per vertex
                        from collections import defaultdict
                        vlosses = defaultdict(np.double)
                        vlosscount = defaultdict(int)
                        for i in range(len(valid_pairs)):
                            pair = valid_pairs[i]
                            vlosses[pair[0]] += separationloss[i]
                            vlosses[pair[1]] += separationloss[i]
                            vlosscount[pair[0]] += 1
                            vlosscount[pair[1]] += 1

                        # NOTE: Not all vertices will be covered in vlosses b/c they are boundary vertices
                        vseplosses = np.zeros(len(batch_parts['pred_V'])) # Soup vs
                        for k, v in sorted(vlosses.items()):
                            vseplosses[k] = v / vlosscount[k]

                        # NOTE: can't let mesh re-export the faces because the indexing will be off
                        export_views(ogvs, batch_parts["ogT"], save_path, filename=f"{key}_mesh_{self.current_epoch:05}_batch{batch_idx}.png",
                                    plotname=f"Avg {key}: {np.mean(val):0.4f}", outline_width=0.01,
                                    vcolor_vals=vseplosses,
                                    device="cpu", n_sample=30, width=200, height=200,
                                    vmin=0, vmax=0.5, shading=False)

                    elif key == "edgecutloss":
                        from collections import defaultdict
                        edgecutloss = val # E x 1
                        cylinderpos = ogvs[source.ogedge_vpairs_nobound[keepidxs].detach().cpu().numpy()]
                        cylindervals = np.stack([edgecutloss, edgecutloss], axis=1) # E x 2

                        ## Also plot edges which are excluded by keepidxs
                        excludeidxs = np.setdiff1d(np.arange(len(source.ogedge_vpairs_nobound)), keepidxs)
                        subcylinders = ogvs[source.ogedge_vpairs_nobound[excludeidxs].detach().cpu().numpy()]

                        if len(subcylinders) == 0:
                            subcylinders = None

                        export_views(ogvs, batch_parts["ogT"], save_path, filename=f"{key}_mesh_{self.current_epoch:05}_batch{batch_idx}.png",
                                    plotname=f"Avg {key}: {np.mean(val):0.4f}", cylinders=cylinderpos,
                                    cylinder_scalars=cylindervals, outline_width=0.01,
                                    device="cpu", n_sample=30, width=200, height=200,
                                    vmin=0, vmax=self.args.edgecut_weight, shading=False,
                                    subcylinders = subcylinders)

                    elif key == "edgegradloss": # E x 2
                        cylindervals = []
                        cylinderpos = []
                        ogvs = batch_parts["source_V"].detach().cpu().numpy()
                        count = 0
                        for k, v in sorted(edgecorrespondences.items()):
                            # If only one correspondence, then it is a boundary
                            if len(v) == 1:
                                continue
                            cylinderpos.append(ogvs[list(k)])
                            cylindervals.append([np.sum(val[count]), np.sum(val[count])])
                            count += 1

                        export_views(ogvs, batch_parts["ogT"], save_path, filename=f"{key}_mesh_{self.current_epoch:05}_batch{batch_idx}.png",
                                    plotname=f"Avg {key}: {np.mean(val):0.4f}", cylinders=cylinderpos,
                                    cylinder_scalars=cylindervals, outline_width=0.01,
                                    device="cpu", n_sample=30, width=200, height=200,
                                    vmin=0, vmax=0.5, shading=False)
                    elif key == "distortionloss":
                        export_views(batch_parts["source_V"].detach().cpu().numpy(), batch_parts["ogT"], save_path,
                                     filename=f"{key}_mesh_{self.current_epoch:05}_batch{batch_idx}.png",
                                        plotname=f"Avg {key}: {np.mean(val):0.4f}",
                                        fcolor_vals=val, device="cpu", n_sample=30, width=200, height=200,
                                        vmin=0, vmax=1, shading=False)
                    elif key == "invjloss":
                        export_views(batch_parts["source_V"].detach().cpu().numpy(), batch_parts["ogT"], save_path,
                                     filename=f"{key}_mesh_{self.current_epoch:05}_batch{batch_idx}.png",
                                        plotname=f"Avg {key}: {np.mean(val):0.4f}",
                                        fcolor_vals=val, device="cpu", n_sample=30, width=200, height=200,
                                        vmin=0, vmax=1, shading=False)
                    elif key == "fliploss":
                        export_views(batch_parts["source_V"].detach().cpu().numpy(), batch_parts["ogT"], save_path,
                                     filename=f"{key}_mesh_{self.current_epoch:05}_batch{batch_idx}.png",
                                        plotname=f"Avg {key}: {np.mean(val):0.4f}",
                                        fcolor_vals=val, device="cpu", n_sample=30, width=200, height=200,
                                        vmin=0, vmax=0.6, shading=False)
                    else:
                        continue

            # Log together: 3D surface losses + initial tutte cut
            images = [os.path.join(save_path, f"{key}_mesh_{self.current_epoch:05}_batch{batch_idx}.png") for key in lossdict[0].keys() if "loss" in key]

            # Filter out all renders that dont exist
            images = [imgpath for imgpath in images if os.path.exists(imgpath)]
            self.logger.log_image(key='3D losses', images=images, step=self.current_epoch)

        return val_loss

    ### TRAINING STEP HERE ###
    def my_step(self, source_batch, batch_idx, validation=False):
        # sanity checking the poisson solve, getting back GT vertices from GT jacobians. This is not used in this training.
        # GTT = batches.get_batch(0).poisson.jacobians_from_vertices(pred_V[0])
        # 		GT_V = batches.get_batch(0).poisson.solve_poisson(GTT)
        source = source_batch[0]
        target = source_batch[1]
        source.to(self.device)
        target.to(self.device)

        # Initialize stitchweights if not yet set
        if batch_idx not in self.stitchweights:
            self.stitchweights[batch_idx] = torch.ones(len(source.valid_pairs), device=self.device)

        initj = None
        initfuv = None
        sourcedim = 3
        if self.args.init == "tutte":
            sourcedim = 2
            initj = source.tuttej.squeeze().to(self.device)
            initfuv = source.tuttefuv.squeeze().to(self.device)
        elif self.args.init == "isometric":
            sourcedim = 2
            initj = source.isoj.squeeze().to(self.device) # NOTE: this is guaranteed to be isometric, so don't need to composite for computing distortion
            initfuv = source.isofuv.squeeze().to(self.device)
        elif self.args.init == "slim":
            sourcedim = 2
            initj = source.slimj.squeeze().to(self.device)
            initfuv = source.slimfuv.squeeze().to(self.device)

        # Debugging: fuvs make sense
        # if self.args.debug and self.args.init:
        #     import matplotlib.pyplot as plt
        #     fig, axs = plt.subplots(figsize=(6, 4))
        #     checkinit = initfuv.reshape(-1, 2).detach().cpu().numpy()
        #     checktris = np.arange(len(checkinit)).reshape(-1, 3)
        #     axs.triplot(checkinit[:,0], checkinit[:,1], checktris, linewidth=0.5)
        #     plt.axis('off')
        #     plt.savefig(f"scratch/{source.source_ind}_fuv.png")
        #     plt.close(fig)
        #     plt.cla()

        # Need to export mesh soup to get correct face to tutte uv indexing
        mesh = Mesh(source.get_vertices().detach().cpu().numpy(), source.get_source_triangles())
        vs, fs, es = mesh.export_soup()
        vertices = torch.from_numpy(vs).float().to(self.device)
        faces = torch.from_numpy(fs).long().to(self.device)

        if self.args.softpoisson or self.args.optweight:
            pred_V, pred_J, pred_J_poiss, pred_J_restricted_poiss, weights = self.predict_map(source, target, initj=initj if initj is not None else None)
        else:
            pred_V, pred_J, pred_J_poiss, pred_J_restricted_poiss = self.predict_map(source, target, initj=initj if initj is not None else None)

        # TODO: individual J/weight optimization -- check what happens if we start with ideal Jacobs and optimize just the weights

        # Drop last dimension of restricted J
        if pred_J_restricted_poiss.shape[2] == 3:
            pred_J_restricted_poiss = pred_J_restricted_poiss[:,:,:2]

        ## Stitching loss schedule
        if self.args.stitchschedule == "linear":
            ratio = self.current_epoch/self.trainer.max_epochs
            edgecut_weight = ratio * self.args.edgecut_weight_max + (1 - ratio) * self.args.edgecut_weight_min
            self.args.edgecut_weight = edgecut_weight
            # NOTE: Need this hack bc when lossfcn is loaded from weights args is a separate object
            self.lossfcn.args.edgecut_weight = edgecut_weight
        elif self.args.stitchschedule == "cosine":
            ratio = self.current_epoch/self.args.sparse_cosine_steps
            edgecut_weight = self.args.edgecut_weight_max - 0.5 * (self.args.edgecut_weight_max - self.args.edgecut_weight_min) * (1 + np.cos(np.pi * ratio))
            self.args.edgecut_weight = edgecut_weight
            # NOTE: Need this hack bc when lossfcn is loaded from weights args is a separate object
            self.lossfcn.args.edgecut_weight = edgecut_weight
        elif self.args.stitchschedule == "constant":
            ratio = self.current_epoch/self.trainer.max_epochs
            if ratio < self.args.stitchschedule_constant:
                edgecut_weight = 0
                self.args.edgecut_weight = 0
            else:
                edgecut_weight = self.args.edgecut_weight_max
                self.args.edgecut_weight = self.args.edgecut_weight_max
            # NOTE: Need this hack bc when lossfcn is loaded from weights args is a separate object
            self.lossfcn.args.edgecut_weight = edgecut_weight

        ## Sparse loss schedule
        if self.args.sparse_schedule == "linear":
            ratio = self.current_epoch/self.trainer.max_epochs
            sparsecuts_weight = ratio * self.args.sparselossweight_max + (1 - ratio) * self.args.sparselossweight_min
            self.args.sparsecuts_weight = sparsecuts_weight
        elif self.args.sparse_schedule == "cosine":
            ratio = self.current_epoch/self.args.sparse_cosine_steps
            sparsecuts_weight = self.args.stitchlossweight_max - 0.5 * (self.args.stitchlossweight_max - self.args.stitchlossweight_min) * (1 + np.cos(np.pi * ratio))
            self.args.sparsecuts_weight = sparsecuts_weight

        # NOTE predict_map already composites pred_J against initj
        pred_V = pred_V[:, :, :2].squeeze().reshape(-1, 3, 2)
        loss = self.lossfcn.computeloss(vertices, faces, ZeroNanGrad.apply(pred_V), ZeroNanGrad.apply(pred_J_poiss[:,:,:2,:]),
                                        weights=weights, stitchweights=self.stitchweights[batch_idx],
                                        source=source, keepidxs=source.keepidxs)

        lossrecord = self.lossfcn.exportloss()
        self.lossfcn.clear() # This resets the loss record dictionary

        ### ==== SDS Losses ==== ###
        if self.args.sdsloss:
            # Prereqs: texture image, texture description
            assert self.args.textureimg is not None and self.args.texturetext is not None, "Need to specify textureimg and texturetext for SDS loss"

            from diffusion_guidance.deepfloyd_if import DeepFloydIF
            diffusion = DeepFloydIF() # optionally you can pass a config at initialization

            # Text encoding
            text_z, text_z_neg = diffusion.encode_prompt(self.args.texturetext)

            # TODO: Render texture image on mesh for randomly sampled views
            from PIL import Image
            from torchvision.transforms.functional import pil_to_tensor
            textureimg = pil_to_tensor(Image.open(self.args.textureimg))

            sds = self.diffusion(rgb_images, text_z_inputs, text_z_neg_inputs)
            sdsloss = sds['loss_sds']

        # If running stitchweights, then update here
        # NOTE: stitchweights is len(valid_pairs) but weights is len(valid_edges)
        if self.args.stitchweight and self.args.softpoisson:
            if self.args.stitchweight == "stitchloss":
                self.stitchweights[batch_idx] = 1/(lossrecord[0]['edgeseparation'] + 1e-8)
                assert self.stitchweights[batch_idx].requires_grad == False
            elif self.args.stitchweight == "softweight":
                if self.args.softpoisson == "edges":
                    self.stitchweights[batch_idx][source.edgeidxs] = weights
                else:
                    self.stitchweights[batch_idx] = weights
            elif self.args.stitchweight == "softweightdetach":
                if self.args.softpoisson == 'edges':
                    self.stitchweights[batch_idx][source.edgeidxs] = weights.detach()
                else:
                    self.stitchweights[batch_idx] = weights.detach()
                assert self.stitchweights[batch_idx].requires_grad == False
            else:
                raise ValueError(f"Unknown stitchweight {self.args.stitchweight}")

        if self.verbose:
            print(
                f"batch of {target.get_vertices().shape[0]:d} source <--> target pairs, each mesh {target.get_vertices().shape[1]:d} vertices, {source.get_source_triangles().shape[1]:d} faces")
        ret = {
            "target_V": vertices.detach(),
            "source_V": vertices.detach(),
            "pred_V": pred_V.detach(),
            "pred_J": pred_J.detach(),
            "T": faces.detach().cpu().numpy(),
            'source_ind': source.source_ind,
            'target_inds': target.target_inds,
            "lossdict": lossrecord,
            "loss": loss,
        }

        if self.args.softpoisson or self.args.optweight:
            ret['weights'] = weights.detach().cpu().numpy()

        # Need to adjust the return values if no poisson solve
        if self.args.no_poisson and len(pred_V) == len(faces):
            ret['pred_V'] = pred_V.detach().reshape(-1, 2)

            # Triangle soup
            ret['ogT'] = ret['T'] # Save original triangle indices
            ret['T'] = np.arange(len(faces)*3).reshape(len(faces), 3)

            # Debugging: predicted fuvs make sense
            if self.args.debug:
                import matplotlib.pyplot as plt
                fig, axs = plt.subplots(figsize=(6, 4))
                axs.triplot(ret['pred_V'][:,0].detach().cpu().numpy(), ret['pred_V'][:,1].detach().cpu().numpy(), ret['T'], linewidth=0.5)
                plt.axis('off')
                plt.savefig(f"scratch/{source.source_ind}_fuv_pred.png")
                plt.close(fig)
                plt.cla()

        # if self.args.test:
        #     ret['pred_J_R'] = poisson_J_restricted.detach()
        #     ret['target_J_R'] = GT_J_restricted.detach()

        return ret

    def test_step_end(self, batch_parts):
        def screenshot(fname, V, F):
            fig = matplotlib.pyplot.figure()
            ax = matplotlib.pyplot.axes(projection='3d')
            ax.plot_trisurf(V[:, 0], V[:, 1], V[:, 2], triangles=F, edgecolor=[[0, 0, 0]], linewidth=0.01, alpha=1.0)
            matplotlib.pyplot.savefig(fname)
            matplotlib.pyplot.close(fig)

        # this next few lines make sure cupy releases all memory
        if USE_CUPY:
            mempool = cupy.get_default_memory_pool()
            pinned_mempool = cupy.get_default_pinned_memory_pool()
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
        loss = batch_parts["loss"].mean()
        if math.isnan(loss):
            print("loss is nan during validation!")
        tb = self.logger.experiment

        # tb.add_scalar("test vertex loss", batch_parts["vertex_loss"].mean().cpu().numpy(), global_step=self.global_step)
        # tb.add_scalar("test jacobian loss", batch_parts["jacobian_loss"].mean().cpu().numpy(),
        #               global_step=self.global_step)
        # colors = self.colors(batch_parts["source_V"][0].cpu().numpy(), batch_parts["T"][0])
        #
        # self.log('test_loss', loss.item(), logger=True, prog_bar=True, on_step=False, on_epoch=True)

        # TODO: fix below
        # if self.args.xp_type == "uv":
        #     if self.val_step_iter % 100 == 0:
        #         print("saving validation intermediary results.")
        #         for idx in range(len(batch_parts["pred_V"])):
        #             path = Path(self.logger.log_dir) / f"valid_batchidx_{idx:05}.png"
        #             plot_uv(path, batch_parts["target_V"][idx].squeeze().cpu().numpy(), batch_parts["T"][idx].squeeze())

        # self.log('test_loss', loss.item(), logger=True, prog_bar=True, on_step=False, on_epoch=True)
        MAX_SOURCES_TO_SAVE = 11000000
        MAX_TARGETS_TO_SAVE = 10000000

        WRITE_TB = True
        QUALITATIVE = (not self.args.statsonly) and (not self.args.only_final_stats)
        QUANTITATIVE = True
        ONLY_FINAL = self.args.only_final_stats
        sdir = 'None'
        tpath = 'None'
        if QUALITATIVE:
            colors = self.colors(batch_parts["source_V"][0].cpu().numpy(), batch_parts["T"][0])
        if QUANTITATIVE:
            pred_time = batch_parts["pred_time"]
            self.__test_stats.add_pred_time(pred_time)
        for source_batch_ind in range(len(batch_parts["pred_V"])):
            if source_batch_ind > MAX_SOURCES_TO_SAVE:
                break
            source_mesh_ind = batch_parts['source_ind'][source_batch_ind]
            if not ONLY_FINAL:
                sdir = os.path.join(self.logger.log_dir, f"S{source_mesh_ind:06d}")
                print(f'writing source {source_mesh_ind}')
                if not os.path.exists(sdir):
                    try:
                        os.mkdir(sdir)
                    except Exception as e:
                        print(f"had exception {e}, continuing to next source")
                        continue
                sfile = os.path.join(sdir, f'{source_mesh_ind:06d}')
            source_T = batch_parts["T"][source_batch_ind].squeeze()
            source_V = batch_parts["source_V"][source_batch_ind].squeeze().cpu().detach()
            source_V_n = source_V.cpu().numpy()
            if QUANTITATIVE:
                source_areas = igl.doublearea(source_V_n, source_T)
                source_area = sum(source_areas)
            if QUALITATIVE:
                igl.write_obj(sfile + ".obj", source_V_n, source_T)
                screenshot(os.path.join(self.logger.log_dir, f"S{source_mesh_ind:06d}") + '.png', source_V_n, source_T)
            if QUANTITATIVE:
                self.__test_stats.add_area(source_area)
            if WRITE_TB:
                colors = self.colors(numpy.array(source_V), numpy.array(source_T).astype(int))
                tb.add_mesh("test_source", vertices=source_V.unsqueeze(0).cpu().numpy(),
                            faces=numpy.expand_dims(source_T, 0),
                            global_step=self.global_step, colors=colors)

            for target_batch_ind in range(batch_parts["pred_V"][source_batch_ind].shape[0]):
                if target_batch_ind > MAX_TARGETS_TO_SAVE:
                    break
                target_mesh_ind = batch_parts['target_inds'][source_batch_ind][target_batch_ind]
                if not ONLY_FINAL:
                    tdir = os.path.join(sdir, f'{target_mesh_ind:06d}')
                    try:
                        os.mkdir(tdir)
                    except Exception as e:
                        print(f"had exception {e}, continuing to next source")
                        continue

                tpath = os.path.join(self.logger.log_dir, f'{target_mesh_ind:06d}_from_{source_mesh_ind:06d}')
                pred_V = batch_parts["pred_V"][source_batch_ind][target_batch_ind].squeeze().cpu().detach()
                target_V = batch_parts["target_V"][source_batch_ind][target_batch_ind].squeeze().cpu().detach()
                pred_J = batch_parts["pred_J_R"][source_batch_ind][target_batch_ind].squeeze().cpu().detach()
                target_J = batch_parts["target_J_R"][source_batch_ind][target_batch_ind].squeeze().cpu().detach()
                assert len(pred_J.shape) == 3
                assert len(target_J.shape) == 3
                target_V_n = target_V.numpy()
                pred_V_n = pred_V.numpy()

                if QUANTITATIVE:
                    target_N = igl.per_vertex_normals(target_V_n, source_T)
                    pred_N = igl.per_vertex_normals(pred_V_n, source_T)
                    dot_N = np.sum(pred_N * target_N, 1)
                    dot_N = np.clip(dot_N, 0, 1)  # adding this to avoid Nans
                    angle_N = np.arccos(dot_N)
                    angle_sum = np.sum(angle_N)
                    array_has_nan = np.isnan(angle_sum)
                    if array_has_nan:
                        print("loss is nan during angle validation!")

                    self.__test_stats.add_angle_N(np.mean(angle_N))
                    self.__test_stats.add_V(pred_V, target_V)
                    self.__test_stats.add_J(pred_J, target_J)

                if QUALITATIVE:

                    igl.write_obj(tpath + '_target.obj', target_V_n, source_T)
                    igl.write_obj(tpath + '_pred.obj', pred_V_n, source_T)
                    if self.args.xp_type == "uv":
                        save_mesh_with_uv.write_with_uv(tpath + '_source_textured_by_uv_target.obj', source_V_n,
                                                        source_T, np.delete(target_V.numpy(), 2, axis=1))
                        save_mesh_with_uv.write_with_uv(tpath + '_source_textured_by_uv_pred.obj', source_V_n, source_T,
                                                        np.delete(pred_V.numpy(), 2, axis=1))
                    else:
                        screenshot(tpath + '_pred.png', pred_V_n, source_T)
                        screenshot(tpath + '_target.png', target_V_n, source_T)
                if WRITE_TB and target_batch_ind == 0:
                    tb.add_mesh("test_target_gt", vertices=target_V.unsqueeze(0).numpy(),
                                faces=numpy.expand_dims(source_T, 0),
                                global_step=self.global_step, colors=colors)
                    tb.add_mesh("test_target_pred", vertices=pred_V.unsqueeze(0).numpy(),
                                faces=numpy.expand_dims(source_T, 0),
                                global_step=self.global_step, colors=colors)

                if self.args.xp_type == "uv":

                    if QUALITATIVE:
                        plot_uv(tpath + "_uv", target_V,
                                pred_V, source_T)

                    def save_slim_stats(fname, mat, source_areas, source_area):
                        mat = mat.double()
                        source_areas = source_areas.astype('float64')
                        source_area = source_area.astype('float64')
                        mat = np.delete(mat, 2, axis=1)
                        s = numpy.linalg.svd(mat, compute_uv=False)
                        assert len(s.shape) == 2
                        assert s.shape[0] == pred_J.shape[0]
                        s = np.sort(s, axis=1)

                        dist_metric = s.copy()
                        dist_metric[:, 0] = 1 / (dist_metric[:, 0] + 1e-8)
                        dist_metric = np.max(dist_metric, axis=1)
                        dist_metric = np.sum(dist_metric > 10)

                        s[s < 1e-8] = 1e-8
                        slim = s ** 2 + numpy.reciprocal(s) ** 2
                        det = np.linalg.det(mat)
                        assert len(det.shape) == 1 and det.shape[0] == pred_J.shape[0]
                        flips = np.sum(det <= 0)
                        avg_slim = np.sum(np.expand_dims(source_areas, 1) * slim) / (source_area)
                        if not ONLY_FINAL:
                            np.savez(f'{fname}.npz', det=det, flips=flips, avg_slim=avg_slim, slim=slim,
                                     singular_values=s, tri_area=source_area, dist_metric=dist_metric)
                        return avg_slim, flips, dist_metric

                    if QUANTITATIVE:
                        avg_slim_gt, flips_gt, dist_metric_gt = save_slim_stats(tpath + '_gt_slim_stats', target_J,
                                                                                source_areas, source_area)
                        avg_slim_pred, flips_pred, dist_metric_pred = save_slim_stats(tpath + '_pred_slim_stats',
                                                                                      pred_J, source_areas, source_area)
                        self.__test_stats.add_slim(avg_slim_pred)
                        self.__test_stats.add_flips(flips_pred)
                        self.__test_stats.add_dist_metric(dist_metric_pred)
                        self.__test_stats.add_slim_gt(avg_slim_gt)
                        self.__test_stats.add_flips_gt(flips_gt)
                        self.__test_stats.add_dist_metric_gt(dist_metric_gt)
                        with open(tpath + '_slim_summary.txt', 'w') as f:
                            f.write(f'{"":10}|  {"avg slim":20} | flips | d<10 \n')
                            f.write(f'----------------------------------------------\n')
                            f.write(f'{"GT   ":10}| {avg_slim_gt:20} | {flips_gt} | {dist_metric_gt} \n')
                            f.write(f'{"Ours ":10}| {avg_slim_pred:20} | {flips_pred} | {dist_metric_pred} \n')

        self.__test_stats.dump(os.path.join(self.logger.log_dir, 'teststats'))
        return loss

    def colors(self, v, f):
        vv = igl.per_vertex_normals(v, f)
        vv = (numpy.abs(vv) + 1) / 2
        colors = vv * 255
        return torch.from_numpy(colors).unsqueeze(0)

    def on_train_batch_end(self, outputs, batch, batch_idx, unused=0):
        # if self.global_step % FREQUENCY == 0:
        #     # hack to ensure cupy doesn't leak on us
        #     mempool = cupy.get_default_memory_pool()
        #     pinned_mempool = cupy.get_default_pinned_memory_pool()
        #     mempool.free_all_blocks()
        #     pinned_mempool.free_all_blocks()
        return

    def get_layer_normalization_type(self):
        if hasattr(self.args, 'layer_normalization'):
            layer_normalization = self.args.layer_normalization
        else:
            assert hasattr(self.args, 'batchnorm_decoder')
            layer_normalization = self.args.batchnorm_decoder
        return layer_normalization

    def get_pointnet_layer_normalization_type(self):
        if hasattr(self.args, 'pointnet_layer_normalization'):
            layer_normalization = self.args.pointnet_layer_normalization
        else:
            assert hasattr(self.args, 'batchnorm_encoder')
            layer_normalization = self.args.batchnorm_encoder
        return layer_normalization

    def on_train_batch_end(self, outputs, batch, batch_idx, unused=0):
        pass

    def configure_optimizers(self):
        if self.args.optimizer == "adam":
            optimizer = torch.optim.Adam(list(self.parameters()), lr=self.lr)
        if self.args.optimizer == "sgd":
            optimizer = torch.optim.SGD(list(self.parameters()), lr=self.lr)

        # TODO: Consider having separate LR between face latents vs jacobians
        # scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[self.args.lr_epoch_step[0],self.args.lr_epoch_step[1]], gamma=0.1)
        # scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=200, factor=0.8, threshold=0.0001,
        #                                                         min_lr=1e-7, verbose=True)

        # Add weight as additional parameter
        if self.args.optweight:
            self.trainer.fit_loop.setup_data()
            dataloader = self.trainer.train_dataloader
            for i, bundle in enumerate(dataloader):
                source, target = bundle
                initweights = source.initweights
                optweights = torch.zeros(initweights.shape).to(self.device).double() + 1e-7
                optweights.requires_grad_()
                additional_parameters = [optweights]
                optimizer.add_param_group({"params": additional_parameters, 'lr': self.lr})

        # Add translation as additional parameter
        # NOTE: With gradient stitching, we use L0 weighted least squares instead
        if self.args.no_poisson and self.args.lossedgeseparation:
            self.trainer.fit_loop.setup_data()
            dataloader = self.trainer.train_dataloader
            for i, bundle in enumerate(dataloader):
                source, target = bundle
                faces = source.get_source_triangles()
                init_translate = torch.ones(faces.shape[0], 1, 2).to(self.device).float() * 0.5
                init_translate.requires_grad_()
                additional_parameters = [init_translate]
                optimizer.add_param_group({"params": additional_parameters, 'lr': self.lr}) # Direct optimization needs to be 100x larger

        return {"optimizer": optimizer,
                # "lr_scheduler": {
                #     "scheduler": scheduler1,
                #     "monitor": "train_loss",
                #     },
                }

def custom_collate(data):
    assert len(data) == 1, 'we work on a single batch object'
    return data[0]


def load_network_from_checkpoint(gen, args=None, cpuonly = False):
    if cpuonly:
        map_location={'cuda:0':'cpu'}
        model = MyNet.load_from_checkpoint(checkpoint_path=gen, map_location=map_location)
    else:
        model = MyNet.load_from_checkpoint(checkpoint_path=gen)
    if args is None:
        args = model.args
    # model.eval()
    # below we should add checks for any argument that cannot be changed retroactively for a loaded network
    # note we should also handle things like the encoder having different fields than what we specify
    # (e.g., loading pose params even if we don't want them) but that is too much hassle to check for something
    # that in any case breaks the loaded network

    loaded_normalization = model.get_layer_normalization_type()
    loaded_pointnet_normalization = model.get_pointnet_layer_normalization_type()
    model.args = args
    cur_normalization = model.get_layer_normalization_type()
    cur_pointnet_normalization = model.get_pointnet_layer_normalization_type()

    if cur_normalization != loaded_normalization:
        warnings.warn(
            f"args specify layer normalization of type {cur_normalization}, but network loaded from checkpoint"
            f" has {loaded_normalization} normalization")
    if cur_pointnet_normalization != loaded_pointnet_normalization:
        warnings.warn(
            f"args specify pointnet layer normalization of type {cur_pointnet_normalization}, but network loaded from checkpoint"
            f" has {loaded_pointnet_normalization} normalization")
    return model


def main(gen, args):
    pl.seed_everything(48, workers=True)

    # Save directory
    save_path = os.path.join(args.outputdir, args.expname)
    if args.overwrite and os.path.exists(save_path):
        from utils import clear_directory
        clear_directory(save_path)
    Path(save_path).mkdir(exist_ok=True, parents=True)

    if not args.compute_human_data_on_the_fly:
        ### DEFAULT GOES HERE ###
        with open(os.path.join(args.root_dir_train, args.data_file)) as file:
            data = json.load(file)
            pairs_train = data['pairs']
    else:
        if args.experiment_type == "REGISTER_TEMPLATE":
            pairs_train = [(f"ren_template", f"{(2*i):08d}") for i in range(args.size_train)]
        elif  args.experiment_type == "TPOSE":
            pairs_train = [(f"{(2*i):08d}", f"{(2*i+1):08d}") for i in range(args.size_train)]
    print("TRAIN :", len(pairs_train))

    model = None
    LOADING_CHECKPOINT = isinstance(gen, str)
    if LOADING_CHECKPOINT:
        model = load_network_from_checkpoint(gen, args)
        LOADING_CHECKPOINT = gen
        gen = model.encoder

    if args.split_train_set:
        train_max_ind = math.ceil(len(pairs_train) * args.train_percentage / 100)
        train_pairs = pairs_train[:train_max_ind]
        valid_pairs = pairs_train[train_max_ind:min(len(pairs_train), train_max_ind + 10000)]
        if args.test:
            if args.test_set == 'valid':
                print('++++++++++++++++++++++ TESTING ON VALIDATION PART +++++++++++++++++++++++')
                train_pairs = valid_pairs
            if args.test_set == 'train':
                print('++++++++++++++++++++++ TESTING ON TRAINING PART +++++++++++++++++++++++')
            if args.test_set == 'all':
                print('++++++++++++++++++++++ TESTING ON ENTIRE DATASET +++++++++++++++++++++++')
                train_pairs = pairs_train

    else:
        ### DEFAULT GOES HERE ###
        if not args.compute_human_data_on_the_fly:
            with open(os.path.join(args.root_dir_test, args.data_file)) as file:
                data = json.load(file)
                pairs_test = data['pairs']
        else:
            if args.experiment_type == "REGISTER_TEMPLATE":
                pairs_test = [(f"ren_template", f"{(2*i):08d}") for i in range(args.size_test)]
            elif  args.experiment_type == "TPOSE":
                pairs_test = [(f"{(2*i):08d}", f"{(2*i+1):08d}") for i in range(args.size_test)]

        print("TEST :", len(pairs_test))
        valid_pairs = pairs_test
        train_pairs = pairs_train

    id = None
    if args.continuetrain:
        import re
        if os.path.exists(os.path.join(args.outputdir, args.expname, 'wandb', 'latest-run')):
            for idfile in os.listdir(os.path.join(args.outputdir, args.expname, 'wandb', 'latest-run')):
                if idfile.endswith(".wandb"):
                    result = re.search(r'run-([a-zA-Z0-9]+)', idfile)
                    if result is not None:
                        id = result.group(1)
                        break
        else:
            print(f"Warning: No wandb record found in {os.path.join(args.outputdir, args.expname, 'wandb', 'latest-run')}!. Starting log from scratch...")

    logger = WandbLogger(project=args.projectname, name=args.expname, save_dir=os.path.join(args.outputdir, args.expname), log_model=False,
                         offline=args.debug, resume='must' if args.continuetrain and id is not None else 'allow', id = id)

    # if args.gpu_strategy:
    #     if os.name != 'nt':  # no support for windows because of gloo
    #         if args.gpu_strategy == 'ddp':
    #             plugins = pl.plugins.training_type.DDPPlugin(find_unused_parameters=False)
    #         elif args.gpu_strategy == 'ddp_spawn':
    #             plugins = pl.plugins.training_type.DDPSpawnPlugin(find_unused_parameters=False)
    #

    checkpoint_callback = ModelCheckpoint(monitor="epoch", mode="max", save_on_train_epoch_end=True,
                                          dirpath=os.path.join(save_path, "ckpt"), every_n_epochs=10)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    ################################ TRAINER #############################
    trainer = pl.Trainer(accelerator=has_gpu, devices=args.n_devices, precision=args.precision, log_every_n_steps=200,
                         max_epochs=args.epochs, sync_batchnorm=args.n_devices != 1,
                         check_val_every_n_epoch=args.val_interval,
                         logger=logger,
                         plugins=[SLURMEnvironment(requeue_signal=SIGUSR1)] if not args.debug else None,
                         accumulate_grad_batches=args.accumulate_grad_batches,
                         num_sanity_val_steps=1,
                         enable_model_summary=False,
                         enable_progress_bar=True,
                         num_nodes=1,
                         gradient_clip_val=args.gradclip,
                         deterministic= args.deterministic,
                         callbacks=[checkpoint_callback,lr_monitor])
    ################################ TRAINER #############################
    # Cache directory
    if args.overwritecache:
        from utils import clear_directory
        traincache = os.path.join(args.root_dir_train, "cache")
        testcache = os.path.join(args.root_dir_test, "cache")
        if os.path.exists(traincache):
            clear_directory(traincache)
        if os.path.exists(testcache):
            clear_directory(testcache)

    if isinstance(trainer.precision, int):
        if trainer.precision == 16:
            use_dtype = torch.half
        elif trainer.precision == 32:
            use_dtype = torch.float
        elif trainer.precision == 64:
            use_dtype = torch.double
        else:
            raise Exception("trainer's precision is unexpected value")
    else:
        if "16" in trainer.precision:
            use_dtype = torch.half
        elif "32" in trainer.precision:
            use_dtype = torch.float
        elif "64" in trainer.precision:
            use_dtype = torch.double
        else:
            raise Exception("trainer's precision is unexpected value")

    train_dataset = DeformationDataset(train_pairs, gen.get_keys_to_load(True),
                                       gen.get_keys_to_load(False), use_dtype, train=True, args=args)

    # Initialize the highest weights dim here
    # if args.softpoisson:
    #     train_dataset.get_weights_dim()
    #     print(f"\n========== Soft poisson training: weights dim is {train_dataset.weightsdim} ========== \n")

    train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=custom_collate, pin_memory = (args.unpin_memory is None),
                              shuffle=(args.test is None), num_workers=args.workers, persistent_workers=args.workers > 0)

    if args.no_validation or args.test:
        valid_loader = None
    else:
        # NOTE: In the single cut neuralopt case -- we just copy over the training data
        if args.ninit == 1:
            valid_dataset = train_dataset
        else:
            valid_dataset = DeformationDataset(valid_pairs, gen.get_keys_to_load(True),
                                            gen.get_keys_to_load(False), use_dtype, train=False, args=args)
        # if args.softpoisson:
        #     valid_dataset.get_weights_dim()

        valid_loader = DataLoader(valid_dataset, batch_size=1, collate_fn=custom_collate, pin_memory=(args.unpin_memory is None),
                                  shuffle=False, num_workers=0, persistent_workers=0)

    # ================ #
    # model
    # ================ #
    gen.type(use_dtype)
    if model is None:
        assert (isinstance(gen, DeformationEncoder.DeformationEncoder))
        model = MyNet(gen, gen.get_code_length(train_dataset), point_dim=train_dataset.get_point_dim(), args=args,
                      face_dim = args.facedim)

    # NOTE: Network not initializing with correct device!!!
    if has_gpu == "gpu":
        model.to(torch.device("cuda:0"))
        model.lossfcn.device = torch.device("cuda:0")
    else:
        model.to(torch.device("cpu"))
        model.lossfcn.device = torch.device("cpu")

    model.type(use_dtype)
    model.lr = args.lr
    # trainer.tune(model)
    if args.overfit_one_batch:
        print("=======OVERFITTING========")
        # Going to attempt something a bit risky because it is just too slow. The goal is to load a batch once and for all and overfit it.
        overfitting_batch = next(iter(train_loader))

        class LocalDataset(torch.utils.data.Dataset):
            def __init__(self, batch):
                self.batch = batch

            def __getitem__(self, idx):
                return self.batch

            def __len__(self):
                return 1

        local_dataset = LocalDataset(overfitting_batch)
        train_loader = DataLoader(local_dataset, batch_size=1, collate_fn=custom_collate, pin_memory=(args.unpin_memory is None),
                                  num_workers=0)

        trainer = pl.Trainer(accelerator=has_gpu, devices=args.n_devices, precision=32, max_epochs=10000,
                             overfit_batches=1)
        trainer.fit(model, train_loader, ckpt_path=LOADING_CHECKPOINT if LOADING_CHECKPOINT else None)
        return

    if args.test:
        print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& TEST &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        trainer.test(model, train_loader, ckpt_path=LOADING_CHECKPOINT if LOADING_CHECKPOINT else None)
        return

    # If loss args doesn't have stitchrelax, then add it
    if not hasattr(model.lossfcn.args, "stitchrelax"):
        model.lossfcn.args.stitchrelax = True

    if not hasattr(model.lossfcn.args, "stitchlossweight"):
        model.lossfcn.args.stitchlossweight = 1

    trainer.fit(model, train_loader, valid_loader, ckpt_path=LOADING_CHECKPOINT if LOADING_CHECKPOINT else None)

    # Save UVs
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    for idx, data in enumerate(train_loader):
        devdata = (data[0].to(model.device), data[1].to(model.device))
        ret = model.my_step(devdata, idx, validation=True)
        source, target = data
        sourcepath = source.source_dir
        np.save(os.path.join(sourcepath, f"latest_preduv.npy"), ret['pred_V'].squeeze().detach().cpu().numpy())
        np.save(os.path.join(sourcepath, f"latest_predt.npy"), ret['T'])

        # if args.no_poisson:
        #     np.save(os.path.join(sourcepath, f"latest_poissonuv.npy"), ret['poissonUV'].squeeze().detach().cpu().numpy())
        #     np.save(os.path.join(sourcepath, f"latest_poissont.npy"), ret['ogT']) # NOTE: poisson T uses original triangles!

    ### GENERATE GIFS
    pref = ""
    # if args.lossgradientstitching:
    #     pref = "gradstitch_"

    from PIL import Image
    import glob
    import re
    for batchi, batch in enumerate(train_loader):
        source, target = batch
        sourcename = os.path.basename(source.source_dir)
        vispath = os.path.join(save_path, "renders", sourcename)

        ## Default UV gif
        fp_in = f"{vispath}/frames/{pref}epoch_*.png"
        fp_out = f"{vispath}/{pref}train.gif"
        imgs = [Image.open(f) for f in sorted(glob.glob(fp_in)) if re.search(r'.*(\d+)\.png', f)]

        # Resize images
        basewidth = 400
        wpercent = basewidth/imgs[0].size[0]
        newheight = int(wpercent * imgs[0].size[1])
        imgs = [img.resize((basewidth, newheight)) for img in imgs]

        imgs[0].save(fp=fp_out, format='GIF', append_images=imgs[1:],
                save_all=True, duration=100, loop=0, disposal=2)

        ## Individual losses
        lossnames = model.lossfcn.lossnames
        # if args.lossgradientstitching:
        #     lossnames.append('stitchdistortionloss')

        for key in lossnames:
            if "loss" in key:
                # Embedding viz
                fp_in = f"{vispath}/frames/{key}_{pref}epoch_*.png"
                fp_out = f"{vispath}/train_{pref}{key}.gif"
                imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]

                # Resize images
                basewidth = 400
                wpercent = basewidth/imgs[0].size[0]
                newheight = int(wpercent * imgs[0].size[1])
                imgs = [img.resize((basewidth, newheight)) for img in imgs]

                imgs[0].save(fp=fp_out, format='GIF', append_images=imgs[1:],
                        save_all=True, duration=100, loop=0, disposal=2)

                model.logger.log_image(key=f"{key} gif", images=[fp_out])

                # Mesh viz
                fp_in = f"{vispath}/frames/{key}_mesh_*.png"
                fp_out = f"{vispath}/train_{key}_mesh.gif"
                imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]

                # Resize images
                basewidth = 1000
                wpercent = basewidth/imgs[0].size[0]
                newheight = int(wpercent * imgs[0].size[1])
                imgs = [img.resize((basewidth, newheight)) for img in imgs]

                imgs[0].save(fp=fp_out, format='GIF', append_images=imgs[1:],
                        save_all=True, duration=100, loop=0, disposal=2)

        # Stitching
        if args.lossgradientstitching and args.opttrans:
            # Base
            fp_in = f"{vispath}/frames/stitch_mesh_*.png"
            fp_out = f"{vispath}/stitch_mesh.gif"
            imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]

            # Resize images
            basewidth = 1000
            wpercent = basewidth/imgs[0].size[0]
            newheight = int(wpercent * imgs[0].size[1])
            imgs = [img.resize((basewidth, newheight)) for img in imgs]
            imgs[0].save(fp=fp_out, format='GIF', append_images=imgs[1:],
                    save_all=True, duration=100, loop=0, disposal=2)

            model.logger.log_image(key=f"mesh cut gif", images=[fp_out])

        # Hard poisson
        if args.hardpoisson:
            fp_in = f"{vispath}/frames/hard_poisson_epoch_*.png"
            fp_out = f"{vispath}/hard_poisson.gif"
            imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]

            # Resize images
            basewidth = 400
            wpercent = basewidth/imgs[0].size[0]
            newheight = int(wpercent * imgs[0].size[1])
            imgs = [img.resize((basewidth, newheight)) for img in imgs]
            imgs[0].save(fp=fp_out, format='GIF', append_images=imgs[1:],
                    save_all=True, duration=100, loop=0, disposal=2)

            fp_in = f"{vispath}/frames/distortionloss_hard_poisson_epoch_*.png"
            fp_out = f"{vispath}/distortionloss_hard_poisson.gif"
            imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]

            # Resize images
            basewidth = 400
            wpercent = basewidth/imgs[0].size[0]
            newheight = int(wpercent * imgs[0].size[1])
            imgs = [img.resize((basewidth, newheight)) for img in imgs]
            imgs[0].save(fp=fp_out, format='GIF', append_images=imgs[1:],
                    save_all=True, duration=100, loop=0, disposal=2)

            model.logger.log_image(key=f"hard poisson gif", images=[f"{vispath}/hard_poisson.gif", f"{vispath}/distortionloss_hard_poisson.gif"])
    # ================ #
