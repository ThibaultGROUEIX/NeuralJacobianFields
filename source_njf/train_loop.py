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
from utils import stitchtopology
import MeshProcessor

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment
from signal import SIGUSR1

from meshing.mesh import Mesh
from meshing.analysis import computeFacetoEdges

USE_CUPY = True
has_gpu = "auto"
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
    def __init__(self, encoder, code_dim, args, point_dim=6, verbose=False):
        print("********** Some Network info...")
        print(f"********** code dim: {code_dim}")
        print(f"********** centroid dim: {point_dim}")
        super().__init__()
        self.args = args
        self.lossfcn = UVLoss(args, self.device)

        # NOTE: code dim refers to the pointnet encoding. Point_dim is centroid position (also potentially fourier features)
        layer_normalization = self.get_layer_normalization_type()
        if hasattr(self.args, "dense") and self.args.dense:
            print("==== We are predicting FLAT vectors! ==== ")
            if self.args.dense == "xyz":
                channels = (point_dim + code_dim) * 3
            else:
                channels = point_dim + code_dim
            self.per_face_decoder = nn.Sequential(nn.Linear(point_dim + code_dim, channels),
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
            # print("Using IDENTITY (no normalization) in per_face_decoder!")
            self.per_face_decoder = nn.Sequential(nn.Linear(point_dim + code_dim, 128),
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
            # print("Using BATCHNORM in per_face_decoder!")
            self.per_face_decoder = nn.Sequential(nn.Linear(point_dim + code_dim, 128),
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
            # print("Using GROUPNORM2 in per_face_decoder!")
            self.per_face_decoder = nn.Sequential(nn.Conv1d(point_dim + code_dim, 128, 1),
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
            # print("Using GROUPNORM in per_face_decoder!")
            self.per_face_decoder = nn.Sequential(nn.Linear(point_dim + code_dim, 128),
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
            # print("Using LAYERNORM in per_face_decoder!")
            self.per_face_decoder = nn.Sequential(nn.Linear(point_dim + code_dim, 128),
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

        self.__IDENTITY_INIT = self.args.identity
        if self.__IDENTITY_INIT:
            self.per_face_decoder[-1].bias.data.zero_()
            self.per_face_decoder[-1].weight.data.zero_()

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

    ##################
    # inference code below
    ##################
    def forward(self, x):
        '''
		The MLP applied to a (batch) of global code concatenated to a centroid (z|c)
		:param x: B x (|z|+|c|) batch of (z|c) vectors
		:return: B x 9 batch of 9 values that are the 3x3 matrix predictions for each input vector
		'''
        if self.code_dim + self.point_dim < x.shape[1]:
            print("WARNING: discarding part of the latent code.")
            x = x[:, :self.code_dim + self.point_dim]

        return self.per_face_decoder(x.type(self.per_face_decoder[0].bias.type()))

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
            stacked = source.get_centroids_and_normals()
        else:
            # take all encodings z_i of targets, and all centroids c_j of triangles, and create a cartesian product of the two as a 2D tensor so each sample in it is a vector with rows (z_i|c_j)
            net_input = PerCentroidBatchMaker.PerCentroidBatchMaker(codes, source.get_centroids_and_normals(), args=self.args)
            stacked = net_input.to_stacked()
            if self.args.layer_normalization != "GROUPNORM2":
                stacked = net_input.prep_for_linear_layer(stacked)
            else:
                stacked = net_input.prep_for_conv1d(stacked)
        # feed the 2D tensor through the network, and get a 3x3 matrix for each (z_i|c_j)
        res = self.forward(stacked)

        # No global codes
        if self.args.dense or codes is None:
            ret = res.reshape(1, source.mesh_processor.faces.shape[0], 3, 3)

            if self.__IDENTITY_INIT:
                for i in range(0, 3):
                    ret[:, :, i, i] += 1

            return ret

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

        # torch.cuda.synchronize()
        # print(f"training_step  {time.time() - start}")
        # self.log("V_loss", self.__vertex_loss_weight * a['vertex_loss'].item(), prog_bar=True, logger=False)
        # self.log("J_loss", a['jacobian_loss'].item(), prog_bar=True, logger=False)

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

    def get_gt_map(self, source, target):
        GT_V = target.get_vertices()
        # if self.__align_2d:  # need to align source and target using procrustes
        # 	assert self.__2D_target  # this should only be applied when the target is in fact a 2D thing like UV
        # 	# now just apply procrustes for each GT target in our batch and align it to prediction
        # 	gt = GT_V
        # 	pred = pred_V
        # 	gt = two_dimensional_procrustes(gt.cpu(), pred.clone().detach().cpu())
        # 	GT_V = gt.to(self.device)
        # ground truth jacobians, restricted as well
        GT_J = source.jacobians_from_vertices(GT_V)
        GT_J_restricted = source.restrict_jacobians(GT_J)
        return GT_V, GT_J, GT_J_restricted

    def predict_map(self, source, target, initj=None):
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

        pred_V = source.vertices_from_jacobians(pred_J)

        # Get back jacobians from predicted vertices
        pred_J = source.poisson.jacobians_from_vertices(pred_V)

        pred_J_restricted = source.restrict_jacobians(pred_J)
        return pred_V, pred_J, pred_J_restricted

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
        lossdict = batch_parts['lossdict']

        # Construct mesh
        mesh = Mesh(batch_parts["source_V"].detach().cpu().numpy(), batch_parts["ogT"])

        # NOTE: BELOW DEPRECATED -- we save the edge correspondences when we compute the edge losses!
        # Ftoe matrix
        # # Face to edges map
        # ftoe = computeFacetoEdges(mesh)

        # # Remap indexing to ignore the boundaries
        # neweidx = []
        # oldeidx = []
        # count = 0
        # for key, edge in sorted(mesh.topology.edges.items()):
        #     if not edge.onBoundary():
        #         neweidx.append(count)
        #         oldeidx.append(edge.index)
        #         count += 1
        # ebdtoe = np.zeros(np.max(list(mesh.topology.edges.keys())) + 1)
        # ebdtoe[oldeidx] = neweidx
        # ebdtoe = ebdtoe.astype(int)

        # new_ftoe = []
        # for es in ftoe:
        #     new_ftoe.append(ebdtoe[es])
        # ftoe = new_ftoe

        # Log losses
        for key, val in lossdict[0].items():
            if "loss" in key:
                self.log(f"val_{key}", np.mean(val), logger=True, prog_bar=False, batch_size=1, on_epoch=True, on_step=False)

        # Log path
        source, target = batch
        sourcename = os.path.basename(source.source_dir)
        source_path = os.path.join(self.logger.save_dir, "renders", sourcename)
        save_path = os.path.join(self.logger.save_dir, "renders", sourcename, "frames")

        if not os.path.exists(save_path):
            Path(save_path).mkdir(exist_ok=True, parents=True)

        # Save latest predictions
        np.save(os.path.join(source_path, f"latest_preduv.npy"), batch_parts['pred_V'].squeeze().detach().cpu().numpy())

        if self.args.opttrans:
            np.save(os.path.join(source_path, f"latest_preduv.npy"), batch_parts['pred_V_opttrans'].squeeze().detach().cpu().numpy())

        np.save(os.path.join(source_path, f"latest_predt.npy"), batch_parts['T'])

        # if self.args.no_poisson:
        #     np.save(os.path.join(source_path, f"latest_poissonuv.npy"), batch_parts['poissonUV'].squeeze().detach().cpu().numpy())
        #     np.save(os.path.join(source_path, f"latest_poissont.npy"), batch_parts['ogT']) # NOTE: poisson T uses original triangles!

        val_loss = batch_parts['loss'].item()
        if self.args.xp_type == "uv":
            # NOTE: batch_parts['T'] = triangle soup indexing if no poisson solve
            # If recutting Tutte: then plot the original tutte uvs
            if self.args.init == "tutte" and self.args.ninit == -1:
                source = batch[0]
                initj = source.tuttej.squeeze().to(self.device)
                initfuv = source.tuttefuv.squeeze().to(self.device)
                tutteuv = initfuv.reshape(-1, 2)
                tuttefaces = np.arange(len(tutteuv)).reshape(-1, 3)
                plot_uv(save_path, f"tutte init epoch {self.current_epoch:05}", tutteuv.squeeze().detach().cpu().numpy(),
                            tuttefaces, losses=None)

            if len(batch_parts["pred_V"].shape) == 4:
                for idx in range(len(batch_parts["pred_V"])):
                    plot_uv(save_path, f"epoch {self.current_epoch:05} batch {idx:05}", batch_parts["pred_V"][idx].squeeze().detach().cpu().numpy(),
                            batch_parts["T"][idx].squeeze(), losses=lossdict[idx])
            else:
                plot_uv(save_path, f"epoch {self.current_epoch:05}", batch_parts["pred_V"].squeeze().detach().cpu().numpy(),
                        batch_parts["T"].squeeze(), losses=lossdict[0])

            # Log the plotted imgs
            images = [os.path.join(save_path, f"epoch_{self.current_epoch:05}.png")] + \
                        [os.path.join(save_path, f"{key}_epoch_{self.current_epoch:05}.png") for key in lossdict[0].keys() if "loss" in key]
            self.logger.log_image(key='uvs', images=images, step=self.current_epoch)

            if "pred_V_opttrans" in batch_parts.keys():
                plot_uv(save_path, f"opttrans epoch {self.current_epoch:05}", batch_parts["pred_V_opttrans"].squeeze().detach().cpu().numpy(),
                            batch_parts["T"].squeeze(), losses=lossdict[0])

                # Log the plotted imgs
                images = [os.path.join(save_path, f"opttrans_epoch_{self.current_epoch:05}.png")] + \
                            [os.path.join(save_path, f"{key}_opttrans_epoch_{self.current_epoch:05}.png") for key in lossdict[0].keys() if "loss" in key]
                self.logger.log_image(key='opttrans uvs', images=images, step=self.current_epoch)

            ### Losses on 3D surfaces
            # NOTE: mesh is original mesh topology (not soup)
            for key, val in lossdict[0].items():
                if "loss" in key: # Hacky way of avoiding aggregated values
                    if "edge" in key:
                        # Vtoeloss: F*3 x 1 (vertices ordered by face indexing)
                        edgecorrespondences = lossdict[0]['edgecorrespondences']

                        # fresnel mesh: ordered by fverts flatten
                        # Edgekey to edge loss map
                        edgekeys = list(sorted(edgecorrespondences.keys()))
                        edgekeytoeloss = {edgekeys[i]: i for i in range(len(edgekeys))}
                        vtoeloss = np.zeros(len(mesh.faces)*3)
                        count = 0
                        for edgekey, v in sorted(edgecorrespondences.items()):
                            # If only one correspondence, then it is a boundary
                            if len(v) == 1:
                                continue
                            eloss = val[count]
                            vtoeloss[list(v[0])] += eloss
                            vtoeloss[list(v[1])] += eloss
                            count += 1

                        export_views(mesh, save_path, filename=f"{key}_mesh_{self.current_epoch:05}.png",
                                    plotname=f"Sum {key}: {np.sum(vtoeloss):0.4f}",
                                    vcolor_vals=vtoeloss, device="cpu", n_sample=25, width=200, height=200,
                                    vmin=0, vmax=2, shading=False)
                    else:
                        export_views(mesh, save_path, filename=f"{key}_mesh_{self.current_epoch:05}.png",
                                    plotname=f"Sum {key}: {np.sum(val):0.4f}",
                                    fcolor_vals=val, device="cpu", n_sample=25, width=200, height=200,
                                    vmin=0, vmax=0.6, shading=False)

            ## Poisson values
            # Check poisson UVs (true UV)
            if "poissonUV" in batch_parts.keys():
                plot_uv(save_path, f"poisson epoch {self.current_epoch:05}", batch_parts["poissonUV"].squeeze().detach().cpu().numpy(),
                        batch_parts["ogT"].squeeze(), losses={'distortionloss': batch_parts['poissonDistortion']})

                images = [os.path.join(save_path, f"poisson_epoch_{self.current_epoch:05}.png"),
                          os.path.join(save_path, f"distortionloss_poisson_epoch_{self.current_epoch:05}.png")]
                self.logger.log_image(key='poisson uvs', images=images, step=self.current_epoch)


                export_views(mesh, save_path, filename=f"poisson_mesh_{self.current_epoch:05}.png",
                            plotname=f"Poisson Distortion Loss: {np.sum(batch_parts['poissonDistortion']):0.4f}",
                            fcolor_vals=batch_parts['poissonDistortion'], device="cpu", n_sample=25, width=200, height=200,
                            vmin=0, vmax=0.6, shading=False)

            if self.args.lossgradientstitching and self.args.opttrans:
                # Convert edge cuts to vertex values (separate for each tri => in order of tris)
                cutedges = batch_parts['cutEdges'] # Indices of cut edges
                cutvs = []
                for eidx, e in sorted(mesh.topology.edges.items()):
                    if eidx in cutedges:
                        cutvs.append(e.halfedge.vertex.index)
                        cutvs.append(e.halfedge.twin.vertex.index)
                cutvs = list(set(cutvs))

                vs, fs, es = mesh.export_soup()
                # cutvs = np.unique(es[cutedges]) # Indices of cut vs
                vcut_vals = []
                for f in fs:
                    for v in f:
                        if v in cutvs:
                            vcut_vals.append(1)
                        else:
                            vcut_vals.append(0)
                vcut_vals = np.array(vcut_vals)

                export_views(mesh, save_path, filename=f"stitch_mesh_{self.current_epoch:05}.png",
                            plotname=f"Total Cut Length: {np.sum(batch_parts['cutLength']):0.4f}",
                            vcolor_vals= vcut_vals, device="cpu", n_sample=25, width=200, height=200,
                            vmin=0, vmax=1, outline_width=0.005, shading=False)

                images = [os.path.join(save_path, f"stitch_mesh_{self.current_epoch:05}.png")]
                self.logger.log_image(key='opttrans cut', images=images, step=self.current_epoch)

        return val_loss

    ### TRAINING STEP HERE ###
    def my_step(self, source_batch, batch_idx, validation=False):
        # sanity checking the poisson solve, getting back GT vertices from GT jacobians. This is not used in this training.
        # GTT = batches.get_batch(0).poisson.jacobians_from_vertices(pred_V[0])
        # 		GT_V = batches.get_batch(0).poisson.solve_poisson(GTT)

        # TODO: include in source batch the precomputed Jacobian of Tutte embeddings
        source = source_batch[0]
        target = source_batch[1]

        initj = None
        initfuv = None
        sourcedim = 3
        if self.args.init == "tutte":
            initj = source.tuttej.squeeze().to(self.device)
            initfuv = source.tuttefuv.squeeze().to(self.device)
        elif self.args.init == "isometric":
            sourcedim = 2
            initj = None # NOTE: this is guaranteed to be isometric, so don't need to composite for computing distortion
            initfuv = source.isofuv.squeeze().to(self.device)

        # Debugging: tutte fuvs make sense
        if self.args.debug and self.args.init:
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(figsize=(6, 4))
            checkinit = initfuv.reshape(-1, 2).detach().cpu().numpy()
            checktris = np.arange(len(checkinit)).reshape(-1, 3)
            axs.triplot(checkinit[:,0], checkinit[:,1], checktris, linewidth=0.5)
            plt.axis('off')
            plt.savefig(f"scratch/{source.source_ind}_fuv.png")
            plt.close(fig)
            plt.cla()

        # Need to export mesh soup to get correct face to tutte uv indexing
        mesh = Mesh(source.get_vertices().detach().cpu().numpy(), source.get_source_triangles())
        vs, fs, es = mesh.export_soup()
        vertices = torch.from_numpy(vs).float().to(self.device)
        faces = torch.from_numpy(fs).long().to(self.device)

        if self.args.no_poisson:
            pred_J = self.predict_jacobians(source, target).squeeze() # NOTE: this takes B x F x 3 x 3 => F x 3 x 3
            pred_J_restricted = source.restrict_jacobians(pred_J)

            # Compute the UV map by dropping last dimension of predJ and multiplying against the face matrices
            fverts = vertices[faces].transpose(1,2) # F x 3 x 3
            if self.args.init:
                # Batch size 1
                # TODO: swap the einsums so we interpret Jacobians correctly (NOT the transpose)
                # Reconstruct the initialization UV (minus global translation)
                pred_V = torch.einsum("bcd,bde->bce", (pred_J[:,:2,:2], initfuv.transpose(1,2))).transpose(1,2) # F x 3 x 2
                # pred_V = torch.einsum("bcd,bde->bce", (tuttefuv, pred_J[:,:2,:2])) # F x 3 x 2
                # pred_V = torch.einsum("bcd,bce,bej->bcj", (fverts, tuttej, pred_J[:,:2,:2]))
            else:
                if len(fverts.shape) == 3:
                    pred_V = torch.einsum("bcd,bde->bce", (pred_J[:,:2,:], fverts)).transpose(1,2) # F x 3 x 2
                    # pred_V = torch.einsum("bcd,bde->bce", (fverts, pred_J[:,:,:2])) # F x 3 x 2
                else:
                    pred_V = torch.einsum("abcd,abde->abce", (pred_J[:,:,:2,:], fverts)).transpose(2,3) # B x F x 3 x 2
                    # pred_V = torch.einsum("abcd,abde->abce", (fverts, pred_J[:,:,:,:2])) # B x F x 3 x 2

            # If gradient stitching, then we save a new tensor with the translated components
            # NOTE: stop gradient here so the translation loss doesn't affect the network
            # if self.args.lossgradientstitching:
            #     trans_V = pred_V.detach() + self.trainer.optimizers[0].param_groups[batch_idx + 1]['params'][0]
            # Add back the optimized translational components
            if not self.args.lossgradientstitching:
                pred_V += self.trainer.optimizers[0].param_groups[batch_idx + 1]['params'][0]

            # Center (for visualization purposes)
            # if len(fverts.shape) == 3:
            #     pred_V = pred_V - torch.mean(pred_V, dim=[0,1], keepdim=True)
            # else:
            #     pred_V = pred_V - torch.mean(pred_V, dim=[1,2], keepdim=True)
        else:
            pred_V, pred_J, pred_J_restricted = self.predict_map(source, target, initj=initj if initj is not None else None)

        # Drop last dimension of restricted J
        if pred_J_restricted.shape[2] == 3:
            pred_J_restricted = pred_J_restricted[:,:,:2]
        GT_V, GT_J, GT_J_restricted = self.get_gt_map(source, target)

        if UNIT_TEST_POISSON_SOLVE:
            success = self.check_map(source, target, GT_J, GT_V) < 0.0001
            # print(self.check_map(source, target, GT_J, GT_V))
            assert(success), f"UNIT_TEST_POISSON_SOLVE FAILED!! {self.check_map(source, target, GT_J, GT_V)}"

        # Compute losses
        # HACK: lerp seplossdelta
        if self.args.seploss_schedule:
            ratio = self.global_step/self.trainer.max_epochs
            seplossdelta = ratio * self.args.seplossdelta_min + (1 - ratio) * self.args.seplossdelta
        else:
            seplossdelta = self.args.seplossdelta

        ## Stitching loss schedule
        if self.args.stitchloss_schedule == "linear":
            ratio = self.global_step/self.trainer.max_epochs
            stitchweight = ratio * self.args.stitchlossweight_max + (1 - ratio) * self.args.stitchlossweight_min
            self.args.stitchlossweight = stitchweight
        elif self.args.stitchloss_schedule == "cosine":
            # TODO
            ratio = self.global_step/self.trainer.max_epochs
            stitchweight = ratio * self.args.stitchlossweight_max + (1 - ratio) * self.args.stitchlossweight_min
            self.args.stitchlossweight = stitchweight

        # NOTE: pred_V should be F x 3 x 2
        loss = self.lossfcn.computeloss(vertices, faces, pred_V, pred_J[:,:2,:sourcedim], initjacobs=initj,
                                        seplossdelta=seplossdelta, transuv=None)
        lossrecord = self.lossfcn.exportloss()
        self.lossfcn.clear() # This resets the loss record dictionary

        loss = loss.type_as(GT_V)
        if self.verbose:
            print(
                f"batch of {target.get_vertices().shape[0]:d} source <--> target pairs, each mesh {target.get_vertices().shape[1]:d} vertices, {source.get_source_triangles().shape[1]:d} faces")
        ret = {
            "target_V": vertices.detach(),
            "source_V": vertices.detach(),
            "pred_V": pred_V.detach(),
            "T": faces.detach().cpu().numpy(),
            'source_ind': source.source_ind,
            'target_inds': target.target_inds,
            "lossdict": lossrecord,
            "loss": loss
        }

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

        if validation:
            # NOTE: We use isoj only to get the final UVs
            if self.args.init == "isometric":
                initj = source.isoj.squeeze().to(self.device)

            # NOTE: Post-process topology if running edge gradient optimization
            # Run poisson solve on updated topology
            if self.args.opttrans:
                # Replace predV with L0 lstsq translations
                from source_njf.utils import leastSquaresTranslation

                vs = source.get_vertices().detach().cpu().numpy()
                fs = faces.detach().cpu().numpy()

                opttrans, cutedges, cutlength = leastSquaresTranslation(vs, fs, pred_V.detach().cpu().numpy(), return_cuts = True,
                                                   iterate=True, debug=False, patience=5, cut_epsilon=self.args.cuteps)
                ret['pred_V_opttrans'] = (pred_V.detach() + torch.from_numpy(opttrans).to(self.device)).reshape(-1, 2)
                ret['cutEdges'] = cutedges
                ret['cutLength'] = cutlength

                # TODO: WE'RE CUTTING THE BOUNDARY EDGES ON ACCIDENT
                # postv, postf, cutedges, cutlength = stitchtopology(ret['source_V'].detach().cpu().numpy(), ret['ogT'], lossrecord[0]['edgegradloss'],
                #                                                    epsilon=self.args.stitcheps,
                #                                                    return_cut_edges=True, return_cut_length = True)
                # meshprocessor = MeshProcessor.MeshProcessor.meshprocessor_from_array(postv, postf, source.source_dir, source._SourceMesh__ttype, cpuonly=source.cpuonly, load_wks_samples=source._SourceMesh__use_wks, load_wks_centroids=source._SourceMesh__use_wks)
                # meshprocessor.prepare_temporary_differential_operators(source._SourceMesh__ttype)
                # poissonsolver = meshprocessor.diff_ops.poisson_solver
                # with torch.no_grad():
                #     # Need full Jacobian for poisson solve
                #     if len(pred_J.shape) == 3:
                #         pred_J = pred_J.unsqueeze(0)
                #     if len(initj.shape) == 3:
                #         initj = initj.unsqueeze(0)
                #     if initj is not None:
                #         pred_J = torch.cat([torch.einsum("abcd,abde->abce", (pred_J[:,:,:2,:2], initj[:,:,:2,:])),
                #                             torch.zeros(pred_J.shape[0], pred_J.shape[1], 1, pred_J.shape[3], device=self.device)],
                #                         dim=2) # B x F x 3 x 3
                #     if self.args.align_2D:  # if the target is in 2D the last column of J is 0
                #         pred_J[:, :,2,:] = 0 # NOTE: predJ shape is B x F x 3 x 3 (where first 3-size is interpreted as axes)
                #     # TODO: ASK NOAM ABOUT WHETHER YOU CAN DO THIS WITH TRIANGLE SOUP
                #     stitchv = poissonsolver.solve_poisson(pred_J)
                #     stitchJ = poissonsolver.jacobians_from_vertices(stitchv)
            else:
                ret['pred_V'] = pred_V.detach().reshape(-1, 2)
            ret['T'] = np.arange(len(faces)*3).reshape(len(faces), 3)

            # NOTE: Poisson solve on updated topology
            # with torch.no_grad():
            #     pred_V, poisson_J, poisson_J_restricted = self.predict_map(source, target, initj=initj if initj is not None else None)

            # # Predicted UV should also be same up to global translation
            # # predfvert = ret['pred_V'][ret['T']]
            # # poisfvert = pred_V[0][ret['ogT']]
            # # diff = predfvert - poisfvert[:,:,:2]

            # ret['poissonUV'] = pred_V
            # # NOTE: Initj already FACTORED into the predict map!!
            # poissondirichlet = symmetricdirichlet(vertices, faces, poisson_J[0, :,:2,:])
            # ret['poissonDistortion'] = poissondirichlet.detach().cpu().numpy()

        if self.args.test:
            ret['pred_J_R'] = poisson_J_restricted.detach()
            ret['target_J_R'] = GT_J_restricted.detach()

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
        # scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[self.args.lr_epoch_step[0],self.args.lr_epoch_step[1]], gamma=0.1)
        scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=200, factor=0.8, threshold=0.0001,
                                                                min_lr=1e-7, verbose=True)

        # Add translation as additional parameter
        # NOTE: With gradient stitching, we use L0 weighted least squares instead
        if self.args.no_poisson and not self.args.lossgradientstitching:
            self.trainer.fit_loop.setup_data()
            dataloader = self.trainer.train_dataloader
            for i, bundle in enumerate(dataloader):
                source, target = bundle
                faces = source.get_source_triangles()
                init_translate = torch.ones(faces.shape[0], 1, 2).to(self.device).float() * 0.5
                init_translate.requires_grad_()
                additional_parameters = [init_translate]
                optimizer.add_param_group({"params": additional_parameters, 'lr': self.lr}) # Direct optimization needs to be 100x larger

                # HACK: Need to also extend scheduler's min_lrs
                scheduler1.min_lrs.append(1e-7)

        return {"optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler1,
                    "monitor": "train_loss",
                    },
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
                         strategy='ddp',
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

    if trainer.precision == 16 or "16" in trainer.precision:
        use_dtype = torch.half
    elif trainer.precision == 32 or "32" in trainer.precision:
        use_dtype = torch.float
    elif trainer.precision == 64 or "64" in trainer.precision:
        use_dtype = torch.double
    else:
        raise Exception("trainer's precision is unexpected value")

    train_dataset = DeformationDataset(train_pairs, gen.get_keys_to_load(True),
                                       gen.get_keys_to_load(False), use_dtype, train=True, args=args)
    train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=custom_collate, pin_memory = (args.unpin_memory is None),
                              shuffle=(args.test is None), num_workers=args.workers, persistent_workers=args.workers > 0)

    if args.no_validation or args.test:
        valid_loader = None
    else:
        valid_dataset = DeformationDataset(valid_pairs, gen.get_keys_to_load(True),
                                           gen.get_keys_to_load(False), use_dtype, train=False, args=args)
        valid_loader = DataLoader(valid_dataset, batch_size=1, collate_fn=custom_collate, pin_memory=(args.unpin_memory is None),
                                  shuffle=False, num_workers=0, persistent_workers=0)


    # ================ #
    # model
    # ================ #
    gen.type(use_dtype)
    if model is None:
        assert (isinstance(gen, DeformationEncoder.DeformationEncoder))
        model = MyNet(gen, gen.get_code_length(train_dataset), point_dim=train_dataset.get_point_dim(), args=args)

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
            fp_out = f"{vispath}/frames/stitch_mesh.gif"
            imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]

            # Resize images
            basewidth = 1000
            wpercent = basewidth/imgs[0].size[0]
            newheight = int(wpercent * imgs[0].size[1])
            imgs = [img.resize((basewidth, newheight)) for img in imgs]
            imgs[0].save(fp=fp_out, format='GIF', append_images=imgs[1:],
                    save_all=True, duration=100, loop=0, disposal=2)

            model.logger.log_image(key=f"mesh cut gif", images=[fp_out])

        ## Poisson solve
        # if args.no_poisson:
        #     # Base
        #     fp_in = f"{vispath}/frames/poisson_epoch_*.png"
        #     fp_out = f"{vispath}/train_poisson.gif"
        #     imgs = [Image.open(f) for f in sorted(glob.glob(fp_in)) if re.search(r'.*(\d+)\.png', f)]

        #     # Resize images
        #     basewidth = 400
        #     wpercent = basewidth/imgs[0].size[0]
        #     newheight = int(wpercent * imgs[0].size[1])
        #     imgs = [img.resize((basewidth, newheight)) for img in imgs]
        #     imgs[0].save(fp=fp_out, format='GIF', append_images=imgs[1:],
        #             save_all=True, duration=100, loop=0, disposal=2)

        #     # Embedding distortion
        #     fp_in = f"{vispath}/frames/distortionloss_poisson_epoch_*.png"
        #     fp_out = f"{vispath}/train_poisson_distortionloss.gif"
        #     imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]

        #     # Resize images
        #     basewidth = 400
        #     wpercent = basewidth/imgs[0].size[0]
        #     newheight = int(wpercent * imgs[0].size[1])
        #     imgs = [img.resize((basewidth, newheight)) for img in imgs]
        #     imgs[0].save(fp=fp_out, format='GIF', append_images=imgs[1:],
        #             save_all=True, duration=100, loop=0, disposal=2)

        #     # Mesh distortion
        #     fp_in = f"{vispath}/frames/poisson_mesh_*.png"
        #     fp_out = f"{vispath}/train_poisson_mesh.gif"
        #     imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]

        #     # Resize images
        #     basewidth = 1000
        #     wpercent = basewidth/imgs[0].size[0]
        #     newheight = int(wpercent * imgs[0].size[1])
        #     imgs = [img.resize((basewidth, newheight)) for img in imgs]
        #     imgs[0].save(fp=fp_out, format='GIF', append_images=imgs[1:],
        #             save_all=True, duration=100, loop=0, disposal=2)
    # ================ #
