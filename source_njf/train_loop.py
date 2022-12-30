#defines the network and the train loop
import warnings

import igl
import matplotlib
import numpy as np
import numpy.random
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import DeformationEncoder
import SourceToTargetsFile
from results_saving_scripts import save_mesh_with_uv
from DeformationDataset import DeformationDataset
from torch.utils.data import DataLoader
from torch import nn
import torch
import PerCentroidBatchMaker
import time

import pytorch_lightning as pl
from lightning_lite.utilities.seed import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

USE_CUPY = False
has_gpu = "cpu" 
if USE_CUPY and torch.cuda.is_available():
    import cupy
    has_gpu="gpu"

import math
from pytorch_lightning.loggers import TensorBoardLogger
from results_saving_scripts.plot_uv import plot_uv
from pathlib import Path
from results_saving_scripts import paper_stats
import json

FREQUENCY = 100 # frequency of logguing - every FREQUENCY iteration step
UNIT_TEST_POISSON_SOLVE = False

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

        layer_normalization = self.get_layer_normalization_type()
        if layer_normalization == "IDENTITY":
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

        self.__IDENTITY_INIT = True
        if self.__IDENTITY_INIT:
            self.per_face_decoder._modules["12"].bias.data.zero_()
            self.per_face_decoder._modules["12"].weight.data.zero_()
        
        self.__global_trans = False
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
        # take all encodings z_i of targets, and all centroids c_j of triangles, and create a cartesian product of the two as a 2D tensor so each sample in it is a vector with rows (z_i|c_j)
        net_input = PerCentroidBatchMaker.PerCentroidBatchMaker(codes, source.get_centroids_and_normals(), args=self.args)
        stacked = net_input.to_stacked()
        if self.args.layer_normalization != "GROUPNORM2":
            stacked = net_input.prep_for_linear_layer(stacked)
        else:
            stacked = net_input.prep_for_conv1d(stacked)
        # feed the 2D tensor through the network, and get a 3x3 matrix for each (z_i|c_j)
        res = self.forward(stacked)
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
        a = self.my_step(source_batches, batch_id)
        # torch.cuda.synchronize()
        # print(f"training_step  {time.time() - start}")
        # self.log("V_loss", self.__vertex_loss_weight * a['vertex_loss'].item(), prog_bar=True, logger=False)
        # self.log("J_loss", a['jacobian_loss'].item(), prog_bar=True, logger=False)
        return a

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

    def predict_map(self, source, target):
        pred_J = self.predict_jacobians(source, target)
        if self.args.align_2D:  # if the target is in 2D the last column of J is 0
            pred_J[:, :,2] = 0  # TODO this line is sus, shouldn't it be pred_J[:,:,2,:], and in any case this hould happen w.r.t. restricted jacobians!!!!
            
        # TODO: debug this -- check shape of pred_V and what happens when align_2D is turned on? 
        pred_V = source.vertices_from_jacobians(pred_J)
        pred_J_restricted = source.restrict_jacobians(pred_J) # Should'nt this line come before?
        return pred_V, pred_J, pred_J_restricted

    def check_map(self, source, target, GT_J, GT_V):
        pred_V = source.vertices_from_jacobians(GT_J)
        return torch.max(torch.absolute(pred_V - GT_V))


    def validation_step(self, batch, batch_idx):
        return self.my_step(batch, batch_idx)

    ### TRAINING STEP HERE ###
    def my_step(self, source_batch, batch_idx, test=False):
        vertex_loss = torch.tensor(0.0, device=self.device)
        jacobian_loss = torch.tensor(0.0, device=self.device)

        # sanity checking the poisson solve, getting back GT vertices from GT jacobians. This is not used in this training.
        # GTT = batches.get_batch(0).poisson.jacobians_from_vertices(pred_V[0])
        # 		GT_V = batches.get_batch(0).poisson.solve_poisson(GTT)
        source = source_batch[0]
        target = source_batch[1]

        pred_V, pred_J, pred_J_restricted = self.predict_map(source, target)

        # TODO: Get UV coordinates from pred_V and pass through ARAP counting loss 
        GT_V, GT_J, GT_J_restricted = self.get_gt_map(source, target)

        if UNIT_TEST_POISSON_SOLVE:
            success = self.check_map(source, target, GT_J, GT_V) < 0.0001
            print(self.check_map(source, target, GT_J, GT_V))
            assert(success), f"UNIT_TEST_POISSON_SOLVE FAILED!! {self.check_map(source, target, GT_J, GT_V)}"

        # Compute losses 
        # TODO: get tensor vertices, faces, and face areas 
        from losses import arap, count_loss, get_local_tris, meshArea2D
        vertices = source.get_vertices() 
        faces = torch.from_numpy(source.get_source_triangles()).long()
        local_tris = get_local_tris(vertices, faces, device=self.device)
        pred_UV = pred_V[0,:,:2]
        # Compute face areas for counting loss 
        # TODO: BATCH BELOW 
        fareas = meshArea2D(vertices, faces, return_fareas=True)
        arapenergy = arap(local_tris, faces, pred_UV,
                            device=self.device, renormalize=False,
                            return_face_energy=True, timeit=False)
        countloss = count_loss(arapenergy, fareas, return_softloss=False, device=self.device, 
                                threshold = 0.01)

        loss = countloss
        loss = loss.type_as(GT_V)
        val_loss = loss
        if self.verbose:
            print(
                f"batch of {target.get_vertices().shape[0]:d} source <--> target pairs, each mesh {target.get_vertices().shape[1]:d} vertices, {source.get_source_triangles().shape[1]:d} faces")
        ret = {
            "target_V": target.get_vertices().detach(),
            "source_V": source.get_vertices().detach(),
            "loss": loss,
            "val_loss": val_loss,
            "count_loss": countloss.detach(), 
            "pred_V": pred_V.detach(),
            "T": source.get_source_triangles(),
            'source_ind': source.source_ind,
            'target_inds': target.target_inds
        }

        if self.args.test:
            ret['pred_J_R'] = pred_J_restricted.detach()
            ret['target_J_R'] = GT_J_restricted.detach()
            
        return ret

    def validation_step_end(self, batch_parts):
        self.val_step_iter += 1
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
        if self.log_validate:
            self.log_validate = False
            tb.add_scalar("valid count loss", batch_parts["count_loss"].mean(), global_step=self.global_step)
            colors = self.colors(batch_parts["source_V"].cpu().numpy(), batch_parts["T"])
            # Replace here vertices and faces by something useful.

            tb.add_mesh("valid_predicted_mesh", vertices=batch_parts["pred_V"][0:1],
                        faces=numpy.expand_dims(batch_parts["T"], 0),
                        global_step=self.global_step, colors=colors)
            tb.add_mesh("valid_source_mesh", vertices=batch_parts["source_V"].unsqueeze(0),
                        faces=numpy.expand_dims(batch_parts["T"], 0),
                        global_step=self.global_step, colors=colors)
            tb.add_mesh("valid_target_mesh", vertices=batch_parts["target_V"][0:1],
                        faces=numpy.expand_dims(batch_parts["T"], 0),
                        global_step=self.global_step, colors=colors)
        # self.log('validation_loss', loss.item(), logger=True, prog_bar=True, on_step=True, on_epoch=True)
        if self.args.xp_type == "uv":
            if self.val_step_iter % 100 == 0:
                print("saving validation intermediary results.")
                for idx in range(len(batch_parts["pred_V"])):
                    path = Path(self.logger.log_dir) / f"valid_batchidx_{idx}.png"
                    plot_uv(path, batch_parts["pred_V"][idx].squeeze().detach(),
                            batch_parts["pred_V"][idx].squeeze().detach(), batch_parts["T"][idx].squeeze())

        # self.log('valid_loss', loss.item(), logger=True, prog_bar=True, on_step=True, on_epoch=True)
        return loss

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
        # self.log('test_loss', loss.item(), logger=True, prog_bar=True, on_step=True, on_epoch=True)
        if self.args.xp_type == "uv":
            if self.val_step_iter % 100 == 0:
                print("saving validation intermediary results.")
                for idx in range(len(batch_parts["pred_V"])):
                    path = Path(self.logger.log_dir) / f"valid_batchidx_{idx}.png"
                    plot_uv(path, batch_parts["target_V"][idx].squeeze().cpu().numpy(),
                            batch_parts["pred_V"][idx].squeeze().detach().cpu().numpy(), batch_parts["T"][idx].squeeze())

        # self.log('test_loss', loss.item(), logger=True, prog_bar=True, on_step=True, on_epoch=True)
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

    def validation_epoch_end(self, losses):
        self.val_step_iter = 0
        self.log_validate = True

    def colors(self, v, f):
        vv = igl.per_vertex_normals(v, f)
        vv = (numpy.abs(vv) + 1) / 2
        colors = vv * 255
        return torch.from_numpy(colors).unsqueeze(0)

    def training_step_end(self, batch_parts):
        # torch.cuda.synchronize()
        # This is called after each training_step, and stores the output of training_step in a list. This list can be longuer than 1 if training is distributed on multiple GPUs
        # this next few lines make sure cupy releases all memory
        loss = batch_parts["loss"].mean()
        if self.global_step % FREQUENCY == 0:  # skipping for now AttributeError: 'MyNet' object has no attribute 'colors'

            tb = self.logger.experiment
            colors = self.colors(batch_parts["source_V"].cpu().numpy(), batch_parts["T"])

            tb.add_mesh("training_predicted_mesh", vertices=batch_parts["pred_V"][0:1],
                        faces=numpy.expand_dims(batch_parts["T"], 0),
                        global_step=self.global_step, colors=colors)
            tb.add_mesh("training_source_mesh", vertices=batch_parts["source_V"].unsqueeze(0),
                        faces=numpy.expand_dims(batch_parts["T"], 0),
                        global_step=self.global_step, colors=colors)
            tb.add_mesh("training_target_mesh", vertices=batch_parts["target_V"][0:1],
                        faces=numpy.expand_dims(batch_parts["T"], 0),
                        global_step=self.global_step, colors=colors)
            # tb.add_mesh("source_samples", vertices = batch_parts["source_samples"][0].unsqueeze(0), global_step=self.global_step)
            tb.add_scalar("train count loss", batch_parts["count_loss"].mean().item(), global_step=self.global_step)
            tb.add_scalar("train loss", batch_parts["loss"].mean().item(), global_step=self.global_step)

        # self.log('train_loss', loss.item(), logger=True, prog_bar=True, on_step=True, on_epoch=True)
        if self.args.xp_type == "uv":
            if self.global_step % 100 == 0:
                print("saving training intermediary results.")
                if len(batch_parts["pred_V"]) == 1: 
                    path = os.path.join(self.logger.log_dir, f"train_{self.global_step}.png")                        
                    plot_uv(path, batch_parts["pred_V"].squeeze().detach().numpy(),
                            batch_parts["pred_V"].squeeze().detach().numpy(), batch_parts["T"].squeeze())
                else:
                    for idx in range(len(batch_parts["pred_V"])):
                        path = os.path.join(self.logger.log_dir, f"train_{self.global_step}_batch{idx}.png")                    
                        plot_uv(path, batch_parts["pred_V"][idx].squeeze().detach().numpy(),
                                batch_parts["pred_V"][idx].squeeze().detach().numpy(), batch_parts["T"][idx].squeeze())
        return loss

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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[self.args.lr_epoch_step[0],self.args.lr_epoch_step[1]], gamma=0.1),
            'name': 'scheduler'
            }
        return [optimizer], [lr_scheduler]


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


def main(gen, log_dir_name, args):
    seed_everything(48, workers=True)

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

    logger = TensorBoardLogger("lightning_logs", name=log_dir_name)

    if args.gpu_strategy:
        if os.name != 'nt':  # no support for windows because of gloo
            if args.gpu_strategy == 'ddp':
                plugins = pl.plugins.training_type.DDPPlugin(find_unused_parameters=False)
            elif args.gpu_strategy == 'ddp_spawn':
                plugins = pl.plugins.training_type.DDPSpawnPlugin(find_unused_parameters=False)
    #

    checkpoint_callback = ModelCheckpoint(every_n_train_steps=500)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    ################################ TRAINER #############################
    trainer = pl.Trainer(accelerator=has_gpu, devices=args.n_devices, precision=args.precision, log_every_n_steps=200, 
                         max_epochs=args.epochs, sync_batchnorm=args.n_devices != 1,
                         val_check_interval=(1.0 if args.no_validation else args.val_interval), logger=logger,
                         plugins=None,
                         accumulate_grad_batches=args.accumulate_grad_batches,
                         num_sanity_val_steps=2,
                         enable_model_summary=False,
                         enable_progress_bar=True,
                         num_nodes=1,
                         deterministic= args.deterministic,
                         strategy='ddp' if torch.cuda.is_available() else None,
                         callbacks=[checkpoint_callback,lr_monitor])
    ################################ TRAINER #############################

    if trainer.precision == 16:
        use_dtype = torch.half
    elif trainer.precision == 32:
        use_dtype = torch.float
    elif trainer.precision == 64:
        use_dtype = torch.double
    else:
        raise Exception("trainer's precision is unexpected value")

    print(len(train_pairs))
    train_dataset = DeformationDataset(train_pairs, gen.get_keys_to_load(True),
                                       gen.get_keys_to_load(False), use_dtype, train=True, args=args)
    print(len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=custom_collate, pin_memory = (args.unpin_memory is None),
                              shuffle=(args.test is None), num_workers=args.workers, persistent_workers=args.workers > 0)
    
    if args.no_validation or args.test:
        valid_loader = None
    else:
        valid_dataset = DeformationDataset( valid_pairs, gen.get_keys_to_load(True),
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
        trainer.fit(model, train_loader)
        return

    if args.test:
        print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& TEST &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        trainer.test(model, train_loader)
        return

    print(len(train_loader))
    trainer.fit(model, train_loader, valid_loader)
# ================ #
