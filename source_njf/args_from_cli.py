import argparse
import json
import os
import unittest
import warnings

import SourceToTargetsFile
from train_loop import main
from tqdm import tqdm
import sys
from easydict import EasyDict

def get_arg_parser():
	"""
	:return: arg parser with all relevant parametrs
	"""
	parser = argparse.ArgumentParser()

	parser.add_argument("--projectname",
						help='name of project (saved in wandb)',
						type = str, default='njfwand')
	parser.add_argument("--expname",
						help='name of experiment folder',
						type = str, required=True)
	parser.add_argument("--root_dir_train",
						help='location of the root dir to get data from',
						type = str,default=None)
	parser.add_argument("--root_dir_test",
						help='location of the root dir to get data from',
						type = str,default=None)
	parser.add_argument("--version",
						help='version no',
						type = int, default=0)
	parser.add_argument("--outputdir",
						help='output directory',
						type = str, default='outputs')

	### ARCH
	parser.add_argument("--fft",
							help='fourier features',
							action="store_true")
	parser.add_argument("--fft_dim", type=int,
							help='fourier features dimension',
							default=256)
	parser.add_argument("--noencoder", help="no encoder. TURN THIS ON IF YOU DONT WANT TO TRAIN THE ENCODER",action="store_true")

	### TRAINING
	parser.add_argument("--continuetrain", help='continue training', action="store_true")
	parser.add_argument("--split_train_set", help="split the train set to create a test set",
						action="store_true")
	parser.add_argument("--train_percentage",help = "the train/test split in percentage, default is 90",default=90,type=int)
	parser.add_argument("--val_interval",help = "validation interval",default=None,type=int)
	parser.add_argument("--valrenderratio",help = "ratio of samples to do validation renders for",default=1,type=float)
	parser.add_argument("--gradclip",help = "gradient clipping",default=0,type=float)
	parser.add_argument("--epochs",help = "number of training epochs",default=2000,type=int)
	parser.add_argument("--optimizer",choices={"adam", "sgd"}, help='type of optimizer', type = str,default="adam")
	parser.add_argument("--identity", help='initialize network from identity', action="store_true")
	parser.add_argument("--globaltrans", help='also predict global translation per shape code', action="store_true")
	parser.add_argument("--init", choices={"tutte", "isometric"}, help="initialize 2D embedding", default=None, type=str)
	parser.add_argument("--ninit", type=int, default=1, help="re-initialize this mesh n many times. only valid for isometric initialization. -1 indicates new initialization for EACH load")
	parser.add_argument("--basistype", choices={"basis", "rot", "global"}, help="how to sample new triangle local basis", default="basis", type=str)
	parser.add_argument("--initjinput", help="use the initialization jacobian as part of input",
							action="store_true")
	parser.add_argument("--no_vertex_loss", help="use source/target vertex l2 loss",
						action="store_true")
	parser.add_argument("--no_poisson", help="no poisson solve", action="store_true")
	parser.add_argument("--no_jacobian_loss", help="use source/target jacobian l2 loss", action="store_true")
	parser.add_argument("--targets_per_batch",
						help=f"maximal number of target deformations per batch, default is 16",type=int,default=1)
	parser.add_argument("--sources_per_batch",
						help=f"maximal number of sources (each with its own deformation set) per batch, default is 1", type=int, default=1)
	parser.add_argument( "--workers",
						help=f"number of worker threads for dataloading, default is 8",
						type=int, default=8)
	parser.add_argument('--two_dimensional_target',
						help='If specified, restrict target prediction to 2D', action="store_true")
	parser.add_argument("--align_2D",
						help='If specified, align target and source using procrustes over xy axis', action="store_true")
	parser.add_argument("--data_file",help='if specified, use given path to load json file holding all pairs (relative to data dir''s location)',
						type = str,default="data.json")
	parser.add_argument("--only_numbers",
						help='if specified, look for directories that are just number files',
						action="store_true")
	parser.add_argument("--checkpoint",help = "if specified, start training from given checkpoint", type=str)
	parser.add_argument("--unpin_memory",
						help='if specified, don''t pin memory',
							action="store_true")
	parser.add_argument('--no_pointnet_normals',help='if specified, don''t give pointnet normals of points',
							action="store_true")
	parser.add_argument('--no_wks', help='if specified, don''t give pointnet wks',
						action="store_true")
	parser.add_argument( "--top_k_eig",
							help=f"number of eigenvalues to compute for wks ",
							type=int, default=50)
	parser.add_argument("--lr", help="learning rate, default is 1e-4", default=1e-4,
						type=float)
	parser.add_argument("--normalize_jac_loss", help="normalize jacobians 1/norm(GT) when comparing them, default is false",action="store_true")
	parser.add_argument("--precision",help ="the precision we work in, should be 16,32,or 64 (default)",default=64,type=int)
	parser.add_argument("--vertex_loss_weight", help="the weight to place on the vertex loss (jacobian loss is unweighted) default = 1.0", default=1.0, type=float)

 	###### POSTPROCESS ######
	parser.add_argument("--opttrans", help = "predict l0 translation and visualize", action="store_true")
	###### LOSSES ######
	parser.add_argument("--lossdistortion", help = "choice of distortion loss", default=None, type=str, choices={'arap', 'dirichlet', 'edge'})
	parser.add_argument("--losscount", help = "use counting loss over distortion energy", action="store_true")
	parser.add_argument("--lossgradientstitching", choices={'l2', 'split'}, help = "use gradient stitching loss", default=None)
	parser.add_argument("--cuteps", help="epsilon for edge stitching post-process", default=1e-1, type=float)

	parser.add_argument("--stitchlossweight", help="weight for edge stitching losses", default=1, type=float)
	parser.add_argument("--stitchlossweight_min", help="max weight for edge stitching losses", default=0, type=float)
	parser.add_argument("--stitchlossweight_max", help="max weight for edge stitching losses", default=1, type=float)
	parser.add_argument("--stitchloss_schedule", type=str, choices={'linear', 'cosine'}, help = "type of schedule for edgeloss", default=None)
	parser.add_argument("--stitchrelax", help = "l0 relaxation on stitching loss", action="store_true")

	parser.add_argument("--lossedgeseparation", help = "use edge separation loss", action="store_true")
	parser.add_argument("--eseploss", type=str, choices={'l1', 'l2'}, help = "type of regression loss for edge sep", default="l1")
	parser.add_argument("--seplossdelta", help="initial delta for edge separation loss", default=0.5, type=float)
	parser.add_argument("--seplossdelta_min", help="min delta for edge separation loss", default=0.01, type=float)
	parser.add_argument("--seploss_schedule", help="apply linear schedule to the separation loss delta parameter", action="store_true")
	parser.add_argument("--lossautocut", help = "use counting loss over distortion energy", action="store_true")

	parser.add_argument("--gpu_strategy",help ="default: no op (whatever lightning does by default). ddp: use the ddp multigpu startegy; spawn: use the ddpspawn strategy",default=None,choices={'ddp','ddpspawn', 'cpuonly'})
	parser.add_argument("--no_validation",help = "skip validation",action="store_true")
	parser.add_argument("--n_gpu", help="num of gpus, default is all", type=int, default=-1)
	parser.add_argument("--n_devices", help="more general version of n_gpu, defaults to cpu", type=int, default=1)
	parser.add_argument("--pointnet_layer_normalization",help = "type of normalization for the PointNet encoder's layers",default="GROUPNORM",type=str,choices={'GROUPNORM','BATCHNORM','IDENTITY'})
	parser.add_argument("--layer_normalization",help = "type of normalization for the decoder's layers",default="GROUPNORM",type=str,choices={'GROUPNORM','GROUPNORM2', 'BATCHNORM','IDENTITY', 'LAYERNORM'})
	parser.add_argument("--dense",help = "type of dense MLP to use",default=None,type=str,choices={'random', 'input', 'xyz'})
	parser.add_argument("--overfit_one_batch", help="overfit a particular batch of the training data",action="store_true")
	parser.add_argument("--xp_type", help ="only runs the validation",default=None,type=str)
	parser.add_argument("--test", help="run in test mode", action="store_true")
	parser.add_argument("--statsonly", help="applies only to test time, only write stats, no viz", action="store_true")
	parser.add_argument("--random_scale",help="randomly scale the source (source), target (target), both each with their own random scale (both), both with "
												"the same scale each time (same), or none (none = default)",type=str,default='none',choices={'source','target','both','same','none'})
	parser.add_argument("--test_set",help="which set to run test on [valid],train,all",type=str,default='valid',choices={'valid','train','all'})
	parser.add_argument('--only_final_stats',help= "during test time, only aggregate the final stats", action="store_true")
	parser.add_argument('--checkpoint_often',help="save checkpoint more often (every 1000 steps)",action='store_true')
	parser.add_argument('--random_centering',help="random centering of train samples",action='store_true')
	parser.add_argument('--deterministic',help="sets deterministic == true in the trainer. Some operations will be slower on the GPU",action='store_true')

	parser.add_argument('--compute_human_data_on_the_fly',help="computes human data on the fly",action='store_true')
	parser.add_argument('--dataset_fail_safe',help="returns first pairs in the dataset if it encounters a data issue",action='store_true')
	parser.add_argument('--size_train',help="size_train_dataset",type=int, default=1000)
	parser.add_argument('--size_test',help="size_test_dataset",type=int, default=1000)
	parser.add_argument('--experiment_type',help="size_test_dataset",type=str, choices={"TPOSE","REGISTER_TEMPLATE","DEFAULT"}, default="DEFAULT")
	parser.add_argument('--store_faces_per_sample',help="store_faces_for_each samples individually",action='store_true')
	parser.add_argument('--lr_epoch_step', nargs="+", type=int,help="reduce learning rate once within range and once after",default=[30,40])
	parser.add_argument('--accumulate_grad_batches', type=int,help="accumulate gradients acrosss multiple batches",default=1)
	parser.add_argument('--shuffle_triangles',type=bool,help="Shuffle triangles before NJF decoder, to avoid that group-norm overfits to a particular triangulation.",default=1)

	## DEBUGGING
	parser.add_argument('--mem', action='store_true')
	parser.add_argument('--debug', action='store_true')
	parser.add_argument('--overwrite', action='store_true')
	parser.add_argument('--overwritecache', action='store_true', help='overwrite stored mesh operators')

	return parser


def parse_args():
	"""
	parse command line args and return them
	:return: the args in stadnard arg parser's args format
	"""
	parser = get_arg_parser()
	args = parser.parse_args()
	def wrap(a):
		class WrapperForArgsToPreventAddingArgsThatDoNotExist:

			def __setattr__(self, key, value):
				if not hasattr(a, key):
					raise TypeError(f"you are trying to put a field ({key}) that doesn't exist in args. "
									"This is probably a typo on your end and you want to put a valid field there.")
				warnings.warn(f"Notice that you are setting the command line arg {key} = {value} programmatically (probably in your main)")
				a.__setattr__( key, value)
			def __getattribute__(self, item):
				return a.__getattribute__(item)
			def __hasattr__(self,key):
				return a.__hasattr__(key)
			def __str__(self):
				return a.__str__()
		return  WrapperForArgsToPreventAddingArgsThatDoNotExist()
	args =  EasyDict(wrap(args).__dict__)

	if args.root_dir_test is None:
		# If there is not test directory, the train directory is split.
		print(f"ARGS : Setting split_train_set to {True}. The default was {args.split_train_set}")
		print(f"ARGS : Setting root_dir_test to {args.root_dir_train}. The default was {args.root_dir_test}")
		args.split_train_set = True
		args.root_dir_test = args.root_dir_train

	return args





def get_default_args():
	"""
	:return: args object as without actually using the command line (all vals set to default)
	"""
	with unittest.mock.patch('sys.argv',['dummy_args']):#we do not use any real CLI arguments
		return parse_args()

