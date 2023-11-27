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
	parser.add_argument('--arch', type=str, choices={'diffusionnet', 'meshcnn', 'mlp'}, help="architecture to use", default='mlp')
	parser.add_argument('--softpoisson', type=str, choices={'edges', 'valid'}, help="SOFT POISSON", default=None)
	parser.add_argument('--sparsepoisson', action="store_true")
	parser.add_argument("--spweight", choices={'sigmoid', 'seamless', 'cosine', 'softmax'}, type=str,
						help = "how to map dot product to soft poisson weights", default='sigmoid')
	parser.add_argument("--softmax", help="softmax the predicted weights",action="store_true")
	parser.add_argument("--fft", type=int,
							default=0)
	parser.add_argument("--fftscale", type=int,
							help='fft scale',
							default=10)
	parser.add_argument("--vertexdim", type=int,
								help='vertex latent dimension',
								default=32)
	parser.add_argument("--facedim", type=int,
								help='face latent dimension',
								default=3)
	parser.add_argument("--noencoder", help="no encoder. TURN THIS ON IF YOU DONT WANT TO TRAIN THE ENCODER",action="store_true")

	### TRAINING
	parser.add_argument("--continuetrain", help='continue training', action="store_true")
	parser.add_argument("--checkpointdir", help='directory to ckpt if continue train', type = str, default=None)
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
	parser.add_argument("--slimiters", type=int, default=500, help="number of iterations to optimize SLIM")
	parser.add_argument("--init", choices={"tutte", "isometric", "slim"}, help="initialize 2D embedding", default=None, type=str)
	parser.add_argument("--ninit", type=int, default=1, help="re-initialize this mesh n many times. only valid for isometric initialization. -1 indicates new initialization for EACH load")
	parser.add_argument("--basistype", choices={"basis", "rot", "global"}, help="how to sample new triangle local basis", default=None, type=str)
	parser.add_argument("--initjinput", help="use the initialization jacobian as part of input",
							action="store_true")
	parser.add_argument("--initweightinput", help="use the initialization weights as part of input",
							action="store_true")
	parser.add_argument("--optweight", help="optimize the weights instead of getting them from network",
							action="store_true")

	parser.add_argument("--simplecut", help="enforce single boundary cut", action="store_true")
	parser.add_argument("--min_cuts", type=int, help="minimum # cuts for tutte init", default=0)
	parser.add_argument("--max_cuts", type=int, help="maximum # cuts for tutte init", default=0)

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
	parser.add_argument("--hardpoisson", type=str, choices={'loss', 'weight'}, help = "cutting options for hard poisson",
						default=None)
	parser.add_argument("--cuteps", help="epsilon for edge stitching post-process", default=1e-2, type=float)
	parser.add_argument("--weightcuteps", help="epsilon for edge stitching post-process (for pred weights)", default=1e-2, type=float)


	parser.add_argument("--opttrans", help = "predict l0 translation and visualize", action="store_true")

	###### LOSSES ######
	## NEW Stitching loss framework
	parser.add_argument("--stitchingloss", help = "choice of stitching losses. can use multiple", default=None,
						type = str, nargs='+',
						choices={'vertexseploss', 'edgecutloss', 'edgegradloss'})
	parser.add_argument("--stitchdist", help = "type of stitching distance metric", default='l1', type=str,
						choices={'l1', 'l2'})
	parser.add_argument("--stitchweight", choices={'stitchloss', 'softweight', 'softweightdetach'},
						help = "iterative reweighting of the stitching loss", default=None)
	parser.add_argument("--ignorei", help = "cone cutting experiment", default=0,
						type = int)
	parser.add_argument("--gtuvloss", help="use ground truth uv supervision", action="store_true")
	parser.add_argument("--gtjloss", help="ground truth jacobian loss", action="store_true")
	parser.add_argument("--removecutfromloss", action="store_true")

	# Seamless
	parser.add_argument("--seamlessvertexsep", help = "use counting loss over distortion energy", action="store_true")
	parser.add_argument("--seamlessedgecut", help = "use counting loss over distortion energy", action="store_true")
	parser.add_argument("--seamlessgradloss", help = "use counting loss over distortion energy", action="store_true")
	parser.add_argument("--seamlessdelta", help="initial delta for edge separation loss", default=0.0005, type=float)

	# Weights
	parser.add_argument("--stitchschedule", help = "apply linear schedule on stitching loss", type=str,
                     choices={'linear', 'cosine', 'constant'}, default=None)
	parser.add_argument("--stitchschedule_constant", help="% threshold to start enforcing stitch", default=0, type=float)
	parser.add_argument("--edgegrad_weight", help="loss weight", default=1, type=float)
	parser.add_argument("--edgecut_weight", help="loss weight", default=1, type=float)
	parser.add_argument("--edgecut_weight_max", help="max edge cut weight", default=1, type=float)
	parser.add_argument("--edgecut_weight_min", help="min edge cut weight", default=0, type=float)
	parser.add_argument("--vertexsep_weight", help="loss weight", default=1, type=float)
	parser.add_argument("--distortion_weight", help = "distortion weight", default=1, type=float)
	parser.add_argument("--sparsecuts_weight", help = "sparse cuts weight", default=1, type=float)
	parser.add_argument("--sparselossweight_max", help = "sparse cuts weight", default=1, type=float)
	parser.add_argument("--sparselossweight_min", help = "sparse cuts weight", default=1, type=float)
	parser.add_argument("--sparse_schedule", choices={'linear', 'cosine'}, default=None, type=str)
	parser.add_argument("--sparse_cosine_steps", default=1000, type=int)

	# Inverse jacobian loss
	parser.add_argument("--invjloss", help="penalize jacobians which shrink", action="store_true")

	# Sparse cuts loss
	parser.add_argument("--sparsecutsloss", help = "use sparse cuts loss", action="store_true")

	## Distortion loss
	parser.add_argument("--lossdistortion", help = "choice of distortion loss", default=None, type=str, choices={'arap', 'dirichlet', 'edge'})
	parser.add_argument("--losscount", help = "counting loss", action="store_true")
	parser.add_argument("--arapnorm", help = "normalize arap using avg edge len", action="store_true")

	## ELSE
	parser.add_argument("--lossgradientstitching", choices={'cosine', 'split', 'l2', 'l1'}, help = "use gradient stitching loss", default=None)

	parser.add_argument("--stitchlossweight", help="weight for edge stitching losses", default=1, type=float)
	parser.add_argument("--stitchlossweight_min", help="max weight for edge stitching losses", default=0, type=float)
	parser.add_argument("--stitchlossweight_max", help="max weight for edge stitching losses", default=1, type=float)
	parser.add_argument("--stitchloss_schedule", type=str, choices={'linear', 'cosine'}, help = "type of schedule for edgeloss", default=None)
	parser.add_argument("--stitchrelax", help = "l0 relaxation on stitching loss", action="store_true")

	parser.add_argument("--lossedgeseparation", help = "use edge separation loss", action="store_true")
	parser.add_argument("--eseploss", type=str, choices={'l1', 'l2'}, help = "type of regression loss for edge sep", default="l1")

	# NOTE: DEPRECATED
	parser.add_argument("--seplossdelta", help="initial delta for edge separation loss", default=0.0005, type=float)
	parser.add_argument("--seplossdelta_min", help="min delta for edge separation loss", default=0.01, type=float)
	parser.add_argument("--seploss_schedule", help="apply linear schedule to the separation loss delta parameter", action="store_true")
	parser.add_argument("--lossautocut", help = "use counting loss over distortion energy", action="store_true")
	######

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

