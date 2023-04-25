import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'source_njf'))
from Experiment import Experiment
import args_from_cli

class UVExperiment(Experiment):
    def __init__(self, name, cpuonly=False):
        self.cpuonly = cpuonly
        Experiment.__init__(self, name, 'Use NJF to predict UVs and supervise using distortion losses from DA Wand', cpuonly=cpuonly)

    def init_encoder(self, encoder, args):
        # TODO: check whether this samples different points every time
        encoder.add_pointnet(1000, source=True, target=False)

if __name__ == '__main__':
    # Set wandb
    # NOTE: WANDB_DIR MUST EXIST
    os.environ['WANDB_DIR'] = "/net/scratch/rliu/.cache/wandb"
    os.environ['WANDB_CACHE_DIR'] = "/net/scratch/rliu/.cache/wandb"
    os.environ['WANDB_DATA_DIR'] = "/net/scratch/rliu/.cache/wandb"
    #this parses the command line arguments and then trains on the given dataset with the given experiment
    args = args_from_cli.parse_args()

    exp = UVExperiment(args.expname)
    exp.get_args_and_train(args)

#