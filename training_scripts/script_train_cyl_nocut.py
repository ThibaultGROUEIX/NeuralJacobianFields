import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'source_njf'))
from Experiment import Experiment
import args_from_cli

class UVExperiment(Experiment):
    def __init__(self, cpuonly=False):
        self.cpuonly = cpuonly
        Experiment.__init__(self, "Cylinder", 'Use NJF to predict UVs and supervise using distortion losses from DA Wand', cpuonly=cpuonly)

    def init_encoder(self, encoder, args):
        # TODO: might not need pointnet conditioning for generalizability -- depends on what kind of message passing rest of NJF does
        # source|target=True indicates that the source will ba passed to a PointNet encoder
        encoder.add_pointnet(1000, source=True, target=False)

if __name__ == '__main__':
    sys.argv.append("--root_dir_train")
    sys.argv.append("./data/cylinder_nocut")
    sys.argv.append("--root_dir_test")
    sys.argv.append("./data/cylinder_nocut")
    sys.argv.append("--expname")
    sys.argv.append("Cylinder_Nocut")

    sys.argv.append("--experiment_type")
    sys.argv.append("DEFAULT")
    sys.argv.append("--size_train")
    sys.argv.append("1")
    sys.argv.append("--size_test")
    sys.argv.append("1")
    sys.argv.append("--epochs")
    sys.argv.append("10000")
    sys.argv.append("--val_interval")
    sys.argv.append("20")
    sys.argv.append("--data_file")
    sys.argv.append("cylinder.json")
    # sys.argv.append("--overfit_one_batch")
    sys.argv.append("--align_2D")
    sys.argv.append("--xp_type")
    sys.argv.append("uv")
    sys.argv.append("--gpu_strategy")
    sys.argv.append("ddp")
    sys.argv.append("--n_gpu")
    sys.argv.append("1")

    # Initialization
    sys.argv.append("--identity")
    sys.argv.append("--init")
    sys.argv.append("tutte")

    # Loss settings
    sys.argv.append("--no_poisson")
    sys.argv.append("--lossdistortion")
    sys.argv.append("dirichlet")
    sys.argv.append("--lossedgeseparation")
    sys.argv.append("--seplossdelta")
    sys.argv.append("0.01")
    sys.argv.append("--seplossweight")
    sys.argv.append("1")

    sys.argv.append("--targets_per_batch")
    sys.argv.append("16")
    sys.argv.append("--accumulate_grad_batches")
    sys.argv.append("1")

    # sys.argv.append("--debug")

    exp = UVExperiment()

    #this parses the command line arguments and then trains on the given dataset with the given experiment
    args = args_from_cli.parse_args()
    exp.get_args_and_train(args)

#