import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'source_njf'))
from Experiment import Experiment

class UVExperiment(Experiment):
    def __init__(self, cpuonly=False):
        self.cpuonly = cpuonly
        Experiment.__init__(self, "UVExperiment", 'Use NJF to predict UVs and supervise using distortion losses from DA Wand', cpuonly=cpuonly)

    def init_encoder(self, encoder, args):
        # TODO: might not need pointnet conditioning for generalizability -- depends on what kind of message passing rest of NJF does 
        # source|target=True indicates that the source will ba passed to a PointNet encoder
        encoder.add_pointnet(1000, source=True, target=False)


if __name__ == '__main__':
    sys.argv.append("--root_dir_train")
    sys.argv.append("./data/cylinderbody")
    sys.argv.append("--root_dir_test")
    sys.argv.append("./data/cylinderbody")
  
    sys.argv.append("--experiment_type")
    sys.argv.append("DEFAULT")
    sys.argv.append("--size_train")
    sys.argv.append("1")
    sys.argv.append("--size_test")
    sys.argv.append("1")
    sys.argv.append("--epochs")
    sys.argv.append("8000")
    sys.argv.append("--data_file")
    sys.argv.append("cylindertest.json")
    sys.argv.append("--no_validation")
    # sys.argv.append("--overfit_one_batch")
    sys.argv.append("--align_2D")
    sys.argv.append("--xp_type")
    sys.argv.append("uv")
    sys.argv.append("--gpu_strategy")
    sys.argv.append("ddp")

    sys.argv.append("--targets_per_batch")
    sys.argv.append("16")
    sys.argv.append("--accumulate_grad_batches")
    sys.argv.append("1")

    exp = UVExperiment()
    
    #this parses the command line arguments and then trains on the given dataset with the given experiment
    exp.get_args_and_train()

# 