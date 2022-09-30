import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'source_njf'))
from Experiment import Experiment


class DefaultExperiment(Experiment):
    def __init__(self, cpuonly=False):
        self.cpuonly = cpuonly
        Experiment.__init__(self, "DefaultExperiment", 'This experiement takes a folder of meshes as input and a list of pairs (source_mesh, target_mesh) in this folder (in pairs.json) and learns to deform the source into the target.', cpuonly=cpuonly)

    def init_encoder(self, encoder, args):
        # source|target=True indicates that the source will ba passed to a PointNet encoder
        encoder.add_pointnet(1000, source=True, target=True)


if __name__ == '__main__':
    sys.argv.append("--root_dir_train")
    sys.argv.append("./data/example_training_directory")
    sys.argv.append("--root_dir_test")
    sys.argv.append("./data/example_training_directory")

  
    sys.argv.append("--experiment_type")
    sys.argv.append("DEFAULT")

    sys.argv.append("--targets_per_batch")
    sys.argv.append("16")
    sys.argv.append("--accumulate_grad_batches")
    sys.argv.append("1")

    exp = DefaultExperiment()
    #this parses the command line arguments and then trains on the given dataset with the given experiment
    exp.get_args_and_train()

# 