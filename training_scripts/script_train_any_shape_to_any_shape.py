import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'source_njf'))
from Experiment import Experiment


class DefaultExperiment(Experiment):
    def __init__(self, cpuonly=False):
        self.cpuonly = cpuonly
        Experiment.__init__(self, "DefaultExperiment", 'This experiement takes a folder of meshes as input and a list of pairs (source_mesh, target_mesh) in this folder (in pairs.json) and learns to deform the source into the target.', cpuonly=cpuonly)

    def init_encoder(self, encoder, args):
        # source|target=True indicates that the source will be passed to a PointNet encoder
        encoder.add_pointnet(1000, source=True, target=True)


if __name__ == '__main__':
    sys.argv.append("--root_dir_train")
    sys.argv.append("/home/groueix/neural_jacobian_fields_data_train")
    sys.argv.append("--root_dir_test")
    sys.argv.append("/home/groueix/neural_jacobian_fields_data")
    # sys.argv.append("--root_dir_train")
    # sys.argv.append("/home/groueix/db_ren_train_fly_2")
    # sys.argv.append("--root_dir_test")
    # sys.argv.append("/home/groueix/db_ren_test_fly_2")
    sys.argv.append("--workers")
    sys.argv.append("8")

  
    sys.argv.append("--dataset_fail_safe")
    sys.argv.append("--experiment_type")
    sys.argv.append("DEFAULT")

    # sys.argv.append("--no_wks")
    sys.argv.append("--targets_per_batch")
    sys.argv.append("16")
    sys.argv.append("--accumulate_grad_batches")
    sys.argv.append("1")
    sys.argv.append("--checkpoint")
    sys.argv.append("./data/checkpoints/morph_any_shape_into_any_other_withWKS.ckpt")
    sys.argv.append("--lr_epoch_step")
    sys.argv.append("1")
    sys.argv.append("2")
    exp = DefaultExperiment()
    #this parses the command line arguments and then trains on the given dataset with the given experiment
    exp.get_args_and_train()

# 