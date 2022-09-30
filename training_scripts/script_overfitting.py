import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'source_njf'))
from Experiment import Experiment


class RegisterTemplateExperiment(Experiment):

    def __init__(self, cpuonly=False):
        self.cpuonly = cpuonly
        Experiment.__init__(self, "register_template", 'This experiement takes a random mesh in a random pose and shape and deforms it to the tpose with the same shape.', cpuonly=cpuonly)

    def init_encoder(self, encoder, args):
        encoder.add_pointnet(1000, source=False, target=True)


if __name__ == '__main__':
    #this is a line that should always be run before training due to multiprocessing issues with pytorch
    # import multiprocessing
    # multiprocessing.set_start_method("spawn")
    #you should give the dataset link through the command line, but for the sake of the example I am setting it up automatically
    sys.argv.append("--root_dir_train")
    sys.argv.append("/sensei-fs/users/groueix/db_ren_train_fly/")
    sys.argv.append("--root_dir_test")
    sys.argv.append("/sensei-fs/users/groueix/db_ren_test_fly/")
    sys.argv.append("--size_train")
    sys.argv.append("100000")
    # sys.argv.append("--data_file")
    # sys.argv.append("registration.json")

    # sys.argv.append("/home/groueix/db_test_processed/")
    # sys.argv.append("/home/groueix/db_train_processed/")
    sys.argv.append("--workers")
    sys.argv.append("0")
    sys.argv.append("--overfit_one_batch")
    sys.argv.append("--lr_epoch_step")
    sys.argv.append("10000")
    sys.argv.append("11000")
    # # sys.argv.append("--no_vertex_loss")
    # sys.argv.append("--no_jacobian_loss")
    sys.argv.append("--lr")
    sys.argv.append("0.001")
    sys.argv.append("--only_centroids_and_normals")
    sys.argv.append("--no_pointnet_wks")
    sys.argv.append("--random_scale")
    sys.argv.append("none")
    # sys.argv.append("--random_centering")
    sys.argv.append("--vertex_loss_weight")
    sys.argv.append("1")
    sys.argv.append("--precision")
    sys.argv.append("64")
    sys.argv.append("--compute_human_data_on_the_fly")
    sys.argv.append("--experiment_type")
    sys.argv.append("REGISTER_TEMPLATE")
    sys.argv.append("--n_gpu")
    sys.argv.append("1")
    sys.argv.append("--targets_per_batch")
    sys.argv.append("1")
    sys.argv.append("--accumulate_grad_batches")
    sys.argv.append("1")
    sys.argv.append("--layer_normalization")
    sys.argv.append("LAYERNORM")
    # sys.argv.append("LAYERNORM")
    # sys.argv.append("--deterministic")
    sys.argv.append("--checkpoint")
    # sys.argv.append("lightning_logs/register_template/version_8/checkpoints/epoch=32-step=204500.ckpt") #WKS
    sys.argv.append("checkpoints/register_template_withoutWKS.ckpt") #GROUPNORM
    # sys.argv.append("--gpu_strategy")
    # sys.argv.append("ddp")
    # sys.argv.append("/sensei-fs/users/groueix/db_test_processed/")
    #setup our experiment
    exp = RegisterTemplateExperiment()
    #this parses the command line arguments and then trains on the given dataset with the given experiment
    exp.get_args_and_train()

# 