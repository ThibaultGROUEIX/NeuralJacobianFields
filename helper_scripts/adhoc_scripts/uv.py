import sys
import igl
import torch
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from DeformationEncoder import DeformationEncoder
from Experiment import Experiment
import time
import numpy as np

class UVExperiemnt(Experiment):

    def __init__(self):
        Experiment.__init__(self, "Faust", 'just testing this thing out')

    def init_encoder(self, encoder: DeformationEncoder, args):
        encoder.add_pointnet(1000, True, False)
        # encoder.add_loader(False, "handles")

if __name__ == '__main__':
    # this is a line that should always be run before training due to multiprocessing issues with pytorch
    import multiprocessing

    multiprocessing.set_start_method("spawn")
    USE_PREPROCESSED = False  # set this to false if you want to run inference on raw obj's/off's
    # setup our experiment
    exp = UVExperiemnt()
    # load a checkpoint (you should train to get one)
    exp.load_network("checkpoints/thinghy10k_UV.ckpt")

    for i in range(180000, 200002, 2):
    # define one source and a set of targets -- these could be data that's been processd or raw obj's/off's
        if USE_PREPROCESSED:
            source = f"/home/groueix/neural_jacobian_fields/diffusion-net/experiments/uv/data/10k/{str(i).zfill(8)}.obj"
            targets = [f"/home/groueix/neural_jacobian_fields/diffusion-net/experiments/uv/data/10k/{str(i+1).zfill(8)}.obj"]
        else:
            source = f"/home/groueix/neural_jacobian_fields/diffusion-net/experiments/uv/data/10k/{str(i).zfill(8)}.obj"
            targets = [f"/home/groueix/neural_jacobian_fields/diffusion-net/experiments/uv/data/10k/{str(i+1).zfill(8)}.obj"]
        # run the loaded net on the given source and targets. Output is a batch of maps b x V x 3, and a batch of
        # jacobians b x T x 3 x 3, where b = number_of_targets
        with torch.no_grad():
            batch_of_maps, batch_of_jacobians, source, target = exp.evaluate_on_source_and_targets(source, targets)
            igl.write_obj(f"/home/groueix/neural_jacobian_fields/diffusion-net/experiments/uv/test_results_quantitative_NJF/{i}_NJF.obj",  batch_of_maps.cpu().numpy()[0], source.get_source_triangles())
            np.save(f"/home/groueix/neural_jacobian_fields/diffusion-net/experiments/uv/test_results_quantitative_NJF/{i}_pred.npy",  batch_of_maps.cpu().numpy()[0])
        a = 0
