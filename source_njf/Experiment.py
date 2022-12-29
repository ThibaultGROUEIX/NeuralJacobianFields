import warnings
from abc import ABC, abstractmethod

import args_from_cli
from train_loop import load_network_from_checkpoint, MyNet
import torch
from DeformationEncoder import DeformationEncoder
from DeformationDataset import DeformationDataset

import os
import json
from train_loop import main

class Experiment(ABC):
    '''
    base class for experiments
    '''

    def __init__(self,name,description, cpuonly=False):
        self.net = None
        self.name = name
        self.description = description
        self.cpuonly = cpuonly

    def modify_args(self,args):
        '''
        called before setting args, to enable modifying args
        :param args: the original args
        :return: the args, with any modification that should be included
        '''
        return args

    def get_encoder(self,args):
        '''
        initialize the encoder for this experiment and return it
        :param args: the cli args
        :return: the encoder, initialize
        '''
        args = self.modify_args(args)
        encoder = DeformationEncoder(args)
        self.init_encoder(encoder, args)
        return encoder

    @abstractmethod
    def init_encoder(self,encoder,args):
        '''
        abstract method that should be overridden to init the encoder object
        :return: DeformationEncoder object
        '''
        pass

    def load_network(self,checkpoint_path):
        '''
        load a network from a checkpoint, and store it internally
        :param checkpoint_path: path to checkpoint
        :return: the network loaded from the checkpoint
        '''
        self.net = load_network_from_checkpoint(checkpoint_path, cpuonly=self.cpuonly)
        if not self.cpuonly:
           self.net.cuda(0)
        else:
            self.net.cpu()
        self.__args = self.net.args
        return self.net


    def evaluate_on_source_and_targets_preprocess(self,source,targets = [None],args_to_overwrite = None, cpuonly=False):
        '''
        assuming a network's checkpoint has been loaded via load_network, run this experiment with the loaded network
        :param sources_to_targets:
        :return:
        '''
        assert self.net is not None, "call load_network() to load a checkpoint before trying to evaluate"
        args = args_from_cli.get_default_args()
        if args_to_overwrite is not None:
            for key, value in args_to_overwrite.items():
                print(f"ARGS : Setting {key} to {value}. The default was {args[key]}")
                args[key] = value
        if args_to_overwrite is not None:
            for key, value in args_to_overwrite.items():
                # print(f"ARGS : Setting {key} to {value}. The default was {args[key]}")
                setattr(self.__args, key, value)
        for key, value in args.items():
            if key not in self.__args:
                # print(f"ARGS : Setting {key} to {value}. The default was {args[key]}")
                setattr(self.__args, key, value)

        e = self.get_encoder(args)
        dataset = DeformationDataset.dataset_from_files(source,targets, e.get_keys_to_load(True), e.get_keys_to_load(False),
                                                        torch.double, args, cpuonly=cpuonly)

        assert(len(dataset)==1), "code it if you want a bigguer dataset"
        source, target = dataset[0]
        if not cpuonly:
            source.to(self.net.device)
            if target is not None:
                target.to(self.net.device)
        # self.source = source
        # self.target = target
        return source,target

    def evaluate_on_source_and_targets_inference(self,source,targets,args = None, cpuonly=False):
        # Forward
        maps = []
        jacobians = []
        pred_V,pred_J,_ = self.net.predict_map(source,targets)
        maps.extend(pred_V)
        jacobians.extend(pred_J)
        return torch.stack(maps,dim = 0),torch.stack(jacobians, dim = 0), source, targets
        
    def evaluate_on_source_and_targets(self,source,targets,args=None,cpuonly=False):
        source,targets = self.evaluate_on_source_and_targets_preprocess(source,targets,args,cpuonly)
        return self.evaluate_on_source_and_targets_inference(source,targets,args,cpuonly)

    def get_args_and_train(self):
        if self.net is not None:
            warnings.warn("seems like you loaded a network, but are now running training -- FYI, the loaded network is not being used in training (you need to specify the checkpoint in CLI")
        args = args_from_cli.parse_args()
        if args.compute_human_data_on_the_fly:
            # otherwize there is an issue in the workers. They can't all initialize CUDA.
            try:
                import multiprocessing
                multiprocessing.set_start_method("spawn")
                import torch
                torch.multiprocessing.set_start_method('spawn')
            except:
                print("Failed to initialize muttiprocessing but keep going.")
        args = self.modify_args(args)
        self.args = args
        print(f"starting training with args: {args}")


        if args.checkpoint is None:
            gen = self.get_encoder(args)
        else:
            print(f'************************** STARTING TRAINING FROM CHECKPOINT {args.checkpoint}' )
            gen = args.checkpoint
        
        if args.test:
            name = os.path.join(name,'test')

        main(gen, self.name, args)

    # def create_network(encoder, dataset):
    #     return MyNet(encoder, encoder.get_code_length(dataset), point_dim=dataset.get_point_dim())
