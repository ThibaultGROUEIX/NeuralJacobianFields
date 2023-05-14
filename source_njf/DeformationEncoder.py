import torch
import SourceMesh
from models import Encoder as Encoder

class PartialDeformationEncoderBase(torch.nn.Module):
    '''
    Abstract class with API for encoders
    '''
    def __init__(self):
        super(PartialDeformationEncoderBase, self).__init__()
    def to_load(self):
        '''
        :return: string, name of field to be loaded for the batch, e.g., if the returned value is "pie" then the data loader
        wiil attempt to load the file "pie.npy" for each example in the batch. This field will be then fed as an argument
        to the to_code function
        '''
        raise NotImplementedError
    def encode(self, loaded):
        '''
        Return the code for the given example (represented by "loaded")
        :param loaded: the content of the loaded field requested in to_load()
        :return: the code for the loaded example
        '''
        raise NotImplementedError

class _LoadedDataEncoder(PartialDeformationEncoderBase):
    '''
    Loads a given key_name as an npy file, and retuns it stacked as a 1D vector
    '''
    def __init__(self,key_name):
        super(_LoadedDataEncoder, self).__init__()
        self.__key_name = key_name
    def to_load(self):
        return self.__key_name
    def encode(self, loaded):
        return loaded

class _PointNetEncoder(PartialDeformationEncoderBase):
    '''
    encodes the "samples.npy" field via a pointnet
    '''

    def __init__(self,output_dim,normalization,use_normals = True,use_wks = True):
        super(_PointNetEncoder, self).__init__()
        self.__use_normals = use_normals
        self.__use_wks = use_wks
        self.__pointnet = Encoder(output_dim,self.__get_in_dim(), normalization=normalization)

    def __get_in_dim(self):
        in_dim = 3
        if self.__use_normals:
            in_dim += 3
        if self.__use_wks:
            in_dim += SourceMesh.WKS_DIM
        return in_dim
    def to_load(self):
        load_list = ["samples"]
        if self.__use_normals:
            load_list.append("samples_normals")
        if self.__use_wks:
            load_list.append("samples_wks")
        return load_list
    def encode(self, loaded):
        if self.__use_wks:
            # assert loaded[-1].shape[-1] == SourceMesh.WKS_DIM
            loaded[-1] = loaded[-1] * SourceMesh.WKS_FACTOR
        r = torch.cat(loaded,dim = 2)
        # HACK_FOR_ONE_ELEMENT_BATCH = loaded.shape[0] == 1
        # if HACK_FOR_ONE_ELEMENT_BATCH:
        #     r = loaded.repeat(2,1,1)
        # else:
        # r = loaded
        r = self.__pointnet(r)
        # if HACK_FOR_ONE_ELEMENT_BATCH:
        #     r = r[0,...].unsqueeze(0)
        return r

class DeformationEncoder(torch.nn.Module):
    '''
    Class for encoding deformations
    '''
    SOURCE = True
    TARGET = False
    def __init__(self,args):
        super(DeformationEncoder, self).__init__()
        self.__keys_to_load = {DeformationEncoder.SOURCE:{},DeformationEncoder.TARGET:{}}
        self.__generators = {DeformationEncoder.SOURCE:[],DeformationEncoder.TARGET:[]}
        self.__module_list = torch.nn.ModuleList()
        self.__ttype = None
        self.__args = args

    def __add_generator(self, source_or_target:bool, generator:PartialDeformationEncoderBase):
        '''
        add the given encoder
        :param source_or_target: add it to encode the source, or the target?
        :param generator: the encoder
        '''
        keys = generator.to_load()
        if isinstance(keys,str):
            keys = [keys]
        for key in keys:
            self.__keys_to_load[source_or_target][key] = True
        self.__generators[source_or_target].append(generator)
        self.__module_list.append(generator)

    def add_pointnet(self,code_length:int,source :bool,target:bool):
        '''
        Add a pointnet encoder
        :param code_length: the desired code length of PN's output
        :param source: true/false -- apply the PN to the source mesh?
        :param target: true/false -- apply the PN to the target mesh?
        '''
        # if self.__args:
        #     if self.__args.no_wks:
        #         print("not using wks for pointnet")
        #     use_normals = (not self.__args.no_pointnet_normals) and use_normals
        #     use_normals = (not self.__args.no_wks) and use_normals
        #     encoder = _PointNetEncoder(code_length,use_normals = not self.__args.no_pointnet_normals,use_wks  = not self.__args.no_wks,  normalization=normalization)
        # else:
        encoder = _PointNetEncoder(code_length, normalization=self.__args.pointnet_layer_normalization, use_wks = not self.__args.no_wks, use_normals= not self.__args.no_pointnet_normals)

        # TODO: ask Noam if freezing the encoder is correct...
        encoder.requires_grad = False
        encoder.eval()

        encoder._PointNetEncoder__pointnet.requires_grad = False
        encoder._PointNetEncoder__pointnet.eval()

        for param in encoder._PointNetEncoder__pointnet.parameters():
            param.requires_grad = False

        if self.__ttype is not None:
            encoder.type(self.type())
        to_add = []
        if source:
            to_add.append(True)
        if target:
            to_add.append(False)
        assert(len(to_add)>0)
        for val in to_add:
            self.__add_generator(val, encoder)

    def add_loader(self,source_or_target:bool,field_name):
        '''
        Add an encoder that loads the given field name ("dog" will load "dog.npy" for the given example in the batch)[
        :param source_or_target: this loader is applied to source or target?
        :param field_name: the fieldname to load
        :return:
        '''
        gem = _LoadedDataEncoder(field_name)
        self.__add_generator(source_or_target, gem)

    def get_keys_to_load(self,source_or_target:bool):
        '''
        get all keys that are needed to load from disk
        :param source_or_target: give keys for source, or target
        :return: list of strings for all keys
        '''
        return self.__keys_to_load[source_or_target]

    def __get_partial_code(self, the_obj, source_or_target: bool):
        '''
        given a batch object, process and return the partial code representing the source/target
        :param batch: Batch object, for which codes are to be computed
        :param source_or_target: return partial code for source, or target?
        :return:
        '''
        codes = []

        for generator in self.__generators[source_or_target]:
            keys = generator.to_load()
            if isinstance(keys,str):
                code = the_obj.get_loaded_data(keys)
            else:
                assert(isinstance(keys,list))
                code = []
                for key in keys:
                    code.append(the_obj.get_loaded_data(key))
            code = generator.encode(code)
            code = code.view(code.shape[0], -1)
            codes.append(code)
        if len(codes) == 0:
            return None
        ret = torch.cat(codes, dim=1)
        assert(ret.shape[0] == codes[0].shape[0])
        return ret
    # NOTE: Pointnet encoding HERE
    def encode_deformation(self, source,target):
        '''
        get the code for the deformations in a batch
        :param batch: Batch object
        :return: a batch of codes representing the deformations in the batch
        '''
        s = self.__get_partial_code(source, DeformationEncoder.SOURCE)
        t = self.__get_partial_code(target, DeformationEncoder.TARGET)
        if s is None:
            return t
        if t is None:
            return s
        s = s.repeat(t.shape[0],1)
        ret = torch.cat((s,t),1)
        assert(ret.shape[0] == s.shape[0])
        return ret

    def get_code_length(self,dataset):
        '''
        return the length of the code for a given example. Since this is decided at run time, this loads one batch from
        the dataset and checks the code length direclty
        :param dataset: Dataset object that returns batches
        :return: integer specifying code lenth
        '''
        if self.__args.noencoder:
            return 0

        b = dataset[0]

        c = self.encode_deformation(b[0],b[1])
        return c.shape[1]


