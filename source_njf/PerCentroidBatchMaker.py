import torch
class PerCentroidBatchMaker:
    '''
    This class prepares data for the network.
    The neural network needs to receive ONE code C describing the source/target pair (and everything else) and ONE centroid+normal of the triangle for which it is predicitng a jacobian
    Hence, this class Takes [k x C] target codes, and [T x (something like 3)] source centroids, nad creates a tensor of
    k*T x (C + something like 3) for the network to consume. It also has a function for the inverse
    '''
    def __init__(self, codes, centroids_and_normals, args):
        self.args = args
        self.__codes = codes
        self.__centroids_and_normals = centroids_and_normals
        assert (len(codes.shape) == 2)
        assert (len(centroids_and_normals.shape) == 2)
        self.__tri_len = centroids_and_normals.shape[0]
        self.__target_len = codes.shape[0]
        self.__code_len = codes.shape[1]
        self.__centroids_and_normals_len = centroids_and_normals.shape[1]
        self.use_linear = False

    def to_stacked(self):
        '''
        :return: codes and centroids ready to be consumed by the net
        '''
        # Ntargets x codelen codes,  trilen x 6 (= cen 3 + nrml 3)

        #want to generate Ntargets x trilen x (6 + codelen codes)
        #first make into right shape
        codes = self.__codes.unsqueeze(1)#dim for tris
        centroids_and_normals = self.__centroids_and_normals.unsqueeze(0)#dim for codes
        codes = codes.expand((-1,self.__tri_len,-1))
        centroids_and_normals = centroids_and_normals.expand((self.__target_len,-1,-1))
        cartesian = torch.cat((codes,centroids_and_normals),2)
        output = cartesian.view(self.__target_len,self.__tri_len, self.__code_len+self.__centroids_and_normals_len).contiguous()

        # Shuffle to ensure group norm works well
        if self.args.shuffle_triangles:
            # print("shuffling triangles in PerCentroidBatchMaker")
            idx = torch.randperm(output.shape[1])
            self.idx = idx
            self.idx_inv = torch.empty_like(idx)
            self.idx_inv[idx] = torch.arange(idx.size()[0])
            output = output[:,idx]

        return output.contiguous()

    def prep_for_linear_layer(self, tensor):
        self.use_linear = True
        return tensor.view(self.__tri_len*self.__target_len, self.__code_len+self.__centroids_and_normals_len)

    def prep_for_conv1d(self, tensor):
        self.use_linear = False
        return tensor.transpose(1,2).contiguous()


    def back_to_non_stacked(self, input):
        '''
        :param input: stacked output of the network, [codes*tris,org_shape]
        :return: the same output, reshaped so its [codes,tris,org_shape
        '''
        if not self.use_linear:
            input = input.transpose(1,2).contiguous()

        output = input.view(self.__target_len,self.__tri_len,-1)
        if self.args.shuffle_triangles:
            # print("Unshuffling triangles in PerCentroidBatchMaker")
            output = output[:,self.idx_inv]
        return output

if __name__ == '__main__':
    a = torch.tensor([
    [1,1,1,1],
    [2,2,2,2],
    [3,3,3,3]], dtype=torch.float32)
    b = torch.tensor([
        [4, 5, 6],
        [7,8,9],
        [10,11,12],
        [13,14,15],
        [16,17,18]
        ], dtype=torch.float32)

    X = PerCentroidBatchMaker(a, b)
    c = X.to_stacked()
    d = X.back_to_non_stacked(c)
    print(d.shape)
    print(d[2,4,4:8] == b[4,:])


