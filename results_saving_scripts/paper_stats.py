import numpy as np
import torch
class PaperStats:
    def __init__(self):
        self.L2_V = []
        self.L2_J = []
        self.source_areas = []
        self.pred_areas = []
        self.gt_areas = []
        self.pred_times = []
        self.angle_N = []
        self.flips = []
        self.dist_metric = []
        self.slim = []
        self.flips_gt = []
        self.dist_metric_gt = []
        self.slim_gt = []

    def add_flips(self,flips):
        self.flips.append(flips)

    def add_slim(self,slim):
        self.slim.append(slim)

    def add_dist_metric(self,dist_metric):
        self.dist_metric.append(dist_metric)

    def add_flips_gt(self,flips_gt):
        self.flips_gt.append(flips_gt)

    def add_slim_gt(self,slim_gt):
        self.slim_gt.append(slim_gt)

    def add_dist_metric_gt(self,dist_metric_gt):
        self.dist_metric_gt.append(dist_metric_gt)

    def add_area(self,a):
        self.source_areas.append(a)

    def add_V(self,pred_V,gt_V):
        dist, scaling_factor_pred, scaling_factor_gt = PaperStats.nomalized_unit_sphere_L2(pred_V,gt_V)
        self.L2_V.append(dist.mean().item())
        self.pred_areas.append(scaling_factor_pred)
        self.gt_areas.append(scaling_factor_gt)

    def add_J(self,pred_J,gt_J):
        # normalize before computing L2.
        self.L2_J.append(PaperStats.L2(pred_J.view(pred_J.size(0), -1), gt_J.view(gt_J.size(0), -1)).mean().item())

    def add_pred_time(self,pred_time):
        self.pred_times.append(pred_time)

    def normalized_stats(self):
        # a = np.mean(self.source_areas)
        return (self.L2_V, self.L2_J, self.angle_N, self.slim)

    def add_angle_N(self,angle_N):
        self.angle_N.append(angle_N)

    def dump(self,fname):
        np.savez(fname+'.npz',area = self.source_areas,l2_v = self.L2_V,l2_j = self.L2_J,andle_N =self.angle_N,slim = self.slim,flips = self.flips, pred_times=self.pred_times, dist_metric = self.dist_metric, dist_metric_gt = self.dist_metric_gt )
        v,j,n,s = self.normalized_stats()
        with open(fname+'.txt','w') as f:
            f.write(f'L2 V per sample, normalized unit sphere: {np.mean(v)}\n')
            f.write(f'L2 J per sample, normalized unit sphere: {np.mean(j)}\n')
            f.write(f'L2 normals: {np.mean(n)}\n')
            f.write(f'L2 normals degress: {np.mean(n)*57.2958}\n')
            if len(self.flips) > 0:



                f.write(f'UV: Sym Dirichlet & Mean # Flips & Mesh with flips & Mean d>10 & Med d>10 \n')
                f.write(f'GT Paper: {np.mean(self.slim_gt):.1f} & {np.mean(self.flips_gt):.3f} & {100*np.sum([flp >0 for flp in self.flips_gt])/len(self.flips_gt):.1f} &  {np.mean(self.dist_metric_gt):.2f} & {PaperStats.compute_median(self.dist_metric)}  \n')
                f.write(f'Pred Paper: {np.mean(self.slim):.1f} & {np.mean(self.flips):.3f} &  {100*np.sum([flp >0 for flp in self.flips])/len(self.flips):.1f} &  {np.mean(self.dist_metric):.2f} & {PaperStats.compute_median(self.dist_metric)} \n')
                f.write(f'\n')
                f.write(f'\n')
                f.write(f'\n')

                f.write(f'Pred: {np.mean(self.slim):.2f} & {np.mean(self.flips):.3f} & {np.sum([flp >0 for flp in self.flips])}/{len(self.flips)} %{100*np.sum([flp >0 for flp in self.flips])/len(self.flips):.1f} &  {np.mean(self.dist_metric):.2f} & {PaperStats.compute_median(self.dist_metric)} \n')
                f.write(f'GT: {np.mean(self.slim_gt):.2f} & {np.mean(self.flips_gt):.3f} & {np.sum([flp >0 for flp in self.flips_gt])}/{len(self.flips_gt)} %{100*np.sum([flp >0 for flp in self.flips_gt])/len(self.flips_gt):.1f} &  {np.mean(self.dist_metric_gt):.2f} & {PaperStats.compute_median(self.dist_metric)}  \n')
                
                f.write(f'\n')
                f.write(f'\n')
                f.write(f'\n')
                f.write(f'Symmetric Dirichlet: {np.mean(s)}\n')
                f.write(f'Average flip count: {np.mean(self.flips)}\n')
                f.write(f'meshes with flips: {np.sum([flp >0 for flp in self.flips])}/{len(self.flips)} % {100*np.sum([flp >0 for flp in self.flips])/len(self.flips)}\n')
            f.write(f'average feedforward time: {np.mean(self.pred_times[1:])} seconds\n') # I am excluding the first one because it's slower because of initialization. Jitting the network would make things faster.
            f.write(f'average feedforward time fps: {1/np.mean(self.pred_times[1:])} seconds\n') # I am excluding the first one because it's slower because of initialization. Jitting the network would make things faster.

    @staticmethod
    def compute_median(test_list):
        test_list.sort()
        mid = len(test_list) // 2
        res = (test_list[mid] + test_list[~mid]) / 2
        return res

        
    @staticmethod
    def nomalized_unit_sphere_L2(in_array, out_array):
        #center
        in_array = in_array - in_array.mean(axis=0, keepdim = True)
        #unit sphere
        scaling_factor_in = torch.sqrt((in_array**2).sum(1).max()).item()
        in_array = in_array / scaling_factor_in
        #center
        out_array = out_array - out_array.mean(axis=0, keepdim = True)
        #unit sphere
        scaling_factor_out = torch.sqrt((out_array**2).sum(1).max()).item()
        out_array = out_array / scaling_factor_out
        return torch.sqrt(((in_array-out_array)**2).sum(1)), scaling_factor_in, scaling_factor_out

    @staticmethod
    def centered_L2(in_array, out_array):
        #center
        in_array = in_array - in_array.mean(axis=0, keepdim = True)
        #center
        out_array = out_array - out_array.mean(axis=0, keepdim = True)
        return torch.sqrt(((in_array-out_array)**2).sum(1))

    @staticmethod
    def L2(in_array, out_array):
        return torch.sqrt(((in_array-out_array)**2).sum(1))