import os
import glob
import numpy as np
from scipy.sparse import load_npz

import filecmp
dir = { False:'data/faust/_processed', True: 'data/faust_processed'}
# def subdirs(rootdir):
#     ret = []
#     for file in os.listdir(rootdir):
#         d = os.path.join(rootdir, file)
#         if os.path.isdir(d):
#             ret.append(d)
#     return ret
# def compare_subdir(subdirs)
def print_diff_files(dcmp,vals = [0]):
    for name in dcmp.same_files:
         #print("diff_file %s found in %s and %s" % (name, dcmp.left,
         #      dcmp.right))

         if name[-3:]=='npy':
             left = np.load(os.path.join(dcmp.left,name))
             right = np.load(os.path.join(dcmp.right, name))
             val = (np.linalg.norm(left - right))
         elif name[-3:]=='npz':
             left = load_npz(os.path.join(dcmp.left, name))
             right = load_npz(os.path.join(dcmp.right, name))
             val = (left!=right).nnz
         else:
             raise Exception("nothing else should be here")
         #print(f'{name}: {val}')
         vals.append(val)
    for sub_dcmp in dcmp.subdirs.values():
         print(sub_dcmp)
         vals.append(max(print_diff_files(sub_dcmp,vals)))
    return vals


dcmp = filecmp.dircmp('data/faust/_processed', 'data/faust_processed',ignore=['samples.npy','centroids_wks.npy', 'samples_normals.npy', 'samples_wks.npy'])
#dcmp.report_full_closure()
vals  = print_diff_files(dcmp)
print(max(vals))
#files_old = glob.glob(os.path.join(sdir,'*.off'))