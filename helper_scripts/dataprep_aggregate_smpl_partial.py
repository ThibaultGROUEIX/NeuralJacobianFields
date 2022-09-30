import os
import igl
import dataprep_create_errorcolor_gltf
import numpy as np

dir = 'C:/Users/aigerman/Dropbox/ndef/smpl/partials/partial_figure'
# dir = 'C:/Users/aigerman/Dropbox/ndef/smpl/smpl_test'
pred_meshes = []
gt_radii = []
dists = []
fs = []
max_dist = 0
print('read')
V_err = []
for f in os.listdir(dir):

    if not f.endswith('.obj'):
        continue
    print(f)
    temp_f = f[:- 4].split('_')
    if len(temp_f)==1:
        continue
    temp_f = temp_f[-1]
    print(temp_f)
    if not temp_f.isdigit():
        continue

    fs.append(f)
    gt = f.replace('.obj', '_reduced_list_target.npy')
    pV, _, _, pT, _, _ = igl.read_obj(dir + '/' + f)
    gV = np.load(dir + '/'+gt)
    # a = igl.doublearea(gV,gT)
    gt_temp = gV - np.mean(gV, 0)

    gt_norm = np.sum(np.sqrt(gt_temp ** 2), 1)
    gt_radii.append(np.max(gt_norm))
    d = np.sum(np.sqrt((pV - gV) ** 2), 1)
    V_err.append(d)
    max_dist = max(np.max(d), max_dist)
    dists.append(d)
    pred_meshes.append((pV, pT))

assert len(pred_meshes) > 0
max_radius = np.max(gt_radii)
max_dist /= (max_radius)
max_dist = 0.2  # max(max_dist,0.1)
print('write')
for i in range(len(pred_meshes)):
    V, T = pred_meshes[i]
    d = V_err[i]
    vals = d / (max_radius)

    dataprep_create_errorcolor_gltf.write(dir + '/pred/' + fs[i].replace('_pred.obj', ''), V, T, vals, 0, max_dist)

