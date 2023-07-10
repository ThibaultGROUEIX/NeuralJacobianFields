from meshing.io import PolygonSoup
from meshing.mesh import Mesh
import numpy as np
import os
import igl
from scipy.sparse import diags
import polyscope as ps
from source_njf.utils import SLIM, get_jacobian

nocutdir = './data/coarsecylinder_nocut/cylinder.obj'
cylsoup = PolygonSoup.from_obj(nocutdir)
mesh = Mesh(cylsoup.vertices, cylsoup.indices)
# mesh.normalize()

cutcyldir = './data/coarsecylinder/cylinder.obj'
cutsoup = PolygonSoup.from_obj(cutcyldir)
cutmesh = Mesh(cutsoup.vertices, cutsoup.indices)
# cutmesh.normalize()

uv, energy = SLIM(cutmesh)
uv_fverts = uv[cutmesh.faces]

## Visualize the boundary to find the right vertices to cut
# ps.init()
# ps.remove_all_structures()
# fverts = mesh.vertices[mesh.faces]
# ps_mesh = ps.register_surface_mesh("uncut mesh", fverts.reshape(-1, 3), np.arange(len(fverts)*3).reshape(-1, 3), edge_width=1)
# ps_mesh.add_scalar_quantity("vidx", np.arange(len(fverts)*3), defined_on='vertices', enabled=True)

# cutfverts = cutmesh.vertices[cutmesh.faces]
# buffer = np.zeros(cutfverts.reshape(-1, 3).shape)
# buffer[:,0] = 2
# ps_cut = ps.register_surface_mesh("cut mesh", cutfverts.reshape(-1, 3) + buffer, np.arange(len(cutfverts)*3).reshape(-1, 3), edge_width=1)
# ps_cut.add_scalar_quantity("vidx", np.arange(len(cutfverts)*3), defined_on='vertices', enabled=True)

# boundaryvs = [v.index for v in cutmesh.topology.boundaries[0].adjacentVertices()]
# boundaryes = np.array([[i, i+1] for i in range(len(boundaryvs)-1)] + [[len(boundaryvs)-1, 0]])
# ps_curve = ps.register_curve_network("boundary", cutmesh.vertices[boundaryvs], boundaryes, enabled=True)
# ps.show()
# raise

# Map cutmesh vs to mesh vs
fverts = mesh.vertices[mesh.faces]
cutfverts = cutmesh.vertices[cutmesh.faces]
cuttoog = [] # transpose indices to go from cutmesh fverts to mesh fverts
for i in range(len(fverts)):
    f = fverts[i] # 3 x 3
    cutf = cutfverts[i]
    trans = []
    for j in range(len(f)):
        diff = np.linalg.norm(cutf - f[[j]], axis=1)
        tidx = np.argmin(diff)
        np.testing.assert_allclose(diff[tidx], 0)
        trans.append(tidx)
    cuttoog.append(trans)
gtuv = uv_fverts[np.arange(len(uv_fverts)).reshape(-1, 1), np.array(cuttoog)]

# Get Jacobians
fverts = mesh.vertices[mesh.faces]
uvj = get_jacobian(fverts.reshape(-1, 3), np.arange(len(fverts)*3).reshape(-1, 3), gtuv.reshape(-1, 2))

# Sanity check Jacobians
pred_V = np.einsum("abc,acd->abd", fverts, uvj.transpose(0, 2, 1)) # F x 3 x 2
diff = pred_V - gtuv
diff -= np.mean(diff, axis=1, keepdims=True) # Removes effect of per-triangle global translation
np.testing.assert_allclose(diff, np.zeros(diff.shape), rtol=1e-4, atol=1e-4)

cutj = np.concatenate([uvj, np.zeros((uvj.shape[0],1, uvj.shape[2]))], axis=1)

# Triangle soup jacobians
from scipy.sparse import bmat
from source_njf.utils import get_local_tris
import torch
from igl import grad

ogvs = torch.from_numpy(mesh.vertices).float()
ogfs = torch.from_numpy(mesh.faces).long()
fverts = ogvs[ogfs]
local_tris = get_local_tris(ogvs, ogfs).detach().cpu().numpy() # F x 3 x 2

# Convert local tris to soup
isosoup = local_tris.reshape(-1, 2) # V x 2

# Compute differential operators for triangle soup
fverts = mesh.vertices[mesh.faces]
vertices = fverts.reshape(-1, 3)
faces = np.arange(len(vertices)).reshape(-1, 3)
dim = 3
is_sparse = False

grad = igl.grad(vertices, faces)
d_area = igl.doublearea(vertices,faces)
d_area = np.hstack((d_area, d_area, d_area)) # This matches the format for grad matrix (t0, t1, t2, ..., t0, t1, t2, ..., t0, t1, t2, ...)
mass = diags(d_area)

laplace = np.array((grad.T@mass@grad).todense()).astype(np.float64)

## Set weights of Laplacian according to original topology (need v to v correspondences)
from source_njf.utils import vertex_soup_correspondences
from itertools import combinations

vcorr = vertex_soup_correspondences(mesh.faces)
cutvs = [26, 25, 28]
cutvdict = {26: [[33, 32], [45]], 25:[[45,49,56,50], [32, 31, 30, 29]], 28:[[35, 50], [29]]} # Keys index vertices, values are lists of face groups to stitch together

for key, soupvs in vcorr.items():
    # Don't stitch original cut
    # if key in cutvs:
    #     continue
    for vpair in combinations(soupvs, 2):
        if key in cutvs:
            fgroups = cutvdict[key]
            vgroups = []
            for fgroup in fgroups:
                vgroups.append(np.unique(faces[fgroup]).tolist())

            v1_idx = None
            v2_idx = None
            for i, vgroup in enumerate(vgroups):
                if vpair[0] in vgroup:
                    v1_idx = i
                if vpair[1] in vgroup:
                    v2_idx = i

            # If both vertices are in different groups, then cut (don't stitch)
            if v1_idx != v2_idx:
                continue

        laplace[vpair[0], vpair[1]] = 1000.
        laplace[vpair[1], vpair[0]] = 1000.

# Pin the vertex corresponding to face 45 vertex 26
lap_pinned = None
# lap_pinned = [96,101]
# lap_pinned = np.array(lap_pinned)
# laplace = np.delete(np.delete(laplace, lap_pinned, axis=0), lap_pinned, axis=1) # Remove row and column

## Update diagonal of Laplacian such that rows sum to 0
laplace = laplace - np.diag(np.sum(laplace, axis=1))
np.testing.assert_allclose(np.sum(laplace, axis=1), 0, atol=1e-4)

# To dense np array
rhs = grad.T@mass
rhs = rhs.todense()

# Also delete from RHS
# rhs = np.delete(rhs, lap_pinned, axis=0)
grad = grad.todense()

# Triangle soup solve
input = cutj.transpose(0,2,1).reshape(1, -1, 3) # 1 x F*3 x 3

# Reshape the Jacobians to match the format of grad (vertex ordering STAYS THE SAME)
P = np.zeros(input.shape)
k = input.shape[1] // 3
P[:, :k, :] = input[:, ::3] # First row of all jacobians together
P[:, k:2 * k, :] = input[:, 1::3] # Second row of all jacobians together
P[:, 2 * k:, :] = input[:, 2::3] # Third row of all jacobians together
P = P.astype(np.float64)

input_to_solve = rhs @ P

# Solve layer
# Lap = laplacian (cholesky mode but doesn't matter)
out, residuals, rank, s = np.linalg.lstsq(laplace, input_to_solve, rcond=None) # V x 2
out = np.array(out) # matrix to array
c = np.mean(out, axis=0, keepdims=True)
out -= c
print(f"Problem rank: {rank}. Laplacian shape: {laplace.shape}")

if lap_pinned is not None:
    lap_pinned = np.sort(lap_pinned)
    for i in range(len(lap_pinned)):
        # The index should never be larger than length of array
        pidx = lap_pinned[i]
        assert out.shape[0] >= pidx, f"Indexing error: trying to insert at {pidx} for array of size {out.shape[0]}"
        out = np.concatenate([out[:pidx], np.zeros((1, out.shape[1])), out[pidx:]], axis=0)

# Sanity check: check lstsq residual of GT UVs
# gtuv = np.concatenate([gtuv.reshape(-1, 2), np.zeros((len(uv_fverts)*3, 1))], axis=1)
# gtcheck = laplace @ gtuv - input_to_solve
# gtresid = np.linalg.norm(gtcheck, axis=1)

# print("GT UVs")
# print(np.mean(gtresid))
# print(np.sum(gtresid))

# outcheck = laplace @ out - input_to_solve
# outresid = np.linalg.norm(outcheck, axis=1)

# print("Out UVs")
# print(np.mean(outresid))
# print(np.sum(outresid))
# raise

# Sanity check: solution UVs should be exactly the same as solution given original topology Laplacian
# outcheck = out[:,:2]
# diff = outcheck.reshape(-1,3,2) - uv_fverts
# diff -= np.mean(diff, axis=1, keepdims=True)
# np.testing.assert_allclose(diff, 0, atol=1e-6)

# Visualize
gtuv = uv_fverts.reshape(-1, 2)
gtface = np.arange(len(gtuv)).reshape(-1, 3)
buffer = np.zeros(gtuv.shape)
buffer[:,0] = 4

ps.init()
ps.remove_all_structures()
pspois = ps.register_surface_mesh("pois uv", out, np.arange(len(out)).reshape(-1, 3), edge_width=1)
pspois.add_scalar_quantity("corr", np.arange(len(out)//3), defined_on='faces', enabled=True)
psgt = ps.register_surface_mesh("gt uv", gtuv + buffer, gtface, edge_width=1, edge_color=[1,0,0])
psgt.add_scalar_quantity("corr", np.arange(len(mesh.faces)), defined_on='faces', enabled=True)
# ps.show()

# Compute symmetric dirichlet energy on the poisson solve
from source_njf.losses import symmetricdirichlet

poisj = torch.from_numpy(get_jacobian(fverts.reshape(-1, 3), np.arange(len(fverts)*3).reshape(-1, 3), out[:,:2])).float()
energy = symmetricdirichlet(torch.from_numpy(mesh.vertices).float(), torch.from_numpy(mesh.faces).long(), poisj).detach().numpy()
pspois.add_scalar_quantity("symmetric dirichlet", energy, defined_on='faces', enabled=True, cmap='reds', vminmax = (0, 5))
ps.show()