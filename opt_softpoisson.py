### Jacobian + Laplacian weight optimization
from meshing.io import PolygonSoup
from meshing.mesh import Mesh
import numpy as np
import os
import igl
from scipy.sparse import diags
import polyscope as ps
import torch
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument("--objdir",
                    help='path to obj file to optimize for',
                    type = str, default='./data/tetrahedron.obj')
parser.add_argument("--niters",
                    type = int, default=50000)

args = parser.parse_args()

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

### Weighting experiments
datadir = args.objdir
soup = PolygonSoup.from_obj(datadir)
mesh = Mesh(soup.vertices, soup.indices)

ogvertices = torch.from_numpy(mesh.vertices).float().to(device)
ogfaces = torch.from_numpy(mesh.faces).long().to(device)

ogfverts = mesh.vertices[mesh.faces]
soupvs = torch.from_numpy(ogfverts.reshape(-1, 3)).float().to(device)
soupfs = torch.arange(len(soupvs)).reshape(-1, 3).long()

# Compute tutte embedding of original mesh
from source_njf.utils import tutte_embedding, get_jacobian_torch, SLIM
slimuv = torch.from_numpy(SLIM(mesh)[0]).float().to(device)
# slimuv = torch.from_numpy(tutte_embedding(mesh.vertices, mesh.faces)).float()
vertices = torch.from_numpy(mesh.vertices).float().to(device)
faces = torch.from_numpy(mesh.faces).long().to(device)
fverts = vertices[faces]

# Get Jacobians
slimj = get_jacobian_torch(vertices, faces, slimuv).to(device) # F x 2 x 3
slimj = torch.cat((slimj, torch.zeros((slimj.shape[0], 1, 3))), axis=1) # F x 3 x 3
slimj.requires_grad_()

# Sanity check: jacobians return original isosoup up to global translation
pred_V = fverts @ slimj[:,:2,:].transpose(2, 1)
diff = pred_V - slimuv[faces]
diff -= torch.mean(diff, dim=1, keepdims=True) # Removes effect of per-triangle clobal translation
torch.testing.assert_close(diff, torch.zeros_like(diff), atol=1e-5, rtol=1e-4)

# Compute differential operators for triangle soup
dim = 3
is_sparse = False

grad = igl.grad(soupvs.numpy(), soupfs.numpy())
d_area = igl.doublearea(soupvs.numpy(), soupfs.numpy())
d_area = np.hstack((d_area, d_area, d_area)) # This matches the format for grad matrix (t0, t1, t2, ..., t0, t1, t2, ..., t0, t1, t2, ...)
mass = diags(d_area)
rhs = grad.T@mass
rhs = rhs.todense()
laplace = np.array((grad.T@mass@grad).todense())

## Update diagonal of Laplacian such that rows sum to 0
laplace = laplace - np.diag(np.sum(laplace, axis=1))
np.testing.assert_allclose(np.sum(laplace, axis=1), 0, atol=1e-4)

grad = grad.todense()

## Convert to torch tensors
laplace = torch.from_numpy(laplace).float().to(device)
rhs = torch.from_numpy(rhs).float().to(device)

# Triangle soup solve
input = slimj.transpose(2, 1).reshape(1, -1, 3) # 1 x F*3 x 3

# Reshape the Jacobians to match the format of grad (vertex ordering STAYS THE SAME)
P = torch.zeros(input.shape).to(device)
k = input.shape[1] // 3
P[:, :k, :] = input[:, ::3] # First row of all jacobians together
P[:, k:2 * k, :] = input[:, 1::3] # Second row of all jacobians together
P[:, 2 * k:, :] = input[:, 2::3] # Third row of all jacobians together

input_to_solve = rhs @ P

# Get valid pairs (original correspondences) => flat version of Q matrix
from source_njf.utils import vertex_soup_correspondences
from itertools import combinations
vcorrespondences = vertex_soup_correspondences(ogfaces.numpy())
valid_pairs = []
for ogv, vlist in sorted(vcorrespondences.items()):
    valid_pairs.extend(list(combinations(vlist, 2)))

# Initialize random weights
from source_njf.losses import vertexseparation, symmetricdirichlet
from source_njf.utils import get_jacobian_torch

jacweight = 10
distweight = 0.1
sparseweight = 1
seplossdelta = 0.1
weight_idxs = (torch.tensor([pair[0] for pair in valid_pairs]).to(device), torch.tensor([pair[1] for pair in valid_pairs]).to(device))
# weights = (torch.rand(len(weight_idxs[0])) - 0.5) * 4 # -2 to 2
weights = torch.zeros(len(weight_idxs[0])).to(device)
weights.requires_grad_()
optim = torch.optim.Adam([weights, slimj], lr=1e-4)

savedir = "/Users/guanzhi/Documents/Graphics/NJFWand/outputs/optimization"
objname = os.path.basename(datadir).split(".")[0]
screendir = os.path.join(savedir, objname)
framedir = os.path.join(screendir, "frames")
Path(framedir).mkdir(parents=True, exist_ok=True)

### Screenshot initialization mesh and UV
ps.init()
ps.remove_all_structures()
ps_mesh = ps.register_surface_mesh("mesh", soupvs.detach().cpu().numpy(), soupfs.detach().cpu().numpy(), edge_width=1)
ps_mesh.add_scalar_quantity('corr', np.arange(len(soupfs)), defined_on='faces', enabled=True, cmap='viridis')
ps.screenshot(os.path.join(screendir, f"mesh.png"), transparent_bg=True)

ps.remove_all_structures()
ps_uv = ps.register_surface_mesh("uv", slimuv.detach().cpu().numpy(), faces.detach().cpu().numpy(), edge_width=1)
ps_uv.add_scalar_quantity('corr', np.arange(len(soupfs)), defined_on='faces', enabled=True, cmap='viridis')
ps.screenshot(os.path.join(screendir, f"inituv.png"), transparent_bg=True)

for i in range(args.niters):
    # Weights need to be negative going into laplacian
    cweights = -torch.sigmoid(weights)

    ## Recompute the RHS based on new jacobians
    input = slimj.transpose(2, 1).reshape(1, -1, 3) # 1 x F*3 x 3

    # Reshape the Jacobians to match the format of grad (vertex ordering STAYS THE SAME)
    P = torch.zeros(input.shape)
    k = input.shape[1] // 3
    P[:, :k, :] = input[:, ::3] # First row of all jacobians together
    P[:, k:2 * k, :] = input[:, 1::3] # Second row of all jacobians together
    P[:, 2 * k:, :] = input[:, 2::3] # Third row of all jacobians together

    input_to_solve = rhs @ P

    # Debugging: check gradients
    cweights.retain_grad()

    templaplace = laplace.clone().detach()
    templaplace[weight_idxs] = cweights # F*3 x F*3

    # Assign opposite pair
    templaplace[weight_idxs[1], weight_idxs[0]] = cweights

    # Pin vertices
    pinlaplace = templaplace
    pin_input = input_to_solve

    # Diagonal needs to be equal to sum of pair weights for each row
    pinlaplace = pinlaplace - torch.diag(torch.sum(pinlaplace, dim=1))
    torch.testing.assert_close(torch.sum(pinlaplace, dim=1), torch.zeros(pinlaplace.shape[0]), atol=1e-4, rtol=1e-4)

    # Solve with pinned vertices
    ## Solve 1: detach RHS for losses
    uvs = torch.linalg.solve(pinlaplace[1:,1:], pin_input[0, 1:]) # F*3 x 3
    uvs = uvs[:,:2]

    # Sanity check: manually compute gradients through laplacian
    # invlaplace = torch.linalg.inv(pinlaplace[1:,1:])
    # checkgrad = invlaplace[3,3] * torch.sum(invlaplace[:,3])

    # Add back pinned vertex
    full_uvs = torch.cat([torch.zeros(1, 2), uvs], dim=0)

    # distortionloss = arapweight * arap(local_tris, soupfaces, uvs, return_face_energy=True, renormalize=False)
    separationloss = torch.sum(vertexseparation(ogvertices, ogfaces, full_uvs, loss='l2'), dim=1)

    # L0 relaxation
    separationloss = (separationloss * separationloss)/(separationloss * separationloss + seplossdelta)

    # Jacobian matching loss
    poisj = get_jacobian_torch(soupvs, soupfs, full_uvs)
    jacobloss = torch.nn.functional.mse_loss(poisj, slimj[:, :2, :].detach(), reduction='none')

    ## Solve 2: keep RHS and compute jacobians
    pinlaplace_rhs = pinlaplace[1:,1:].clone()
    pin_input_rhs = pin_input[0, 1:].clone()
    uvs_rhs = torch.linalg.solve(pinlaplace_rhs, pin_input_rhs) # F*3 x 3
    uvs_rhs = uvs_rhs[:,:2]

    # Distortion loss on poisson solved jacobians
    full_uvs_rhs = torch.cat([torch.zeros(1, 2), uvs_rhs], dim=0)
    poisj_rhs = get_jacobian_torch(soupvs, soupfs, full_uvs_rhs)
    distortionloss = symmetricdirichlet(ogvertices, ogfaces, poisj_rhs)
    # distortionloss = symmetricdirichlet(ogvertices, ogfaces, slimj[:,:2,:])

    # Weight max loss (want to make sum of weights as high as possible (minimize cuts))
    weightloss = torch.sum(cweights)

    # Compute loss
    # loss = torch.mean(separationloss)
    loss = torch.mean(separationloss) + jacweight * torch.mean(jacobloss) + distweight * torch.mean(distortionloss)
    loss.backward()

    # Sanity check: cweight gradients
    # print(f"Cweight gradients: {cweights.grad}")
    # print(f"Weight gradients: {weights.grad}")
    # print(f"Jacobian gradients: {slimj.grad}")

    optim.step()
    optim.zero_grad()

    print(f"========== Done with iteration {i}. ==========")
    print(f"Separation loss: {torch.mean(separationloss):0.4f}. Jacobian loss: {torch.mean(jacobloss):0.4f}. Distortion loss: {torch.mean(distortionloss):0.4f}.")
    # print(f"Cweights: {-cweights.detach()}")
    # print(f"Weights: {weights.detach()}")
    # print()

    # Visualize every 100 epochs
    if i % 100 == 0:
        ps.init()
        ps.remove_all_structures()
        # ps_mesh = ps.register_surface_mesh("mesh", soupvs.detach().cpu().numpy(), soupfs.detach().cpu().numpy(), edge_width=1)
        ps_uv = ps.register_surface_mesh("uv mesh", full_uvs.detach().cpu().numpy(), soupfs.detach().cpu().numpy(), edge_width=1)
        ps_uv.add_scalar_quantity('corr', np.arange(len(soupfs)), defined_on='faces', enabled=True, cmap='viridis')
        ps.screenshot(os.path.join(framedir, f"uv_{i:06}.png"), transparent_bg=True)

# Convert screenshots to gif
import glob
from PIL import Image

fp_in = f"{framedir}/uv_*.png"
fp_out = f"{screendir}/uv.gif"
imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]

# Resize images
imgs[0].save(fp=fp_out, format='GIF', append_images=imgs[1:],
        save_all=True, duration=20, loop=0, disposal=2)