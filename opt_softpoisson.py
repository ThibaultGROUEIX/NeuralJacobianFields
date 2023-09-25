### Jacobian + Laplacian weight optimization
from meshing.io import PolygonSoup
from meshing.mesh import Mesh
import numpy as np
import os
import igl
from scipy.sparse import diags
import torch
import argparse
from pathlib import Path
import dill as pickle
from collections import defaultdict
import torch

parser = argparse.ArgumentParser()

parser.add_argument("--objdir",
                    help='path to obj file to optimize for',
                    type = str, default='./data/tetrahedron.obj')
parser.add_argument("--savename",
                    help='name of experiment',
                    type = str, default=None)
parser.add_argument("--niters",
                    type = int, default=50000)
parser.add_argument("--init", type=str, choices={'slim', 'isometric'},
                    default='slim')
parser.add_argument("--vertexseploss",
                    action='store_true', default=False)
parser.add_argument("--vertexseptype", type=str, choices={'l2', 'l1'}, default='l2')
parser.add_argument("--vertexsepsqrt", help='whether to sqrt the sum of the vertex losses (across the coordinate dimension)',
                    action='store_true', default=False)
parser.add_argument("--edgegradloss",
                    action='store_true', default=False)
parser.add_argument("--edgecutloss", type=str, choices={'vertexsep', 'seamless'}, default=None,
                    help="vertexsep=use whatever outputs from vertex separation loss, seamless=use seamless l0 to approximate threshold")
parser.add_argument("--jmatchloss",
                    action='store_true', default=False)
parser.add_argument("--distortionloss", type=str, choices={"symmetricdirichlet", "arap"},
                    default="symmetricdirichlet")
parser.add_argument("--weightloss",
                    action='store_true', default=False)
parser.add_argument("--seamless",
                    action='store_true', default=False)
parser.add_argument("--stitchweight", help="methods for weighting the stitching loss (approximates convergence to L0 loss)",
                    choices={'seploss', 'softweight', 'softweightdetach'}, default=None)
parser.add_argument("--edgeweight", help="methods for weighting the edge loss",
                    choices={'seploss', 'softweightdetach'}, default=None)
parser.add_argument("--seplossweight", help="use predicted weights as the seploss delta",
                    choices={'detach', 'opt'}, default=None)

parser.add_argument("--edgeweightonly", help='optimize only edge weights. other valid pairs default to 0',
                    action='store_true', default=False)

parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--edgecutweight", type=float, default=1)
parser.add_argument("--seplossdelta", type=float, default=0.1)
parser.add_argument("--cuteps", type=float, default=0.01)
parser.add_argument("--continuetrain",  action='store_true')

args = parser.parse_args()

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

### Check if on cluster

from sys import platform
cluster = False
savedir = "/Users/guanzhi/Documents/Graphics/NJFWand/outputs/optimization"

if platform == "linux":
    cluster = True
    savedir = "/net/scratch/rliu/NJFWand/outputs/optimization"

### Weighting experiments
datadir = args.objdir
soup = PolygonSoup.from_obj(datadir)
mesh = Mesh(soup.vertices, soup.indices)
mesh.normalize()

ogvertices = torch.from_numpy(mesh.vertices).double().to(device)
ogfaces = torch.from_numpy(mesh.faces).long().to(device)

ogfverts = mesh.vertices[mesh.faces]
soupvs = torch.from_numpy(ogfverts.reshape(-1, 3)).double().to(device)
soupfs = torch.arange(len(soupvs)).reshape(-1, 3).long()

# Initialize the UV/Jacobians
from source_njf.utils import tutte_embedding, get_jacobian_torch, SLIM, get_local_tris
vertices = torch.from_numpy(mesh.vertices).double().to(device)
faces = torch.from_numpy(mesh.faces).long().to(device)
fverts = vertices[faces]

if args.init == "slim":
    ## Enforce disk topology
    if len(mesh.topology.boundaries) == 0:
        from numpy.random import default_rng
        from source_njf.utils import generate_random_cuts
        rng = default_rng()
        n_cuts = rng.integers(1, 2)
        cutvs = generate_random_cuts(mesh, enforce_disk_topo=True, max_cuts = n_cuts)
    inituv = torch.from_numpy(SLIM(mesh)[0]).double().to(device)
elif args.init == "isometric":
    vertices = fverts.reshape(-1, 3).double()
    faces = torch.arange(len(vertices)).reshape(-1, 3).long().to(device)
    inituv = get_local_tris(vertices, faces, device=device).reshape(-1, 2).double()

# Get Jacobians
initj = get_jacobian_torch(vertices, faces, inituv, device=device).to(device) # F x 2 x 3
initj = torch.cat((initj, torch.zeros((initj.shape[0], 1, 3), device=device)), axis=1) # F x 3 x 3
initj.requires_grad_()

# Sanity check: jacobians return original isosoup up to global translation
pred_V = fverts @ initj[:,:2,:].transpose(2, 1)
diff = pred_V - inituv[faces]
diff -= torch.mean(diff, dim=1, keepdims=True) # Removes effect of per-triangle clobal translation
torch.testing.assert_close(diff, torch.zeros_like(diff), atol=1e-5, rtol=1e-4)

# Compute differential operators for triangle soup
dim = 3
is_sparse = False

grad = igl.grad(soupvs.detach().cpu().numpy(), soupfs.detach().cpu().numpy())
d_area = igl.doublearea(soupvs.detach().cpu().numpy(), soupfs.detach().cpu().numpy())
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
laplace = torch.from_numpy(laplace).double().to(device)
rhs = torch.from_numpy(rhs).double().to(device)

# Triangle soup solve
input = initj.transpose(2, 1).reshape(1, -1, 3) # 1 x F*3 x 3

# Reshape the Jacobians to match the format of grad (vertex ordering STAYS THE SAME)
P = torch.zeros(input.shape, dtype=input.dtype).to(device)
k = input.shape[1] // 3
P[:, :k, :] = input[:, ::3] # First row of all jacobians together
P[:, k:2 * k, :] = input[:, 1::3] # Second row of all jacobians together
P[:, 2 * k:, :] = input[:, 2::3] # Third row of all jacobians together

input_to_solve = rhs @ P

# Get valid pairs (original correspondences) => flat version of Q matrix
from source_njf.utils import vertex_soup_correspondences
from itertools import combinations
vcorrespondences = vertex_soup_correspondences(ogfaces.detach().cpu().numpy())
valid_pairs = []
for ogv, vlist in sorted(vcorrespondences.items()):
    valid_pairs.extend(list(combinations(vlist, 2)))

# Need edge to soup pairs to visualize cuts
from source_njf.utils import get_edge_pairs
valid_edge_pairs, valid_edges_to_soup, edgeidxs, edgededupidxs, edges, elens, facepairs = get_edge_pairs(mesh, valid_pairs, device=device)

# Initialize random weights
from source_njf.losses import vertexseparation, symmetricdirichlet, uvgradloss, arap
from source_njf.utils import get_jacobian_torch, clear_directory

jacweight = 10
distweight = 1
sparseweight = 1
seplossdelta = args.seplossdelta
weight_idxs = (torch.tensor([pair[0] for pair in valid_pairs]).to(device), torch.tensor([pair[1] for pair in valid_pairs]).to(device))
if args.edgeweightonly:
    weight_idxs = (torch.tensor([pair[0] for pair in valid_edge_pairs]).to(device), torch.tensor([pair[1] for pair in valid_edge_pairs]).to(device))
# weights = (torch.rand(len(weight_idxs[0])) - 0.5) * 4 # -2 to 2

objname = os.path.basename(datadir).split(".")[0]
if args.savename is None:
    screendir = os.path.join(savedir, objname)
else:
    screendir = os.path.join(savedir, args.savename)
Path(screendir).mkdir(parents=True, exist_ok=True)

if args.continuetrain and os.path.exists(os.path.join(screendir, "weights.pt")) and os.path.exists(os.path.join(screendir, "jacobians.pt")):
    weights = torch.load(os.path.join(screendir, "weights.pt")).double().to(device)
    weights.requires_grad_()
    initj = torch.load(os.path.join(screendir, "jacobians.pt")).double().to(device)
    initj.requires_grad_()

    optim = torch.optim.Adam([weights, initj], lr=args.lr)

    try:
        with open(os.path.join(screendir, "epoch.pkl"), "rb") as f:
            starti = pickle.load(f)
    except Exception as e:
        print(e)
        starti = 0

    try:
        with open(os.path.join(screendir, "lossdict.pkl"), "rb") as f:
            lossdict = pickle.load(f)
    except Exception as e:
        print(e)
        lossdict = defaultdict(list)

    if args.stitchweight:
        try:
            stitchweight = torch.load(os.path.join(screendir, "stitchweight.pt")).double().to(device)
        except Exception as e:
            print(e)
            stitchweight = torch.ones(len(weights)).double().to(device)

    if args.edgeweight:
        try:
            edgeweight = torch.load(os.path.join(screendir, "edgeweight.pt")).double().to(device)
        except Exception as e:
            print(e)
            edgeweight = torch.ones(len(edgeidxs)).double().to(device)

    print(f"\n============ Continuing optimization from epoch {starti} ===========\n")
else:
    clear_directory(screendir)
    weights = torch.zeros(len(weight_idxs[0])).to(device).double()
    weights.requires_grad_()
    optim = torch.optim.Adam([weights, initj], lr=args.lr)
    starti = 0
    lossdict = defaultdict(list)
    stitchweight = torch.ones(len(weights)).double().to(device)
    edgeweight = torch.ones(len(edgeidxs)).double().to(device)

framedir = os.path.join(screendir, "frames")
Path(framedir).mkdir(parents=True, exist_ok=True)

if not cluster:
    ### Screenshot initialization mesh and UV
    import polyscope as ps
    ps.init()
    ps.remove_all_structures()
    ps_mesh = ps.register_surface_mesh("mesh", soupvs.detach().cpu().numpy(), soupfs.detach().cpu().numpy(), edge_width=1)
    ps_mesh.add_scalar_quantity('corr', np.arange(len(soupfs)), defined_on='faces', enabled=True, cmap='viridis')
    ps.screenshot(os.path.join(screendir, f"mesh.png"), transparent_bg=True)

    ps.remove_all_structures()
    ps_uv = ps.register_surface_mesh("uv", inituv.detach().cpu().numpy(), faces.detach().cpu().numpy(), edge_width=1)
    ps_uv.add_scalar_quantity('corr', np.arange(len(soupfs)), defined_on='faces', enabled=True, cmap='viridis')
    ps.screenshot(os.path.join(screendir, f"inituv.png"), transparent_bg=True)
else:
    from results_saving_scripts.plot_uv import plot_uv, export_views
    import matplotlib.pyplot as plt

    export_views(soupvs.detach().cpu().numpy(), soupfs.detach().cpu().numpy(), screendir, filename=f"mesh.png",
                plotname=f"starting mesh", n=1, cmap= plt.get_cmap("viridis"),
                fcolor_vals=np.arange(len(soupfs)), device="cpu", n_sample=100, width=400, height=400,
                vmin=0, vmax=len(soupfs), shading=True)
    plot_uv(screendir, f"inituv", inituv.detach().cpu().numpy(),  faces.detach().cpu().numpy(), losses=None,
            facecolors = np.arange(len(soupfs))/(len(soupfs)-1))

for i in range(starti, args.niters):
    # Weights need to be negative going into laplacian
    cweights = -torch.sigmoid(weights)

    # Update stitch weight
    if args.stitchweight and i > starti:
        if args.stitchweight == "seploss":
            stitchweight = 1/(separationloss.detach() + 1e-12)
            assert stitchweight.requires_grad == False
        elif args.stitchweight == "softweight":
            stitchweight = -cweights
        elif args.stitchweight == "softweightdetach":
            stitchweight = -cweights.detach()
            assert stitchweight.requires_grad == False

    if args.edgeweight and i > starti:
        if args.edgeweight == "seploss":
            edgeweight = 1/(edgecutloss.detach() + 1e-12)
            assert edgeweight.requires_grad == False
        elif args.edgeweight == "softweightdetach":
            if args.edgeweightonly:
                edgeweight = -cweights.detach()
            else:
                edgeweight = -cweights[edgeidxs].detach()
            assert edgeweight.requires_grad == False

    ## Recompute the RHS based on new jacobians
    input = initj.transpose(2, 1).reshape(1, -1, 3) # 1 x F*3 x 3

    # Reshape the Jacobians to match the format of grad (vertex ordering STAYS THE SAME)
    P = torch.zeros(input.shape, device=device, dtype=rhs.dtype)
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
    torch.testing.assert_close(torch.sum(pinlaplace, dim=1), torch.zeros(pinlaplace.shape[0], device=device, dtype=pinlaplace.dtype), atol=1e-4, rtol=1e-4)

    # Solve with pinned vertices
    uvs = torch.linalg.solve(pinlaplace[1:,1:], pin_input[0, 1:]) # F*3 x 3
    uvs = uvs[:,:2]

    # Sanity check: manually compute gradients through laplacian
    # invlaplace = torch.linalg.inv(pinlaplace[1:,1:])
    # checkgrad = invlaplace[3,3] * torch.sum(invlaplace[:,3])

    # Add back pinned vertex
    full_uvs = torch.cat([torch.zeros(1, 2, device=device), uvs], dim=0)

    ### Losses
    loss = 0

    # NOTE: Order is wrt weights order (both follow vpairs)
    separationloss = torch.sum(vertexseparation(ogvertices, ogfaces, full_uvs, loss=args.vertexseptype), dim=1)

    if args.vertexsepsqrt:
        separationloss = torch.sqrt(separationloss)

    # Continual L0
    if args.seamless:
        seamlessloss = (separationloss * separationloss)/(separationloss * separationloss + seplossdelta)

    # Weighted separation loss
    if args.vertexseploss:
        if args.stitchweight:
            # NOTE: Stitch weight is normalized such that sum of separation loss roughly equals 1
            if args.seamless:
                weightedseparationloss = stitchweight * seamlessloss
            else:
                weightedseparationloss = stitchweight * separationloss

            loss += torch.mean(weightedseparationloss)
            lossdict['Vertex Separation Loss'].append(torch.mean(weightedseparationloss).item())
        else:
            if args.seamless:
                loss += torch.mean(seamlessloss)
            else:
                loss += torch.mean(separationloss)
            lossdict['Vertex Separation Loss'].append(torch.mean(separationloss).item())

    if args.edgecutloss:
        if args.edgecutloss == "seamless":
            if args.seamless:
                # No need to transform separationloss then
                edgecutloss = seamlessloss[edgeidxs] * elens * args.edgecutweight
            else:
                seamlessloss = (separationloss * separationloss)/(separationloss * separationloss + seplossdelta)
                edgecutloss = seamlessloss[edgeidxs] * elens * args.edgecutweight
        elif args.edgecutloss == "vertexsep":
            edgecutloss = separationloss[edgeidxs] * elens * args.edgecutweight
        else:
            raise NotImplementedError(f"{args.edgecutloss} for edge loss not implemented")

        if args.edgeweight:
            edgecutloss = edgecutloss * edgeweight

        loss += torch.mean(edgecutloss)
        lossdict['Edge Cut Loss'].append(torch.mean(edgecutloss).item())

    # if args.edgegradloss:
    #     edgegradloss, edgecorrespondences = uvgradloss(ogvertices, ogfaces, full_uvs, return_edge_correspondence=True, loss='l2')

    #     # Continual L0
    #     edgegradloss = (edgegradloss * edgegradloss)/(edgegradloss * edgegradloss + seplossdelta)
    #     loss += torch.mean(edgegradloss)

    #     lossdict['Edge Gradient Loss'].append(torch.mean(edgegradloss).item())

    poisj = get_jacobian_torch(soupvs, soupfs, full_uvs, device=device)
    # if args.jmatchloss:
    #     # Jacobian matching loss
    #     jacobloss = torch.nn.functional.mse_loss(poisj, initj[:, :2, :].detach(), reduction='none')
    #     loss += jacweight * torch.mean(jacobloss)
    #     lossdict['Jacobian Matching Loss'].append((jacweight * torch.mean(jacobloss)).item())

    if args.distortionloss:
        # Distortion loss on poisson solved jacobians
        if args.distortionloss == "symmetricdirichlet":
            distortionloss = symmetricdirichlet(ogvertices, ogfaces, poisj)
        if args.distortionloss == "arap":
            distortionloss = arap(ogvertices, ogfaces, full_uvs, paramtris=full_uvs[soupfs], device=device)

        loss += distweight * torch.mean(distortionloss)
        lossdict['Distortion Loss'].append((distweight * torch.mean(distortionloss)).item())

    if args.weightloss:
        # Weight max loss (want to make sum of weights as high as possible (minimize cuts))
        weightloss = torch.mean(cweights)
        loss += sparseweight * weightloss # NOTE: This will be negative!
        lossdict['Sparse Cuts Loss'].append((sparseweight * weightloss).item())

    # Compute loss
    loss.backward()
    lossdict['Total Loss'].append(loss.item())

    # Sanity check: cweight gradients
    # print(f"Cweight gradients: {cweights.grad}")
    # print(f"Weight gradients: {weights.grad}")
    # print(f"Jacobian gradients: {initj.grad}")

    optim.step()
    optim.zero_grad()

    print(f"========== Done with iteration {i}. ==========")
    lossstr = ""
    for k, v in lossdict.items():
        lossstr += f"{k}: {v[-1]:0.7f}. "
    print(lossstr)

    # Also print quantile of stitching weights if applicable
    if args.stitchweight:
        print(f"Stitching weight quantiles: {np.quantile(stitchweight.detach().cpu().numpy(), [0.1, 0.25, 0.5, 0.75, 0.9])}")

    # print(f"Cweights: {-cweights.detach()}")
    # print(f"Weights: {weights.detach()}")
    # print()

    # Visualize every 1000 epochs
    if i % 1000 == 0:
        if not cluster:
            ps.init()
            ps.remove_all_structures()
            # ps_mesh = ps.register_surface_mesh("mesh", soupvs.detach().cpu().numpy(), soupfs.detach().cpu().numpy(), edge_width=1)
            ps_uv = ps.register_surface_mesh("uv mesh", full_uvs.detach().cpu().numpy(), soupfs.detach().cpu().numpy(), edge_width=1)
            ps_uv.add_scalar_quantity('corr', np.arange(len(soupfs)), defined_on='faces', enabled=True, cmap='viridis')
            ps.screenshot(os.path.join(framedir, f"uv_{i:06}.png"), transparent_bg=True)
        else:
            # Get cut edge pairs
            edges = None
            edgecolors = None

            edgestitchcheck = separationloss[edgeidxs].detach().cpu().numpy()
            cutvidxs = np.where(edgestitchcheck > args.cuteps)[0]
            cutsoup = [valid_edges_to_soup[vi] for vi in cutvidxs]

            # TODO: map colors based on cutvidxs instead (fixes the edge correspondences every iter)
            edgecolors = np.repeat(cutvidxs, 2)/(len(edgestitchcheck)-1)
            edges = []
            for souppair in cutsoup:
                edges.append(full_uvs[list(souppair[0])].detach().cpu().numpy())
                edges.append(full_uvs[list(souppair[1])].detach().cpu().numpy())
            if len(edges) > 0:
                edges = np.stack(edges, axis=0)
                edgecolors = np.array(edgecolors)/np.max(edgecolors)

            plot_uv(framedir, f"uv_{i:06}", full_uvs.detach().cpu().numpy(), soupfs.detach().cpu().numpy(), losses=None,
                    facecolors = np.arange(len(soupfs))/(len(soupfs)-1), edges=edges, edgecolors=edgecolors)

        # Save most recent stuffs
        torch.save(weights.detach().cpu(), os.path.join(screendir, "weights.pt"))
        torch.save(initj.detach().cpu(), os.path.join(screendir, "jacobians.pt"))

        if args.stitchweight:
            torch.save(stitchweight.detach().cpu(), os.path.join(screendir, "stitchweight.pt"))

            # Plot stitch weight as histogram
            # import matplotlib.pyplot as plt
            # fig, axs = plt.subplots()
            # # plot ours
            # axs.set_title(f"{i:06}: Stitch Weights")
            # axs.hist(stitchweight.detach().cpu().numpy(), bins=20)
            # plt.savefig(os.path.join(framedir, f"sweight_{i:06}.png"))
            # plt.close(fig)
            # plt.cla()


        with open(os.path.join(screendir, "epoch.pkl"), 'wb') as f:
            pickle.dump(i, f)

        with open(os.path.join(screendir, "lossdict.pkl"), "wb") as f:
            pickle.dump(lossdict, f)

# Plot loss curves
import matplotlib.pyplot as plt
for k, v in lossdict.items():
    fig, axs = plt.subplots()
    axs.plot(np.arange(len(v)), v)
    axs.set_title(k)

    lossname = k.replace(" ", "_").lower()
    plt.savefig(os.path.join(screendir, f"{lossname}.png"))
    plt.cla()

# Save final UVs, weights, and jacobians
np.save(os.path.join(screendir, "uv.npy"), full_uvs.detach().cpu().numpy())
np.save(os.path.join(screendir, "weights.npy"), weights.detach().cpu().numpy())
np.save(os.path.join(screendir, "targetjacobians.npy"), initj.detach().cpu().numpy())
# np.save(os.path.join(screendir, "poisjacobians.npy"), poisj.detach().cpu().numpy())
np.save(os.path.join(screendir, "stitchweight.npy"), stitchweight.detach().cpu().numpy())

# Plot final UVs
# Get cut edge pairs
edges = None
edgecolors = None
edgestitchcheck = separationloss[edgeidxs]
cutvidxs = torch.where(edgestitchcheck > args.cuteps)[0].detach().cpu().numpy()
cutsoup = [valid_edges_to_soup[vi] for vi in cutvidxs]

count = 0
edgecolors = []
edges = []
for souppair in cutsoup:
    edges.append(full_uvs[list(souppair[0])].detach().cpu().numpy())
    edges.append(full_uvs[list(souppair[1])].detach().cpu().numpy())
    edgecolors.extend([count, count])
    count += 1
if len(edges) > 0:
    edges = np.stack(edges, axis=0)
    edgecolors = np.array(edgecolors)/np.max(edgecolors)

# Show flipped triangles
from source_njf.utils import get_flipped_triangles
flipped = get_flipped_triangles(full_uvs.detach().cpu().numpy(), soupfs.detach().cpu().numpy())
flipvals = np.zeros(len(soupfs))
flipvals[flipped] = 1
lossdict = {'fliploss': flipvals}

plot_uv(screendir, f"finaluv", full_uvs.detach().cpu().numpy(), soupfs.detach().cpu().numpy(),
                    facecolors = np.arange(len(soupfs))/(len(soupfs)-1), edges=edges, edgecolors=edgecolors,
                    losses=lossdict)

# Final losses
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

finaluvs = full_uvs.detach().cpu().numpy()
tris = Triangulation(finaluvs[:, 0], finaluvs[:, 1], triangles=soupfs.detach().cpu().numpy())
edge_cmap=plt.get_cmap("gist_rainbow")

# Plot vertex separation no matter what
assert len(valid_pairs) == len(separationloss), f"Separation loss {len(separationloss)} should have entry for each of {len(valid_pairs)} pairs."

fig, axs = plt.subplots()
if args.seamless:
    separationloss = seamlessloss.detach().cpu().numpy()
else:
    separationloss = separationloss.detach().cpu().numpy()

fig.suptitle(f"Avg Vertex Loss: {np.mean(separationloss):0.8f}")
cmap = plt.get_cmap("Reds")

# Convert separation loss to per vertex
from collections import defaultdict
vlosses = defaultdict(np.double)
vlosscount = defaultdict(int)
for i in range(len(valid_pairs)):
    pair = valid_pairs[i]
    vlosses[pair[0]] += separationloss[i]
    vlosses[pair[1]] += separationloss[i]
    vlosscount[pair[0]] += 1
    vlosscount[pair[1]] += 1

# NOTE: Not all vertices will be covered in vlosses b/c they are boundary vertices
vseplosses = np.zeros(len(finaluvs))
for k, v in sorted(vlosses.items()):
    vseplosses[k] = v / vlosscount[k]

axs.tripcolor(tris, vseplosses, cmap=cmap, shading='gouraud',
                linewidth=0.5, vmin=0, vmax=0.5, edgecolor='black')

# Plot edges if given
if edges is not None:
    for i, e in enumerate(edges):
        axs.plot(e[:, 0], e[:, 1], marker='none', linestyle='-',
                    color=edge_cmap(edgecolors[i]) if edgecolors is not None else "black", linewidth=1.5)

plt.axis('off')
axs.axis("equal")
plt.savefig(os.path.join(screendir, f"finaluv_vertexseploss.png"), bbox_inches='tight', dpi=600)
plt.close()

if args.vertexseploss and args.stitchweight:
    fig, axs = plt.subplots()
    weightedseparationloss = weightedseparationloss.detach().cpu().numpy()

    fig.suptitle(f"Avg Vertex Loss: {np.mean(weightedseparationloss):0.8f}")
    cmap = plt.get_cmap("Reds")

    # Convert separation loss to per vertex
    from collections import defaultdict
    vlosses = defaultdict(np.double)
    vlosscount = defaultdict(int)
    for i in range(len(valid_pairs)):
        pair = valid_pairs[i]
        vlosses[pair[0]] += weightedseparationloss[i]
        vlosses[pair[1]] += weightedseparationloss[i]
        vlosscount[pair[0]] += 1
        vlosscount[pair[1]] += 1

    # NOTE: Not all vertices will be covered in vlosses b/c they are boundary vertices
    vseplosses = np.zeros(len(finaluvs))
    for k, v in sorted(vlosses.items()):
        vseplosses[k] = v / vlosscount[k]

    axs.tripcolor(tris, vseplosses, cmap=cmap, shading='gouraud',
                    linewidth=0.5, vmin=0, vmax=0.5, edgecolor='black')

    # Plot edges if given
    if edges is not None:
        for i, e in enumerate(edges):
            axs.plot(e[:, 0], e[:, 1], marker='none', linestyle='-',
                        color=edge_cmap(edgecolors[i]) if edgecolors is not None else "black", linewidth=1.5)

    plt.axis('off')
    axs.axis("equal")
    plt.savefig(os.path.join(screendir, f"finaluv_weightedvertexseploss.png"), bbox_inches='tight', dpi=600)
    plt.close()

if args.distortionloss:
    distortionloss = distortionloss.detach().cpu().numpy()
    fig, axs = plt.subplots(figsize=(5,5))
    fig.suptitle(f"Avg {args.distortionloss}: {np.mean(distortionloss):0.8f}")
    cmap = plt.get_cmap("Reds")
    axs.tripcolor(tris, distortionloss, facecolors=distortionloss, cmap=cmap,
                linewidth=0.5, vmin=0, vmax=1, edgecolor="black")

    # Plot edges if given
    if edges is not None:
        for i, e in enumerate(edges):
            axs.plot(e[:, 0], e[:, 1], marker='none', linestyle='-',
                        color=edge_cmap(edgecolors[i]) if edgecolors is not None else "black", linewidth=1.5)

    plt.axis('off')
    axs.axis("equal")
    plt.savefig(os.path.join(screendir, f"finaluv_{args.distortionloss}.png"), bbox_inches='tight', dpi=600)
    plt.close()

# Convert screenshots to gif
import glob
from PIL import Image

fp_in = f"{framedir}/uv_*.png"
fp_out = f"{screendir}/uv.gif"
imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]

# Resize images
imgs[0].save(fp=fp_out, format='GIF', append_images=imgs[1:],
        save_all=True, duration=20, loop=0, disposal=2)

# GIF for stitch weights if given
# if args.stitchweight:
#     fp_in = f"{framedir}/sweight_*.png"
#     fp_out = f"{screendir}/stitchweight.gif"
#     imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]

#     # Resize images
#     imgs[0].save(fp=fp_out, format='GIF', append_images=imgs[1:],
#             save_all=True, duration=20, loop=0, disposal=2)