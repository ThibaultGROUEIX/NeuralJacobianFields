### SDS Optimization through Soft Poisson
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
from PIL import Image
import torchvision
from torchvision.transforms import Resize

parser = argparse.ArgumentParser()

parser.add_argument("--debug",
                    action='store_true', default=False)
parser.add_argument("--objdir",
                    help='path to obj file to optimize for',
                    type = str, default='./data/tetrahedron.obj')
parser.add_argument("--savename",
                    help='name of experiment',
                    type = str, default=None)
parser.add_argument("--niters",
                    type = int, default=20000)
parser.add_argument("--init", type=str, choices={'slim', 'isometric','tutte'},
                    default='slim')
parser.add_argument("--vertexseptype", type=str, choices={'l2', 'l1'}, default='l2')
parser.add_argument("--edgecutloss", type=str, choices={'vertexsep', 'seamless'}, default=None,
                    help="vertexsep=use whatever outputs from vertex separation loss, seamless=use seamless l0 to approximate threshold")
parser.add_argument("--distortionloss", type=str, choices={"symmetricdirichlet", "arap"},
                    default=None)
parser.add_argument("--seamless",
                    action='store_true', default=False)
parser.add_argument("--resolution", type=int, help="render resolution", default=128)

parser.add_argument("--sdsloss", type=str, choices={'text2img', 'img2img', 'cascaded'}, default=None)
parser.add_argument("--imageloss", type=str, help="path to gt image texture if given", default=None)
parser.add_argument("--imagelossn", type=int, help="number of gt views for image loss", default=0)
parser.add_argument("--stage2weight", type=float, default=0.5)

parser.add_argument("--textureimg", type=str, default=None)
parser.add_argument('--texturetext', nargs="+", type=str, help="text description", default=None)
parser.add_argument("--interpmode", type=str, choices={'nearest', 'bilinear', 'bicubic'}, default='bilinear')

parser.add_argument("--edgeweightonly", help='optimize only edge weights. other valid pairs default to 0',
                    action='store_true', default=False)

parser.add_argument("--viziter", type=int, default=100)
parser.add_argument("--lr", type=float, default=3e-2)
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
from source_njf.utils import tutte_embedding, get_jacobian_torch, SLIM, get_local_tris, make_cut
vertices = torch.from_numpy(mesh.vertices).double().to(device)
faces = torch.from_numpy(mesh.faces).long().to(device)
fverts = vertices[faces]

ignoreset = []
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
elif args.init == "tutte":
    inituv = torch.from_numpy(tutte_embedding(mesh.vertices, mesh.faces)).to(device) # F x 2

# Get Jacobians
initj = get_jacobian_torch(vertices, faces, inituv, device=device).double() # F x 2 x 3
initj = torch.cat((initj, torch.zeros((initj.shape[0], 1, 3), device=device)), axis=1).double() # F x 3 x 3
initj.requires_grad_()

# Sanity check: jacobians return original isosoup up to global translation
pred_V = fverts @ initj[:,:2,:].transpose(2, 1)
diff = pred_V - inituv[faces]
diff -= torch.mean(diff, dim=1, keepdims=True) # Removes effect of per-triangle clobal translation
torch.testing.assert_close(diff, torch.zeros_like(diff), atol=1e-5, rtol=1e-4)
inituv = inituv.squeeze()

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
np.fill_diagonal(laplace, 0)
laplace[range(len(laplace)), range(len(laplace))] = -np.sum(laplace, axis=1)
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
from source_njf.utils import get_edge_pairs, edge_soup_correspondences
ogmesh = Mesh(ogvertices.detach().cpu().numpy(), ogfaces.detach().cpu().numpy())
valid_edge_pairs, valid_edges_to_soup, edgeidxs, edgededupidxs, edges, elens, facepairs = get_edge_pairs(ogmesh, valid_pairs, device=device)

keep_edgeidxs = []
keepidxs = []
keepelens = []
for i in range(len(edges)):
    if edges[i] not in ignoreset:
        keep_edgeidxs.append(i)
        keepidxs.append(edgeidxs[i])
        keepelens.append(elens[i])
keepelens = torch.tensor(keepelens, device=device).double()

# Initialize fixed weights (for now)
from source_njf.losses import vertexseparation, symmetricdirichlet, uvgradloss, arap
from source_njf.utils import get_jacobian_torch, clear_directory

distweight = 1
seplossdelta = args.seplossdelta
weight_idxs = (torch.tensor([pair[0] for pair in valid_pairs]).to(device), torch.tensor([pair[1] for pair in valid_pairs]).to(device))
if args.edgeweightonly:
    weight_idxs = (torch.tensor([pair[0] for pair in valid_edge_pairs]).to(device), torch.tensor([pair[1] for pair in valid_edge_pairs]).to(device))

objname = os.path.basename(datadir).split(".")[0]
if args.savename is None:
    screendir = os.path.join(savedir, objname)
else:
    screendir = os.path.join(savedir, args.savename)
Path(screendir).mkdir(parents=True, exist_ok=True)

if args.continuetrain and os.path.exists(os.path.join(screendir, "weights.pt")) and os.path.exists(os.path.join(screendir, "jacobians.pt")):
    weights = torch.load(os.path.join(screendir, "weights.pt")).double().to(device)
    initj = torch.load(os.path.join(screendir, "jacobians.pt")).double().to(device)
    theta = torch.load(os.path.join(screendir, "theta.pt")).double().to(device)
    theta.requires_grad_()

    optim = torch.optim.Adam([theta], lr=args.lr)

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

    print(f"\n============ Continuing optimization from epoch {starti} ===========\n")
else:
    clear_directory(screendir)
    weights = torch.ones(len(weight_idxs[0])).to(device).double()
    starti = 0
    lossdict = defaultdict(list)
    edgeweight = torch.ones(len(edgeidxs)).double().to(device)
    # Rotation theta
    theta = torch.tensor([np.pi/4], device=device, requires_grad=True, dtype=torch.double)
    # theta = torch.zeros(1, device=device, requires_grad=True, dtype=torch.double)
    optim = torch.optim.Adam([theta], lr=args.lr)

framedir = os.path.join(screendir, "frames")
Path(framedir).mkdir(parents=True, exist_ok=True)

# Debugging: save weights and indices
torch.save(weights, "./weights.pt")
torch.save(weight_idxs, "./weight_idxs.pt")

id = None
if args.continuetrain:
    import re
    if os.path.exists(os.path.join(screendir, 'wandb', 'latest-run')):
        for idfile in os.listdir(os.path.join(screendir, 'wandb', 'latest-run')):
            if idfile.endswith(".wandb"):
                result = re.search(r'run-([a-zA-Z0-9]+)', idfile)
                if result is not None:
                    id = result.group(1)
                    break
    else:
        print(f"Warning: No wandb record found in {os.path.join(screendir, 'wandb', 'latest-run')}!. Starting log from scratch...")

import wandb

c = wandb.wandb_sdk.wandb_artifacts.get_artifacts_cache()
c.cleanup(int(1e9))

id = None
if args.continuetrain:
    import re
    if os.path.exists(os.path.join(savedir, args.savename, 'wandb', 'latest-run')):
        for idfile in os.listdir(os.path.join(savedir, args.savename, 'wandb', 'latest-run')):
            if idfile.endswith(".wandb"):
                result = re.search(r'run-([a-zA-Z0-9]+)', idfile)
                if result is not None:
                    id = result.group(1)
                    break
    else:
        print(f"Warning: No wandb record found in {os.path.join(savedir, args.savename, 'wandb', 'latest-run')}!. Starting log from scratch...")

wandb.login()
run = wandb.init(project='conetest_opt', name=args.savename, dir=os.path.join(savedir, args.savename),
                    mode= "offline" if args.debug else "online", id = id)
wandb.define_metric("iteration")
wandb.define_metric("loss", step_metric="iteration")
wandb.define_metric("stitchingloss", step_metric="iteration")
wandb.define_metric("araploss", step_metric="iteration")

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

for iteration in range(starti, args.niters):
    # Weights need to be negative going into laplacian
    cweights = -torch.sigmoid(weights)

    ## Recompute the RHS based on new jacobians
    input = initj.transpose(2, 1).reshape(1, -1, 3) # 1 x F*3 x 3

    # Reshape the Jacobians to match the format of grad (vertex ordering STAYS THE SAME)
    P = torch.zeros(input.shape, device=device, dtype=rhs.dtype)
    k = input.shape[1] // 3
    P[:, :k, :] = input[:, ::3] # First row of all jacobians together
    P[:, k:2 * k, :] = input[:, 1::3] # Second row of all jacobians together
    P[:, 2 * k:, :] = input[:, 2::3] # Third row of all jacobians together

    input_to_solve = rhs @ P

    templaplace = laplace.clone().detach()
    templaplace[weight_idxs] = cweights # F*3 x F*3

    # Assign opposite pair
    templaplace[weight_idxs[1], weight_idxs[0]] = cweights

    # Pin vertices
    pinlaplace = templaplace
    pin_input = input_to_solve

    # Diagonal needs to be equal to sum of pair weights for each row
    pinlaplace.fill_diagonal_(0)
    pinlaplace[range(len(pinlaplace)), range(len(pinlaplace))] = -torch.sum(pinlaplace, dim=1)
    torch.testing.assert_close(torch.sum(pinlaplace, dim=1), torch.zeros(pinlaplace.shape[0], device=device, dtype=pinlaplace.dtype), atol=1e-4, rtol=1e-4)

    # Solve with pinned vertices
    uvs = torch.linalg.solve(pinlaplace[1:,1:], pin_input[0, 1:]) # F*3 x 3
    uvs = uvs[:,:2]

    # Rotate using rotmatrix
    rotmatrix = torch.cat([torch.cos(theta), -torch.sin(theta), torch.sin(theta), torch.cos(theta)]).reshape(2, 2)
    uvs = uvs @ rotmatrix

    # Add back pinned vertex
    full_uvs = torch.cat([torch.zeros(1, 2, device=device), uvs], dim=0)

    ### Losses
    loss = 0

    # NOTE: Order is wrt weights order (both follow vpairs)
    # TODO: New edge separation loss should be avg(L2 of corresponding vertices) (E x 1 array)
    # TODO: Just choose the functional form of the edge separation loss such that GT is lower energy than the local minima!
    separationloss = torch.sum(vertexseparation(ogvertices, ogfaces, full_uvs, loss='l2'), dim=1)
    separationloss = torch.sqrt(separationloss)

    # Continual L0
    if args.seamless:
        seamlessloss = (separationloss * separationloss)/(separationloss * separationloss + seplossdelta)

    if args.edgecutloss:
        # Remove the cut from the loss
        edgecutloss = separationloss[keepidxs] * keepelens/torch.sum(keepelens)
        loss += torch.mean(edgecutloss)
        lossdict['Edge Cut Loss'].append(edgecutloss.detach().cpu().numpy())

    poisj = get_jacobian_torch(soupvs, soupfs, full_uvs, device=device)

    if args.distortionloss:
        # Distortion loss on poisson solved jacobians
        if args.distortionloss == "symmetricdirichlet":
            distortionloss = symmetricdirichlet(ogvertices, ogfaces, poisj,
                                                )
        if args.distortionloss == "arap":
            distortionloss = arap(ogvertices, ogfaces, full_uvs, paramtris=full_uvs[soupfs], device=device,
                                  return_face_energy=True)

        loss += distweight * torch.mean(distortionloss)
        lossdict['Distortion Loss'].append(distweight * distortionloss.detach().cpu().numpy())

    # SDS loss
    # Prereqs: texture image, texture description
    if args.sdsloss:
        assert args.textureimg is not None and args.texturetext is not None, "Need to specify textureimg and texturetext for SDS loss"
        from source_njf.diffusion_guidance.deepfloyd_if import DeepFloydIF, DeepFloydIF_Img2Img
        from source_njf.diffusion_guidance.deepfloyd_cascaded import DeepFloydCascaded, DiffusionConfig

        if args.sdsloss == "text2img":
            diffusion = DeepFloydIF() # optionally you can pass a config at initialization
        elif args.sdsloss == "img2img":
            diffusion = DeepFloydIF_Img2Img()
        elif args.sdsloss == "cascaded":
            cfg = DiffusionConfig()
            diffusion = DeepFloydCascaded(cfg)

        # TODO: img2img
        # Text encoding
        texturetext = ' '.join(args.texturetext)
        text_z, text_z_neg = diffusion.encode_prompt(texturetext)

        from PIL import Image
        from torchvision.transforms.functional import pil_to_tensor
        textureimg = pil_to_tensor(Image.open(args.textureimg)).double().to(device)

        rgb_images = []
        # NOTE: Can upscale resolution to get better gradients
        from source_njf.renderer import render_texture
        num_views = 5
        radius = 2.5
        center = torch.zeros(2)
        azim = torch.linspace(center[0], 2 * np.pi + center[0], num_views + 1)[
            :-1].double().to(device)
        elev = torch.zeros(len(azim), device=device).double()

        # Face UVs
        uv_face = full_uvs.reshape(-1, 3, 2) # F x 3 x 2

        # Need to scale UVs between 0-1
        from source_njf.utils import normalize_uv
        with torch.no_grad():
            normalize_uv(uv_face)

        rgb_images.append(render_texture(vertices.double(), faces, uv_face, elev, azim, radius, textureimg/255, lights=None,
                                                resolution=(args.resolution, args.resolution), device=device, lookatheight=0, whitebg=True,
                                                interpolation_mode='bilinear'))
        ## Debugging
        # import matplotlib.pyplot as plt

        # images = rgb_images[0]['image'].detach().cpu().numpy()
        # num_views = 5
        # fig, axs = plt.subplots(int(np.ceil(num_views/5)), num_views)
        # for nview in range(num_views):
        #     j = nview % 5
        #     if nview > 5:
        #         i = nview // 5
        #         axs[i,j].imshow(images[nview].transpose(1,2,0))
        #         axs[i,j].axis('off')
        #     else:
        #         axs[j].imshow(images[nview].transpose(1,2,0))
        #         axs[j].axis('off')
        # plt.axis('off')
        # fig.suptitle(f"Current Textures")
        # plt.savefig(os.path.join(framedir, f"test.png"))
        # plt.close(fig)
        # plt.cla()

        if args.sdsloss == "text2img":
            sds = diffusion(rgb_images[0]['image'], prompt_embeds = text_z)
            sdsloss = sds['loss_sds']
            loss += sdsloss
            lossdict['SDS Loss'].append(sdsloss.item())
        elif args.sdsloss == "cascaded":
            sds = diffusion(rgb_images[0]['image'], prompt_embeds = text_z, stage = 'I')
            sdsloss = sds['loss_sds']
            loss += sdsloss
            lossdict['SDS Stage 1'].append(sdsloss.item())

            sds = diffusion(rgb_images[0]['image'], prompt_embeds = text_z, stage = 'II')
            sdsloss = sds['loss_sds']
            loss += args.stage2weight * sdsloss
            lossdict['SDS Stage 2'].append(sdsloss.item())

    if args.imageloss:
        from PIL import Image
        from torchvision.transforms.functional import pil_to_tensor
        textureimg = pil_to_tensor(Image.open(args.textureimg)).double().to(device)
        rgb_images = []

        from source_njf.renderer import render_texture
        total_views = 5
        num_views = args.imagelossn
        radius = 2.5
        center = torch.zeros(2)
        azim = torch.linspace(center[0], 2 * np.pi + center[0], total_views + 1)[
            :-1].double().to(device)
        elev = torch.zeros(len(azim), device=device).double()

        # Subset to the number of views input
        azim = azim[:num_views]
        elev = elev[:num_views]

        # Face UVs
        uv_face = full_uvs.reshape(-1, 3, 2) # F x 3 x 2

        # Need to scale UVs between 0-1
        from source_njf.utils import normalize_uv
        with torch.no_grad():
            normalize_uv(uv_face)

        pred_images = render_texture(vertices.double(), faces, uv_face, elev, azim, radius, textureimg/255, lights=None,
                                                resolution=(args.resolution, args.resolution), device=device, lookatheight=0, whitebg=True,
                                                interpolation_mode = args.interpmode)
        rgb_images.append(pred_images)

        gt_images = []
        for i in range(num_views):
            gt_image = torchvision.io.read_image(args.imageloss + f"_{i}.png").double().to(device)
            gt_image = Resize((args.resolution, args.resolution))(gt_image)/255
            gt_images.append(gt_image)
        gt_images = torch.stack(gt_images, dim=0)
        imageloss = torch.nn.functional.mse_loss(rgb_images[0]['image'], gt_images, reduction="none")
        loss += torch.mean(imageloss)
        lossdict['Image Loss'].append(imageloss.cpu().detach().numpy())

    # Compute loss
    loss.backward()
    lossdict['Total Loss'].append(loss.item())

    optim.step()
    optim.zero_grad()

    print(f"========== Done with iteration {iteration}. ==========")
    lossstr = ""
    for k, v in lossdict.items():
        lossstr += f"{k}: {np.mean(v[-1]):0.7f}. "
    print(lossstr)

    wandb.log({'total_loss': lossdict['Total Loss'][-1]}, step=iteration, commit=True)
    if args.edgecutloss:
        wandb.log({'stitching_loss': lossdict['Edge Cut Loss'][-1]}, step=iteration, commit=True)
    if args.distortionloss:
        wandb.log({'arap_loss': lossdict['Distortion Loss'][-1]}, step=iteration, commit=True)
    if args.sdsloss:
        if args.sdsloss == "cascaded":
            wandb.log({'sds_loss_1': lossdict['SDS Stage 1'][-1]}, step=iteration, commit=True)
            wandb.log({'sds_loss_2': lossdict['SDS Stage 2'][-1]}, step=iteration, commit=True)
        else:
            wandb.log({'sds_loss': lossdict['SDS Loss'][-1]}, step=iteration, commit=True)
    if args.imageloss:
        wandb.log({'image_loss': np.mean(lossdict['Image Loss'][-1])}, step=iteration, commit=True)

    ogvs = ogvertices.detach().cpu().numpy()
    ogfs = ogfaces.detach().cpu().numpy()
    ogvsoup = ogvs[ogfs].reshape(-1, 3)

    # Visualize every viziter epochs
    if iteration % args.viziter == 0:
        # Convert loss dict keys
        lossdict_convert = {}
        if args.edgecutloss:
            lossdict_convert['Edge Cut Loss'] = 'edgecutloss'
        if args.distortionloss:
            lossdict_convert['Distortion Loss'] = 'distortionloss'

        lossdict_viz = {}
        for key, item in lossdict.items():
            if key in lossdict_convert:
                lossdict_viz[lossdict_convert[key]] = item[-1]

        if not cluster:
            ps.init()
            ps.remove_all_structures()
            # ps_mesh = ps.register_surface_mesh("mesh", soupvs.detach().cpu().numpy(), soupfs.detach().cpu().numpy(), edge_width=1)
            ps_uv = ps.register_surface_mesh("uv mesh", full_uvs.detach().cpu().numpy(), soupfs.detach().cpu().numpy(), edge_width=1)
            ps_uv.add_scalar_quantity('corr', np.arange(len(soupfs)), defined_on='faces', enabled=True, cmap='viridis')
            ps.screenshot(os.path.join(framedir, f"uv_{iteration:06}.png"), transparent_bg=True)
        else:
            # Get cut edge pairs
            edges = None
            edgecolors = None

            edgestitchcheck = separationloss[edgeidxs].detach().cpu().numpy()
            cutvidxs = np.where(edgestitchcheck > args.cuteps)[0]
            cutsoup = [valid_edges_to_soup[vi] for vi in cutvidxs]

            edgecolors = np.repeat(cutvidxs, 2)/(len(edgestitchcheck)-1)
            edges = []
            for souppair in cutsoup:
                edges.append(full_uvs[list(souppair[0])].detach().cpu().numpy())
                edges.append(full_uvs[list(souppair[1])].detach().cpu().numpy())
            if len(edges) > 0:
                edges = np.stack(edges, axis=0)
                edgecolors = np.array(edgecolors)/np.max(edgecolors)

            plot_uv(framedir, f"uv_{iteration:06}", full_uvs.detach().cpu().numpy(), soupfs.detach().cpu().numpy(),
                    losses=lossdict_viz,
                    facecolors = np.arange(len(soupfs))/(len(soupfs)-1), edges=edges, edgecolors=edgecolors)

            imagepaths = [os.path.join(framedir, f"uv_{iteration:06}.png")] + \
                        [os.path.join(framedir, f"{key}_uv_{iteration:06}.png") for key in lossdict_viz.keys() if "loss" in key]
            images = [wandb.Image(Image.open(x)) for x in imagepaths]
            wandb.log({'uvs': images}, commit=True)

            if args.sdsloss or args.imageloss:
                import matplotlib.pyplot as plt

                images = rgb_images[0]['image'].detach().cpu().numpy()
                num_views = len(images)
                fig, axs = plt.subplots(int(np.ceil(num_views/5)), num_views)
                if num_views == 1:
                    axs.imshow(images[0].transpose(1,2,0))
                    axs.axis('off')
                else:
                    for nview in range(num_views):
                        j = nview % 5
                        if nview > 5:
                            i = nview // 5
                            axs[i,j].imshow(images[nview].transpose(1,2,0))
                            axs[i,j].axis('off')
                        else:
                            axs[j].imshow(images[nview].transpose(1,2,0))
                            axs[j].axis('off')
                plt.axis('off')
                fig.suptitle(f"Epoch {iteration} Textures")
                plt.savefig(os.path.join(framedir, f"texture_{iteration:06}.png"))
                plt.close(fig)
                plt.cla()

                # Log the plotted imgs
                images = [wandb.Image(Image.open(os.path.join(framedir, f"texture_{iteration:06}.png")))]
                wandb.log({'textures': images}, commit=True)

            # Plot the image loss image
            if args.imageloss:
                import matplotlib.pyplot as plt

                fig, axs = plt.subplots()
                imageloss = (lossdict['Image Loss'][0] * 255).astype(np.uint8).transpose(0,2,3,1)
                num_views = imageloss.shape[0]
                fig, axs = plt.subplots(int(np.ceil(num_views/5)), num_views)
                if num_views == 1:
                    axs.imshow(Image.fromarray(imageloss[0]))
                    axs.axis('off')
                else:
                    for nview in range(num_views):
                        j = nview % 5
                        if nview > 5:
                            i = nview // 5
                            axs[i,j].imshow(Image.fromarray(imageloss[nview]))
                            axs[i,j].axis('off')
                        else:
                            axs[j].imshow(Image.fromarray(imageloss[nview]))
                            axs[j].axis('off')
                plt.axis('off')
                fig.suptitle(f"Epoch {iteration} Image Loss")
                plt.savefig(os.path.join(framedir, f"imageloss_{iteration:06}.png"))
                plt.close()
                plt.cla()

                # Log the plotted imgs
                images = [wandb.Image(Image.open(os.path.join(framedir, f"imageloss_{iteration:06}.png")))]
                wandb.log({'imageloss': images}, commit=True)

            # Mesh Energies
            for key, val in lossdict_viz.items():
                if "loss" in key: # Hacky way of avoiding aggregated values
                    if key == "edgecutloss":
                        from collections import defaultdict
                        valid_edges_to_soup = [valid_edges_to_soup[i] for i in keep_edgeidxs]
                        edgecutloss = val # edgeidxs x 1

                        ### Roadmap
                        # For each og edge: get avg of the 2 corresponding edge cut vals + get the two soup edges
                        soupedgedict = defaultdict(list)
                        for i in range(len(valid_edges_to_soup)):
                            soupkey = frozenset((frozenset(valid_edges_to_soup[i][0]), frozenset(valid_edges_to_soup[i][1])))
                            edgevals = soupedgedict[soupkey]
                            if len(edgevals) >= 2:
                                raise ValueError("More than 2 valid edges associated with souppair!")

                            # NOTE: Only need this check because sometimes we test what happens when we remove edges from loss
                            edgevals.append(edgecutloss[i])

                        # Plot edges if given
                        cylindervals = []
                        cylinderpos = []
                        for soupkey, edgecutval in soupedgedict.items():
                            p0 = list(list(soupkey)[0])
                            edgeval = np.mean(edgecutval)
                            cylindervals.append([edgeval, edgeval])
                            cylinderpos.append(ogvsoup[p0])
                        cylinderpos = np.stack(cylinderpos, axis=0)
                        cylindervals = np.array(cylindervals)

                        export_views(ogvs, ogfs, framedir, filename=f"{key}_mesh_{iteration:06}.png",
                                    plotname=f"Avg {key}: {np.mean(val):0.4f}", cylinders=cylinderpos,
                                    cylinder_scalars=cylindervals, outline_width=0.01,
                                    device="cpu", n_sample=30, width=200, height=200,
                                    vmin=0, vmax=0.1, shading=False)
                    else:
                        export_views(ogvertices.detach().cpu().numpy(), ogfaces.detach().cpu().numpy(),
                                     framedir, filename=f"{key}_mesh_{iteration:06}.png",
                                        plotname=f"Avg {key}: {np.mean(val):0.4f}",
                                        fcolor_vals=val, device="cpu", n_sample=30, width=200, height=200,
                                        vmin=0, vmax=0.6, shading=False)

            imagepaths = [os.path.join(framedir, f"{key}_mesh_{iteration:06}.png") for key in lossdict_viz.keys() if "loss" in key]
            if len(imagepaths) > 0:
                images = [wandb.Image(Image.open(x)) for x in imagepaths]
                wandb.log({'mesh energies': images}, commit=True)

            # Weights
            # NOTE: below results in 2x each cylinder but it's fine
            cweights_detach = -cweights.detach().cpu().numpy()
            edgecorrespondences, facecorrespondences = edge_soup_correspondences(ogfaces.detach().cpu().numpy())
            soupvedge_to_ogvedge = {}
            for k, v in edgecorrespondences.items():
                if len(v) > 1:
                    soupvedge_to_ogvedge[tuple(sorted([v[0][0], v[1][0]]))] = list(k)
                    soupvedge_to_ogvedge[tuple(sorted([v[0][1], v[1][1]]))] = list(k)

            cylinderpos = []
            for vpair in valid_edge_pairs:
                cylinderpos.append(ogvs[soupvedge_to_ogvedge[tuple(sorted(vpair))]])
            cylinderpos = np.stack(cylinderpos, axis=0)
            cylindervals = np.stack([cweights_detach, cweights_detach], axis=1) # len(edgeidxs) = # edges x 2

            export_views(ogvertices.detach().cpu().numpy(), ogfaces.detach().cpu().numpy(),
                         framedir, filename=f"weights_mesh_{iteration:06}.png",
                        plotname=f"Edge Weights", cylinders=cylinderpos,
                        cylinder_scalars=cylindervals,
                        outline_width=0.01, cmap = plt.get_cmap('Reds_r'),
                        device="cpu", n_sample=30, width=200, height=200,
                        vmin=0, vmax=1, shading=False)

            # Histogram of weights
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots()
            # plot ours
            axs.set_title(f"Iteration {iteration:06}: SP Weights")
            axs.hist(cweights_detach, bins=20)
            plt.savefig(os.path.join(framedir, f"weights_{iteration:06}.png"))
            plt.close(fig)
            plt.cla()

            imagepaths = [os.path.join(framedir, f"weights_mesh_{iteration:06}.png"), os.path.join(framedir, f"weights_{iteration:06}.png")]
            images = [wandb.Image(Image.open(x)) for x in imagepaths]
            wandb.log({'edge weights': images}, step=iteration, commit=True)

        # Save most recent stuffs
        torch.save(weights.detach().cpu(), os.path.join(screendir, "weights.pt"))
        torch.save(initj.detach().cpu(), os.path.join(screendir, "jacobians.pt"))
        torch.save(rotmatrix.detach().cpu(), os.path.join(screendir, "rotmatrix.pt"))
        torch.save(theta.detach().cpu(), os.path.join(screendir, "theta.pt"))

        with open(os.path.join(screendir, "epoch.pkl"), 'wb') as f:
            pickle.dump(i, f)

        with open(os.path.join(screendir, "lossdict.pkl"), "wb") as f:
            pickle.dump(lossdict, f)

# Plot loss curves
# import matplotlib.pyplot as plt
# for k, v in lossdict.items():
#     fig, axs = plt.subplots()
#     axs.plot(np.arange(len(v)), v)
#     axs.set_title(k)

#     lossname = k.replace(" ", "_").lower()
#     plt.savefig(os.path.join(screendir, f"{lossname}.png"))
#     plt.cla()

# Save final UVs, weights, and jacobians
np.save(os.path.join(screendir, "uv.npy"), full_uvs.detach().cpu().numpy())
np.save(os.path.join(screendir, "weights.npy"), weights.detach().cpu().numpy())
np.save(os.path.join(screendir, "targetjacobians.npy"), initj.detach().cpu().numpy())
np.save(os.path.join(screendir, "theta.npy"), theta.detach().cpu().numpy())
# np.save(os.path.join(screendir, "poisjacobians.npy"), poisj.detach().cpu().numpy())

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

# Convert screenshots to gif
# import glob
# from PIL import Image

# fp_in = f"{framedir}/uv_*.png"
# fp_out = f"{screendir}/uv.gif"
# imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]

# # Resize images
# imgs[0].save(fp=fp_out, format='GIF', append_images=imgs[1:],
#         save_all=True, duration=20, loop=0, disposal=2)

# # Wandb log
# wandb.log({"uv opt": wandb.Video(f"{screendir}/uv.gif", fps=30, format='gif')})