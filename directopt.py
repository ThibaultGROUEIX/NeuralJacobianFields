### Direct optimization code
import torch
from source_njf.losses import uvseparation, symmetricdirichlet, uvgradloss, splitgradloss
from results_saving_scripts.plot_uv import plot_uv, export_views
from tqdm import tqdm
from meshing.mesh import Mesh
from meshing.analysis import computeFacetoEdges
from pathlib import Path
import os
import numpy as np
from source_njf.utils import clear_directory
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--savedir",
                    help='path of savedir',
                    type = str, required=True)
parser.add_argument("--vs",
                    type = str, required=True)
parser.add_argument("--fs",
                    type = str,required=True)
parser.add_argument("--init",
                    type = str,required=True)
parser.add_argument("--n_viz",
                    type = int, default = 20)
parser.add_argument("--edgesepdelta",
                    type = float, default = 0.1)
parser.add_argument("--edgesepdelta_min",
                    type = float, default = 0.001)
parser.add_argument("--lr",
                    type = float, default = 1e-3)
parser.add_argument("--grad", choices={'l2', 'split'},
                    help='whether to use gradient stitching energy instead of standard edge separation (will still optimize translation with edge separation)',
                    default=None)
parser.add_argument("--gradrelax", action="store_true")
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--continuetrain", action="store_true")
parser.add_argument("--anneal", action="store_true")
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

EXPORT_FOLDER = args.savedir
Path(EXPORT_FOLDER).mkdir(exist_ok=True, parents=True)

if args.overwrite:
    clear_directory(EXPORT_FOLDER)

VIS_EPOCH = args.n_viz
MAX_DELTA = args.edgesepdelta
MIN_DELTA = args.edgesepdelta_min
DELTA_COSINE_FREQ = 250
NUM_EPOCHS = 20000
if args.debug:
    NUM_EPOCHS=10
ANNEALING = args.anneal
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

vertices = torch.load(args.vs).to(device)
faces = torch.load(args.fs).to(device)
init = torch.load(args.init).to(device).float()

mesh = Mesh(vertices.cpu().numpy(), faces.cpu().numpy())

LOADED_STATE = False
if args.continuetrain:
    statepath = os.path.join(EXPORT_FOLDER, 'state.pt')
    if os.path.exists(statepath):
        state = torch.load(statepath)
        predj = state['predJ'].to(device)
        predj.requires_grad_()
        predtrans = state['predtrans'].to(device)
        predtrans.requires_grad_()

        # Load optimizer and scheduler states
        optimizer = torch.optim.Adam([predj], lr=args.lr)
        optimizer.add_param_group({"params": [predtrans], 'lr': args.lr})

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=500, min_lr=1e-7,
                                                            factor=0.5,
                                                            threshold=0.0001, verbose=True)
        scheduler.load_state_dict(state['scheduler'])
        optimizer.load_state_dict(state['optimizer'])
        START_EPOCH = state['epoch']
        LOADED_STATE = True

        print(f"Loaded state from epoch {START_EPOCH}.")
    else:
        print(f"No state found for {statepath}! Starting from scratch...")

if not LOADED_STATE:
    predj = torch.stack([torch.eye(2)] * len(init)).to(device).float()
    predj.requires_grad_()
    predtrans = torch.zeros((len(init), 2)).to(device).float()
    predtrans.requires_grad_()
    optimizer = torch.optim.Adam([predj], lr=args.lr)
    optimizer.add_param_group({"params": [predtrans], 'lr': args.lr})

    # NOTE: Patience must be smaller than the cosine schedule!
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=500, min_lr=1e-7,
                                                        factor=0.5,
                                                        threshold=0.0001, verbose=True)
    START_EPOCH = 0

# Face to edges map
ftoe = computeFacetoEdges(mesh)

# Remap indexing to ignore the boundaries
neweidx = []
oldeidx = []
count = 0
for key, edge in sorted(mesh.topology.edges.items()):
    if not edge.onBoundary():
        neweidx.append(count)
        oldeidx.append(edge.index)
        count += 1
print(f"{np.min(neweidx)}, {np.max(neweidx)}")
print(f"{np.min(oldeidx)}, {np.max(oldeidx)}")

ebdtoe = np.zeros(np.max(list(mesh.topology.edges.keys())) + 1)
ebdtoe[oldeidx] = neweidx
ebdtoe = ebdtoe.astype(int)

new_ftoe = []
for es in ftoe:
    new_ftoe.append(ebdtoe[es])
ftoe = new_ftoe
print(f"{np.min(np.array([e for es in ftoe for e in es]))}, {np.max(np.array([e for es in ftoe for e in es]))}")

CURRENT_DELTA = MAX_DELTA
for i in (pbar := tqdm(range(START_EPOCH, NUM_EPOCHS))):
    if ANNEALING:
        # Update edgesep delta based on cosine annealing schedule
        CURRENT_DELTA = MIN_DELTA + 0.5 * (MAX_DELTA - MIN_DELTA) * (1 + np.cos((i % DELTA_COSINE_FREQ)/DELTA_COSINE_FREQ * np.pi))

    optimizer.zero_grad()
    pred_V = torch.einsum("bcd,bde->bce", (predj, init.transpose(1,2))).transpose(1,2) # F x 3 x 2

    # Compute losses
    if args.grad == "l2":
        gradloss = torch.sum(uvgradloss(vertices, faces, pred_V), dim=1)
        edgeloss = torch.sum(uvseparation(vertices, faces, pred_V.detach() + predtrans.unsqueeze(1), loss='l2'), dim=[1,2])
        if args.gradrelax:
            edgeloss = edgeloss * edgeloss/(edgeloss * edgeloss + CURRENT_DELTA)
    elif args.grad == 'split':
        gradloss = splitgradloss(vertices, faces, pred_V)
        edgeloss = torch.sum(uvseparation(vertices, faces, pred_V.detach() + predtrans.unsqueeze(1), loss='l2'), dim=[1,2])
        if args.gradrelax:
            edgeloss = edgeloss * edgeloss/(edgeloss * edgeloss + CURRENT_DELTA)
    else:
        pred_V += predtrans.unsqueeze(1)
        edgeloss = uvseparation(vertices, faces, pred_V)
        edgeloss = edgeloss * edgeloss/(edgeloss * edgeloss + CURRENT_DELTA)
        edgeloss = torch.sum(edgeloss, dim=[1,2]) # E x 1

    distloss = symmetricdirichlet(vertices, faces, predj)
    loss = torch.mean(edgeloss) + torch.mean(distloss)
    if args.grad:
        loss += torch.mean(gradloss)
    loss.backward()
    optimizer.step()
    scheduler.step(loss)

    if i % VIS_EPOCH == 0:
        with torch.no_grad():
            # Edge loss to face losses
            ftoeloss = torch.tensor([torch.sum(edgeloss[es]) for es in ftoe]).detach().cpu().numpy()
            lossdict = {}
            lossdict['edgeseploss'] = ftoeloss
            lossdict['symdirloss'] = distloss.detach().cpu().numpy()
            if args.grad:
                fgradloss = torch.tensor([torch.sum(gradloss[es]) for es in ftoe]).detach().cpu().numpy()
                lossdict['gradloss'] = fgradloss

            # Plot 2D embeddings
            # Need to flatten pred v and assign new triangles
            triangles = np.arange(len(faces)*3).reshape(len(faces), 3)
            uv = pred_V.reshape(-1, 2).detach().cpu().numpy()
            plot_uv(EXPORT_FOLDER, f"{i:06}: Loss {loss.item():0.4f}", uv, triangles, losses=lossdict)

            # Plot edgeloss on 3D mesh
            export_views(mesh, EXPORT_FOLDER, filename=f"mesh_eloss_{i:06}.png",
                        plotname=f"Edge Separation Loss: {torch.sum(edgeloss).item():0.4f}",
                        fcolor_vals=ftoeloss, device="cpu", n_sample=24, width=200, height=200)

            export_views(mesh, EXPORT_FOLDER, filename=f"mesh_dloss_{i:06}.png",
                        plotname=f"Symmetric Dirichlet Loss: {torch.sum(distloss).item():0.4f}",
                        fcolor_vals=distloss.detach().cpu().numpy(), device="cpu", n_sample=24, width=200, height=200)

            if args.grad:
                export_views(mesh, EXPORT_FOLDER, filename=f"mesh_gradloss_{i:06}.png",
                        plotname=f"Symmetric Dirichlet Loss: {torch.sum(distloss).item():0.4f}",
                        fcolor_vals=fgradloss, device="cpu", n_sample=24, width=200, height=200)


            # Save current state
            statedict = {'epoch': i+1,
                         'predJ': predj.detach().cpu(),
                         'predtrans': predtrans.detach().cpu(),
                         'optimizer': optimizer.state_dict(),
                         'scheduler': scheduler.state_dict()}
            torch.save(statedict, os.path.join(EXPORT_FOLDER, "state.pt"))

            # Save latest predictions
            torch.save(predj.detach().cpu(), os.path.join(EXPORT_FOLDER, 'latestpredj.pt'))
            torch.save(predtrans.detach().cpu(), os.path.join(EXPORT_FOLDER, 'latestpredtrans.pt'))
            np.save(os.path.join(EXPORT_FOLDER, 'latestuv.pt'), uv)
            np.save(os.path.join(EXPORT_FOLDER, 'latestfaces.pt'), triangles)

            print(f"Saved checkpoint at epoch {i}.")


    # If debugging, check memory
    if args.debug:
        if torch.cuda.is_available():
            # Check memory consumption
            # Get GPU memory usage
            t = torch.cuda.get_device_properties(0).total_memory
            r = torch.cuda.memory_reserved(0)
            a = torch.cuda.memory_allocated(0)
            m = torch.cuda.max_memory_allocated(0)
            f = r-a  # free inside reserved
            print(f"{a/1024**3:0.3f} GB allocated. \nGPU max memory alloc: {m/1024**3:0.3f} GB. \nGPU total memory: {t/1024**3:0.3f} GB.")

        # Get CPU RAM usage too
        import psutil
        print(f'RAM memory % used: {psutil.virtual_memory()[2]}')

    pbar.set_description(f"Loss: {loss.item():0.4f}. Current delta: {CURRENT_DELTA:0.4f}. Current LR: {optimizer.param_groups[0]['lr']:0.5f}")

# Save final predictions
triangles = np.arange(len(faces)*3).reshape(len(faces), 3)
if args.grad:
    uv = (pred_V + predtrans.unsqueeze(1)).detach().cpu()
else:
    uv = pred_V.reshape(-1, 2).detach().cpu()
torch.save(predtrans.detach().cpu(), os.path.join(EXPORT_FOLDER, 'finalpredtrans.pt'))
torch.save(predj.detach().cpu(), os.path.join(EXPORT_FOLDER, 'finalpredj.pt'))
np.save(os.path.join(EXPORT_FOLDER, 'finaluv.pt'), uv)
np.save(os.path.join(EXPORT_FOLDER, 'finalfaces.pt'), triangles)

# Save GIFs
from PIL import Image
import glob
import re
## Default UV gif
fp_in = f"{EXPORT_FOLDER}/*.png"
fp_out = f"{EXPORT_FOLDER}/train.gif"
imgs = [Image.open(f) for f in sorted(glob.glob(fp_in)) if re.search(r'/(\d+).*\.png', f)]

# Resize images
basewidth = 400
wpercent = basewidth/imgs[0].size[0]
newheight = int(wpercent * imgs[0].size[1])
imgs = [img.resize((basewidth, newheight)) for img in imgs]

imgs[0].save(fp=fp_out, format='GIF', append_images=imgs[1:],
        save_all=True, duration=100, loop=0, disposal=2)

## Individual losses
lossnames = ['edgeseploss', 'symdirloss']
if args.grad:
    lossnames.append('gradloss')
for key in lossnames:
    if "loss" in key:
        # Embedding viz
        fp_in = f"{EXPORT_FOLDER}/{key}*.png"
        fp_out = f"{EXPORT_FOLDER}/train_{key}.gif"
        imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]

        # Resize images
        basewidth = 400
        wpercent = basewidth/imgs[0].size[0]
        newheight = int(wpercent * imgs[0].size[1])
        imgs = [img.resize((basewidth, newheight)) for img in imgs]

        imgs[0].save(fp=fp_out, format='GIF', append_images=imgs[1:],
                save_all=True, duration=100, loop=0, disposal=2)