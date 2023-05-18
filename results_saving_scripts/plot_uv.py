import matplotlib.pyplot as plt
import numpy as np
import os
import fresnel

def plot_uv(path, name, pred_vertices, triangles, gt_vertices=None, losses=None, cmin=0, cmax=0.6, logger=None,
            ftoe=None):
    # First center the predicted vertices
    pred_vertices -= np.mean(pred_vertices, 0, keepdims=True) # Sum batched over faces, vertex dimension

    # setup side by side plots
    fname = "_".join(name.split())
    if gt_vertices:
        fig, axs = plt.subplots(1,2,figsize=(15, 4))
        fig.suptitle(name)

        # plot GT
        axs[0].set_title('GT')
        axs[0].axis('equal')

        axs[0].triplot(gt_vertices[:,0], gt_vertices[:,1], triangles, linewidth=0.5)

        # plot ours
        axs[1].set_title('Ours')
        axs[1].axis('equal')
        # axs[1].set_axis_off()

        axs[1].triplot(pred_vertices[:,0], pred_vertices[:,1], triangles, linewidth=0.5)
        plt.axis('off')
        plt.savefig(os.path.join(path, f"{fname}.png"))
        plt.close(fig)
        plt.cla()
    else:
        fig, axs = plt.subplots(figsize=(5, 5))
        # plot ours
        axs.set_title(name)
        axs.tripcolor(pred_vertices[:, 0], pred_vertices[:, 1], triangles=triangles, facecolors=np.ones(len(triangles)) * 0.5,
                                linewidth=0.5, edgecolor="black")
        plt.axis('off')
        axs.axis('equal')
        plt.savefig(os.path.join(path, f"{fname}.png"))
        plt.close(fig)
        plt.cla()

    # # NOTE: this only works with WandbLogger
    # if logger is not None:
    #     logger.experiment.log({fname: wandb.Image(os.path.join(path, f"{fname}.png"))})

    # Plot losses
    if losses is not None:
        for key, val in losses.items():
            if "loss" in key: # Hacky way of avoiding aggregated values
                if "edge" in key and ftoe is not None:
                    ftoeloss = np.array([np.sum(val[es]) for es in ftoe])
                    fig, axs = plt.subplots(figsize=(5, 5))
                    fig.suptitle(f"{name}\nAvg {key}: {np.mean(val):0.4f}")
                    cmap = plt.get_cmap("Reds")
                    axs.tripcolor(pred_vertices[:, 0], pred_vertices[:, 1], ['black'] * len(pred_vertices), triangles=triangles, cmap=cmap,
                                linewidth=0.5, vmin=cmin, vmax=cmax, facecolors=ftoeloss, edgecolor='black')
                    plt.axis('off')
                    axs.axis("equal")
                    plt.savefig(os.path.join(path, f"{key}_{fname}.png"), bbox_inches='tight',dpi=600)
                    plt.close()
                else:
                    fig, axs = plt.subplots(figsize=(5,5))
                    fig.suptitle(f"{name}\nAvg {key}: {np.mean(val):0.4f}")
                    cmap = plt.get_cmap("Reds")
                    axs.tripcolor(pred_vertices[:, 0], pred_vertices[:, 1], triangles=triangles, facecolors=val[:len(triangles)], cmap=cmap,
                                linewidth=0.5, vmin=cmin, vmax=cmax, edgecolor="black")
                    plt.axis('off')
                    axs.axis("equal")
                    plt.savefig(os.path.join(path, f"{key}_{fname}.png"), bbox_inches='tight',dpi=600)
                    plt.close()

                # if logger is not None:
                #     logger.experiment.log({f"{key}_{fname}": wandb.Image(os.path.join(path, f"{key}_{fname}.png"))})

def export_views(mesh, savedir, n=5, n_sample=20, width=150, height=150, plotname="Views", filename="test", fcolor_vals=None,
                 vcolor_vals=None,
                 device="cpu", outline_width=0.005, cmap= plt.get_cmap("Reds"), vmin=0, vmax=1):
    import torch
    import matplotlib as mpl
    from matplotlib import cm

    fresnel_device = fresnel.Device(mode=device)
    scene = fresnel.Scene(device=fresnel_device)
    vertices, faces, _ = mesh.export_soup()
    fverts = vertices[faces].reshape(3 * len(faces), 3)
    mesh = fresnel.geometry.Mesh(scene, vertices=fverts, N=1)
    mesh.material = fresnel.material.Material(color=fresnel.color.linear([0.25, 0.5, 0.9]), roughness=0.1)

    # NOTE: outline width 0 => outline off
    mesh.outline_material = fresnel.material.Material(color=(0., 0., 0.), roughness=0.1, metal=1.)
    mesh.outline_width=outline_width

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    scalarmap = cm.ScalarMappable(norm=norm, cmap=cmap)
    if fcolor_vals is not None:
        # Each face gets own set of 3 vertex colors
        fcolors = scalarmap.to_rgba(fcolor_vals)[:,:3]
        vcolors = [color for color in fcolors for _ in range(3)]
        vcolors = np.vstack(vcolors)

        mesh.color[:] = fresnel.color.linear(vcolors)
        mesh.material.primitive_color_mix = 1.0
    elif vcolor_vals is not None:
        vcolors = scalarmap.to_rgba(vcolor_vals)[:,:3]
        mesh.color[:] = fresnel.color.linear(vcolors)
        mesh.material.primitive_color_mix = 1.0

    scene.lights = fresnel.light.cloudy()

    # TODO: maybe initializing with fitting gives better camera angles
    scene.camera = fresnel.camera.Orthographic.fit(scene, margin=0)
    # Radius is just largest vertex norm
    r = np.max(np.linalg.norm(vertices))
    elevs = torch.linspace(0, 2 * np.pi, n+1)[:n]
    azims = torch.linspace(-np.pi, np.pi, n+1)[:n]
    renders = []
    for i in range(len(elevs)):
        elev = elevs[i]
        azim = azims[i]
        # Then views are just linspace
        # Loop through all camera angles and collect outputs
        pos, lookat, _ = get_camera_from_view(elev, azim, r=r)
        scene.camera.look_at = lookat
        scene.camera.position = pos
        out = fresnel.pathtrace(scene, samples=n_sample, w=width,h=height)
        renders.append(out[:])

    # Plot and save in matplotlib using imshow
    import matplotlib.pyplot as plt
    # plt.subplots_adjust(wspace=0, hspace=0)
    fig, axs = plt.subplots(nrows=1, ncols=len(renders), gridspec_kw={'wspace':0, 'hspace':0}, figsize=(15, 4), squeeze=True)
    for i in range(len(renders)):
        render = renders[i]
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].imshow(render, interpolation='lanczos')
    fig.suptitle(plotname)
    fig.tight_layout()
    plt.savefig(os.path.join(savedir, filename))
    plt.cla()
    plt.close()

# NOTE: This assumes default view direction of (0, 0, -r)
def get_camera_from_view(elev, azim, r=2.0):
    x = r * np.cos(azim) *  np.sin(elev)
    y = r * np.sin(azim) * np.sin(elev)
    z = r * np.cos(elev)

    pos = np.array([x, y, z])
    look_at = -pos
    direction = np.array([0.0, 1.0, 0.0])
    return pos, look_at, direction