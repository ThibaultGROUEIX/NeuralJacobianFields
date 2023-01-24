import matplotlib.pyplot as plt
import numpy as np

def plot_uv(path, pred_vertices, triangles, gt_vertices=None, cvals=None, cmin=0, cmax=1, cvalsuff="_"):
    # setup side by side plots
    if gt_vertices:
        fig, axs = plt.subplots(1,2,figsize=(15, 4))
        fig.suptitle(str(path))

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
        plt.savefig(path)
        plt.close(fig)
    else:
        fig, axs = plt.subplots(figsize=(8, 4))
        fig.suptitle(str(path))

        # plot ours 
        axs.set_title('Ours')

        axs.triplot(pred_vertices[:,0], pred_vertices[:,1], triangles, linewidth=0.5)
        plt.axis('off')
        plt.savefig(path)
        plt.close(fig)
        
    # Plot color values if needed 
    if cvals is not None:
        fig, axs = plt.subplots(figsize=(8, 4))
        fig.suptitle(f"Mean: {np.mean(cvals):0.4f}")
        cmap = plt.get_cmap("coolwarm")
        axs.tripcolor(pred_vertices[:, 0], pred_vertices[:, 1], triangles, facecolors=cvals,  cmap=cmap,
                      linewidth=0.5, vmin=cmin, vmax=cmax, edgecolor="black")
        plt.axis('off')
        plt.savefig(path.replace(".png", f"{cvalsuff}.png"), bbox_inches='tight',dpi=600)
        plt.close()


    # plt.tripcolor(pred_vertices[:, 0], pred_vertices[:, 1], triangles, c, edgecolor="k",  cmap=cmap,linewidth=0.1)
    # plt.gca().set_aspect('equal')
    # plt.axis('off')
    # plt.savefig(path+'_pred.png', bbox_inches='tight',dpi=600)
    # plt.close()

    # fig = plt.figure()
    # ax = plt.axes()
    # ax.plot_trisurf(V[:, 0], V[:, 1], V[:, 2], triangles=F, edgecolor=[[0, 0, 0]], linewidth=0.01, alpha=1.0)
    # matplotlib.pyplot.savefig(fname)
    # matplotlib.pyplot.close(fig)