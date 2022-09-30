import matplotlib.pyplot as plt
import numpy 
import matplotlib

def plot_uv(path, gt_vertices, pred_vertices, triangles):
    # setup side by side plots
    fig, axs = plt.subplots(1,2,figsize=(15, 4))
    fig.suptitle(str(path))

    # plot GT 
    axs[0].set_title('GT')
    axs[0].axis('equal')

    axs[0].triplot(gt_vertices[:,0].cpu(), gt_vertices[:,1].cpu(), triangles, linewidth=0.5)

    # plot ours 
    axs[1].set_title('Ours')
    axs[1].axis('equal')
    # axs[1].set_axis_off()

    axs[1].triplot(pred_vertices[:,0].cpu(), pred_vertices[:,1].cpu(), triangles, linewidth=0.5)
    plt.savefig(path+'_comparison.png')
    plt.close(fig)

    # generate some array of length same as xy
    c = numpy.ones(len(gt_vertices))
    # create a colormap with a single color
    cmap = matplotlib.colors.ListedColormap('gray')
    # tripcolorplot with a single color filling:
    plt.tripcolor(gt_vertices[:, 0], gt_vertices[:, 1], triangles, c, edgecolor="k",  cmap=cmap,linewidth=0.1)
    plt.gca().set_aspect('equal')
    plt.axis('off')
    plt.savefig(path+'_gt.png', bbox_inches='tight',dpi=600)
    plt.close()


    plt.tripcolor(pred_vertices[:, 0], pred_vertices[:, 1], triangles, c, edgecolor="k",  cmap=cmap,linewidth=0.1)
    plt.gca().set_aspect('equal')
    plt.axis('off')
    plt.savefig(path+'_pred.png', bbox_inches='tight',dpi=600)
    plt.close()

    # fig = plt.figure()
    # ax = plt.axes()
    # ax.plot_trisurf(V[:, 0], V[:, 1], V[:, 2], triangles=F, edgecolor=[[0, 0, 0]], linewidth=0.01, alpha=1.0)
    # matplotlib.pyplot.savefig(fname)
    # matplotlib.pyplot.close(fig)