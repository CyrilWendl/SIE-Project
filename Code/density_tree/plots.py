from .helpers import get_grid_labels
from .create_data import data_to_clusters
from matplotlib.pyplot import cm
import matplotlib.pylab as plt
import numpy as np


def plotData(data, title, ax, clusters=None, save=False, linesX=None, linesY=None,
             labels=True, minRange=1, maxRange=100, covariance=2, grid_eval=None, showData=True, covs=None, means=None):
    """
    Generic function to plot randomly generated labelled or unlabelled data.
    :param data: the data to plot
    :param title: the title of the plot
    :param linesX, linesY: x and y splitting lines to plot
    :param save [True | False]: save plot to a pdf file
    :param labels [True | False]: indicator whether data contains labels
    :param minRange, maxRange, covariance: data parameters for setting axis limits
    """
    if showData:
        if labels:
            color = iter(cm.rainbow(np.linspace(0, 1, len(clusters))))
            for i, c in enumerate(data):
                color_cluster = next(color)
                ax.scatter(c[:, 0], c[:, 1], s=40, color=color_cluster)

                x = c[:, 0]
                y = c[:, 1]
                n = [int(c) for c in c[:, 2]]

                for i, txt in enumerate(n):
                    ax.annotate(txt, (x[i], y[i]))
        else:
            ax.plot(data[:, 0], data[:, 1], '.')

    ax.set_title(title)

    # draw split lines after partitioning
    ax.grid()
    if linesX is not None and linesY is not None:
        for y_line in range(len(linesY)):
            ax.axhline(y=linesY[y_line], c="red")
        for x_line in range(len(linesX)):
            ax.axvline(x=linesX[x_line], c="red")

    # draw colored meshgrid
    if grid_eval is not None:
        x_min, x_max = [minRange, maxRange]
        y_min, y_max = [minRange, maxRange]
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))

        Z = grid_eval

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.pcolormesh(xx, yy, Z, alpha=0.2, cmap='rainbow')
        # ax.set_clim(y.min(), y.max())

    ax.set_xlim([minRange - 4 * covariance, maxRange + 4 * covariance])
    ax.set_ylim([minRange - 4 * covariance, maxRange + 4 * covariance])

    # covariance
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]
    nstd = 2
    if covs is not None:
        for i in range(len(covs)):
            vals, vecs = eigsorted(covs[i])
            theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            w, h = 2 * nstd * np.sqrt(vals)
            ell = Ellipse(xy=means[i],
                          width=w, height=h,
                          angle=theta, color='red')
            ell.set_facecolor('none')
            ax.add_artist(ell)

    if (save):
        plt.savefig('/Users/cyrilwendl/Documents/EPFL/Projet SIE/SIE-Project/random_data.pdf', bbox_inches='tight')

def visualize_decision_boundaries(dataset, rootnode, minRange, maxRange, rf=False):
    """visualize decision boundaries for a given decision tree"""
    # plot data
    clusters = data_to_clusters(dataset)
    dataset_grid_eval = get_grid_labels(dataset, rootnode, minRange, maxRange, rf=rf)

    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches((12, 8))

    fig.set_size_inches((15, 6))
    plotData(clusters, "Training Data and Splits", axes[0], clusters=clusters, minRange=minRange,
                   maxRange=maxRange, covariance=0, grid_eval=dataset_grid_eval, showData=True)

    plotData(clusters, "Splits", axes[1], clusters=clusters, minRange=minRange,
                   maxRange=maxRange, covariance=0, grid_eval=dataset_grid_eval, showData=False)

    plt.show()

    # Detail view of the problematic region
    # plotData(clusters_eval, "Test Data and Splits of Training Data", x_split, y_split, clusters = clusters_eval,
    #         minRange = 20, maxRange = 40)