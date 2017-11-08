import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np

def plotData(data, title, linesX = None, linesY = None, clusters = None, save=False, labels=True, minRange=1, maxRange=100, covariance = 2):
    """
    Plot the clustered data
    """
    fig, ax = plt.subplots()

    if labels:
        color = iter(cm.rainbow(np.linspace(0, 1, len(clusters))))
        for i, c in enumerate(data):
            color_cluster = next(color)
            ax.plot(c[:, 0], c[:, 1], '.', color=color_cluster)

            x = c[:, 0]
            y = c[:, 1]
            n = [int(c) for c in c[:, 2]]

            for i, txt in enumerate(n):
                ax.annotate(txt, (x[i], y[i]))
    else:
        ax.plot(data[:, 0], data[:, 1], '.')

    ax.grid()
    plt.title(title)

    # draw split lines after partitioning
    if linesX is not None and linesY is not None:
        for y_line in range(len(linesY)):
            ax.axhline(y=linesY[y_line], c="red")
        for x_line in range(len(linesX)):
            ax.axvline(x=linesX[x_line], c="red")

    axes = plt.gca()
    axes.set_xlim([minRange - 4 * covariance, maxRange + 4 * covariance])
    axes.set_ylim([minRange - 4 * covariance, maxRange + 4 * covariance])

    if (save):
        plt.savefig('/Users/cyrilwendl/Documents/EPFL/Projet SIE/SIE-Project/random_data.pdf', bbox_inches='tight')

    plt.show()