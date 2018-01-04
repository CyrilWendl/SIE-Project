import os

# Custom Libraries
from density_tree.density_tree_create import *
from density_tree.create_data import *
from density_tree.helpers import *
from density_tree.plots import *

# Generate unlablled data
dimensions = 2
nclusters = 5
covariance = 10
npoints = 100
minRange = 10
maxRange = 100

dataset = create_data(nclusters, dimensions, covariance, npoints, minrange=minRange, maxrange=maxRange,
                      labelled=False, random_flip=True, nonlinearities=True)

m = np.mean(dataset, axis=0)
cov = np.cov(dataset.T)

if dimensions == 2:
    figure, axes = plt.subplots(nrows=1, ncols=1)
    plot_data(dataset, "Unlabelled data", axes, labels=False, covs=[cov], means=[m])
    plt.show()


if dimensions == 2:
    figure, axes = plt.subplots(nrows=1, ncols=1)
    plot_data(dataset, "Unlabelled data", axes, labels=False)
    plt.show()

root = create_density_tree(dataset, dimensions=dimensions, clusters=nclusters)


def get_values_preorder(node, cut_dims_root, cut_vals_root):
    cut_dims_root.append(node.split_dimension)
    cut_vals_root.append(node.split_value)
    if node.left is not None:
        get_values_preorder(node.left, cut_dims_root, cut_vals_root)
    if node.right is not None:
        get_values_preorder(node.right, cut_dims_root, cut_vals_root)
    return cut_vals_root, cut_dims_root


cut_vals, cut_dims = get_values_preorder(root, [], [])
cut_vals = np.asarray(cut_vals).astype(float)
cut_dims = np.asarray(cut_dims).astype(int)

x_split = cut_vals[cut_dims == 0]
y_split = cut_vals[cut_dims == 1]

if dimensions == 2:
    fig, ax = plt.subplots(1, 1)
    plot_data(dataset, "Training data after splitting", ax, labels=False, lines_x=x_split, lines_y=y_split,
              minrange=minRange, maxrange=maxRange, covariance=covariance)

    plt.show()

print(cut_dims, cut_vals)


# Printing the Tree

def tree_visualize(root_node):
    tree_string = ""
    tree_string = print_density_tree_latex(root_node, tree_string)
    
    os.system("cd figures; rm main.tex; more main_pt1.tex >> main.tex; echo '' >> main.tex;")
    os.system("cd figures; echo '" + tree_string + "' >> main.tex;  more main_pt2.tex >> main.tex;")
    os.system(
        "cd figures; /Library/TeX/texbin/pdflatex main.tex; convert -density 300 -trim main.pdf -quality 100 main.png")
    # display(Image('./figures/main.png', retina=True))


tree_visualize(root)


# Showing all Clusters Covariances

covs = []
means = []


def get_clusters(node):
    """add all leaf nRodes to an array in preorder traversal fashion"""
    # check for leaf node
    if node.left is not None:
        get_clusters(node.left)
    else:
        covs.append(node.left_cov)
        means.append(node.left_mean)
    if node.right is not None:
        get_clusters(node.right)  
    else:
        covs.append(node.right_cov)
        means.append(node.right_mean)
        

get_clusters(root)

if dimensions == 2:
    figure, axes = plt.subplots(nrows=1, ncols=1)
    plot_data(dataset, "Unlabelled data", axes, labels=False, covs=covs, means=means)
    plt.show()
