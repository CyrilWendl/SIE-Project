from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm_notebook
from .decision_tree_create import *
from .decision_tree_traverse import *

def draw_subsamples(dataset, subsample_pct = .8):
    """draw random subsamples with replacement from a dataset
    :param dataset: the dataset from which to chose subsamples from
    :param subsample_pct: the size of the subsample dataset to create in percentage of the original dataset
    """
    subsample_size = int(np.round(len(dataset) * subsample_pct)) # subsample size
    dataset_indices = np.arange(len(dataset))
    dataset_subset_indices = np.random.choice(dataset_indices, size = subsample_size, replace = True,) # draw random samples with replacement
    dataset_subset = dataset[dataset_subset_indices,:]
    return dataset_subset


def random_forest_build(dataset, ntrees, subsample_pct, n_jobs):
    """Create random forest trees"""
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()

    root_nodes = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(create_decision_tree)(draw_subsamples(dataset, subsample_pct=subsample_pct)) for i in range(ntrees))
    return root_nodes

def random_forest_traverse(dataset, root_nodes):
    """traverse random forest and get labels"""
    # get labels for dataset
    dataset_eval=[]
    # traverse all points
    for i in tqdm_notebook(dataset):
        # traverse all trees
        label=[]
        for tree in root_nodes:
            label.append(descend_decision_tree(i,tree))
        # get most frequent label
        counts = np.bincount(label)
        label = np.argmax(counts)
        dataset_eval.append(np.concatenate([i,[label]]))

    dataset_eval=np.asarray(dataset_eval)
    return dataset_eval