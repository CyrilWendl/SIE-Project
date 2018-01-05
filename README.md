# SIE Project 2017
Folder for various codes for the SIE project at EPFL in Fall Semester 2017.

## Files
### `Code/`
Visualizations
- `decision_tree.ipynb`: Decision Trees and Random Forest on randomly generated labelled data
- `density_tree.ipynb`: Density Trees on randomly generated labelled data
- `MNIST.ipynb`: Trainig of a CNN on the MNIST dataset, retrieval of the FC layer activation weights, Density Forest

### `Code/density_tree`

Package for implementation of Decision Trees, Density Forests and Random Forests
- `create_data.py`: functions for generating labelled and unlabelled data
- `decision_tree.py`: data structure for decision tree nodes
- `decision_tree_create.py`: functions for generating decision trees
- `decision_tree_traverse.py`: functions for traversing a decision tree to predict labels
- `density_tree.py`: data struture for density tree nodes
- `density_tree_create.py`: functions for generating density trees
- `density_tree_traverse.py`: functions for descending density trees and retreiving their Gaussian parameters
- `density_forest.py`: functions for creating density forests
- `helper.py`: helper functions
- `plots.py`: functions for plotting the data
- `random_forests`: functions for creating random forests

## Supervisors:
- Prof. Devis Tuia, University of Wageningen
- Diego Marcos González, University of Wageningen
- Prof. François Golay, EPFL

Cyril Wendl, 2017
