from helpers import *
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys

import natsort as ns
from PIL import Image

import tensorflow

path = os.getcwd()
def prepare_data():
    """prepare data, return images and gt"""
    # modifiable variables
    im_dir = r''+ path + '/Zurich_dataset/images_tif/'
    gt_dir = r''+ path + '/Zurich_dataset/groundtruth/'

    im_names = ns.natsorted(os.listdir(im_dir))
    gt_names = ns.natsorted(os.listdir(gt_dir))
    print("images: %i " % len(im_names))
    print("ground truth images: %i " % len(gt_names))

    # load images
    images = np.asarray([im_load(im_dir + im_name) for im_name in im_names])
    gt = np.asarray([im_load(gt_dir + gt_name) for gt_name in gt_names])

    # histogram stretching and equalization
    im_stretch, im_eq = [],[]
    for i in range(len(images)):
        im_s, im_e = imgs_stretch_eq(images[i])
        im_stretch.append(im_s)
        im_eq.append(im_e)

    # print shapes to control
    for var in images, images[0], gt, gt[0]:
        print(var.shape)

    """
    # Show image and its groundtruth image
    images_ind = np.arange(3)
    for i in images_ind:
        fig, axes = plt.subplots(1, 2)
        fig.set_size_inches(10, 5)
        axes[0].imshow(im_stretch[i][:, :, 0:3], cmap='Greys_r')
        axes[1].imshow(gt[i], cmap='Greys_r')
        plt.show()
    """
    # continue using stretched image
    images = im_stretch

    # gt to labels
    # get label corresponding to each color

    legend = {'Background':[255, 255, 255],
              'Roads': [0, 0, 0],
              'Buildings': [100, 100, 100],
              'Trees':[0, 125, 0],
              'Grass': [0, 255, 0],
              'Bare Soil':[150, 80, 0],
              'Water':[0, 0, 150],
              'Railways':[255, 255, 0],
              'Swimming Pools':[150, 150, 255]}

    # get class names by increasing value (as done above)
    names, colors = [], []
    for name, color in legend.items():
        names.append(name)
        colors.append(np.sum(color))

    gt_maj_label = gt_color_to_label(gt, colors)
    np.shape(gt_maj_label)
    gt = gt_maj_label
    return images, gt
images, gt = prepare_data()