import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("..")
from utils import *
from linear_regression import *
from svm import *
from softmax import *
from features import *
from kernel import *
import os

train_x, train_y, test_x, test_y = get_MNIST_data()

for image_no in range(6,10):
    directory = f"projected_numbers/image{image_no}"
    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.figure()
    plt.imshow(train_x[image_no].reshape(28,28))
    plt.savefig(f"{directory}/original.png")
    plt.close()
    
    train_x_centered, feature_means = center_data(train_x)
    pcs = principal_components(train_x_centered)

    for n_components in [1, 2,3,4, 5, 10, 20, len(pcs)]:
        # n_components = 20
        # image_no = 0

        train_pca = project_onto_PC(train_x, pcs, n_components, feature_means)
        test_pca = project_onto_PC(test_x, pcs, n_components, feature_means)

        plt.figure()
        plt.imshow((pcs[:,:n_components] @ train_pca[image_no]).reshape(28,28))
        plt.title(f"Number = {train_y[image_no]}")
        plt.savefig(f"{directory}/pca{n_components}.png")
        plt.close()

