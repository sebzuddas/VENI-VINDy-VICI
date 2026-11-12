import pytest
from vindy import *
import numpy as np
from vindy.layers import SindyLayer, VindyLayer
from vindy.utils import data_generation


import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import tensorflow as tf

# VENI-VINDy-VICI framework components
from vindy import IdentificationNetwork
from vindy.libraries import PolynomialLibrary
from vindy.layers import SindyLayer, VindyLayer
from vindy.distributions import Gaussian, Laplace
from vindy.callbacks import SaveCoefficientsCallback, PDFThresholdCallback

# Local utility functions for Roessler system
from utils import (
    data_generation, generate_directories, data_plot, 
    training_plot, trajectory_plot, uq_plot
)


def test_init_to_config():
    x = np.random.randint(0, 100, size=(3, 3, 3))
    # x_train = np.concatenate(x, axis=0)
    config_dict = {}
    sindy_layer = VindyLayer(state_dim=3)
    base_model = IdentificationNetwork(sindy_layer=sindy_layer, x=x)# we need to pass sindy_layer and x
    print(vars(base_model))
    


    # we need to initialise a model with given parameters, and then call _init_to_config. This is a private method, so we need to call a method within the subclass which calls this ABC method. 
    # This saves the model parameters. 
