import os
import json
import time

import pandas as pd
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import flax
import ray
# import flax.linen as nn
import netket as nk
import netket.nn as nn

from scipy.sparse.linalg import eigsh
from netket.operator.spin import sigmaz, sigmax 
from ray import tune
from tqdm.autonotebook import tqdm
from IPython.display import display

# Our source code
# Change the imported model to aklt_model etc.
from j1j2_model import OurModel, setup_problem, setup_model, ray_train_loop 

L = 10
g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True, max_neighbor_order=2)
#print(g.edges)
#print(g.distances)
#print(g.nodes)
J = 0.1
hi = nk.hilbert.Spin(s=1, total_sz=0, N=g.n_nodes)
print(sigmaz(hi, 1))
H = sum(J*sigmaz(hi, i)*sigmaz(hi, j) + 0.5*J*sigmaz(hi, i)*sigmaz(hi, j) for i, j in g.edges())
