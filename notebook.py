#!/usr/bin/env python
# coding: utf-8

# # Neural Network Quantum State
# 
# 

# In[1]:


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


# In[ ]:





# In[2]:


os.environ["JAX_PLATFORM_NAME"] = "cpu" # or gpu
# os.environ["RAY_PICKLE_VERBOSE_DEBUG"] = "1" # RAY DEBUG
# os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"  # force gpu w/o right drivers

skip_training = False
print(f'{nk.utils.mpi.available=}')

# Force detecting GPU in WSL
ray.shutdown()
ray.init(num_gpus=1)


# In[3]:


if not skip_training:
    from ray.tune.suggest.hyperopt import HyperOptSearch
    
    search_space = {
        'model': {
            'alpha': tune.randint(1, 3+1), # last value exclusive
        },
        # 'activation': tune.choice(['tanh', 'sigmoid']),
        'learning_rate': tune.uniform(0.0, 0.1),
        'n_epochs': 300, #tune.qrandint(100, 300, 50),
        'n_samples': 1008, #tune.qrandint(100, 1000, 100),
    }
    
    metric = "energy_error"
    mode = "min"

    hyperopt_search = HyperOptSearch(metric=metric, mode=mode)
    hyper_band_scheduler = tune.schedulers.ASHAScheduler(
        metric=metric,
        mode=mode,
        max_t=search_space['n_epochs'].max_value if isinstance(search_space['n_epochs'], tune.sample.Sampler) else search_space['n_epochs'],
        grace_period=20
    )

    analysis = tune.run(
        ray_train_loop,
        config=search_space,
        progress_reporter=tune.JupyterNotebookReporter(True, [metric]),
        scheduler=hyper_band_scheduler,
        resources_per_trial={"cpu": 6, 'gpu': 1/2},
        search_alg=hyperopt_search,
        num_samples=10,
        resume='AUTO',
        # metric=metric,
        # mode=mode,
    )


# ### Select best model parameters

# In[14]:


# TODO

hyperparams = {
    'model': {
        'alpha': 1
    },
    'learning_rate': 0.09,
    'n_epochs': 100,
    'n_samples': 1008,
}

H, hi = setup_problem() #Remove the parameter for other models than j1j2. For j1j2 this chooses J2 when J1 = 1.0
eig_vals, _ = eigsh(H.to_sparse(), k=2, which="SA")
vstate, model, trainer = setup_model(H, hi, hyperparams)
display(f'{vstate.n_parameters=}')
log = nk.logging.RuntimeLog()


# In[17]:


trainer.run(n_iter=hyperparams['n_epochs'], out=log)

ffn_energy = vstate.expect(H)
error = abs((ffn_energy.mean-eig_vals[0])/eig_vals[0])
print("Optimized energy and relative error: ", ffn_energy, error)


# In[7]:


# TODO load saved model
if skip_training:
    pass


# ## Results

# In[18]:


data = log.data

plt.errorbar(data["Energy"].iters, jnp.real(data["Energy"].Mean), yerr=data["Energy"].Sigma, label="FFN")
#plt.hlines([E_gs_analytic], xmin=0, xmax=data["Energy"].iters.max(), color='black', label="Exact")
plt.hlines([eig_vals], xmin=0, xmax=data["Energy"].iters.max(), color='black', label="Exact")
plt.legend()

plt.xlabel('Iterations')
plt.ylabel('Energy')


# ### Wavefunctions

# In[16]:


ket = vstate.to_array()

plt.plot(abs(ket))


# In[9]:


# get quantum geometric tensor of state
# https://github.com/netket/netket/blob/2a7dded3db4705099d4de5450006b46b32ce34ca/netket/optimizer/qgt/qgt_onthefly_logic.py
qgt = vstate.quantum_geometric_tensor()
qgt


# In[10]:


QGT = qgt.to_dense()
jnp.imag(QGT)


# In[ ]:





# # The AKLT Model
# 
# The AKLT model is an extension to the simple 1D Heisenberg spin model, proposed in 1987 by Affleck, I. et al.:
# 
# https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.59.799
# 
# The model is essentially a 1D chain of spin-1/2 nuclei which form pairs. A constraint of having two spin-1/2 nuclei for each site is further imposed, which leads to the system being effectively a spin-1 system. This can be understood then to be a chain of these spin-1 "composite" nuclei. 
# 
# In the ground state of this Hamiltonian, every dimerized pair is referred to as a site and is given by a 
# 
# The AKLT Hamiltonian is given by a linear combination of the spin-1 projection operators in the respective Hilbert space $$\mathcal{H} \subset \mathbb{C}^3$$.

# 
