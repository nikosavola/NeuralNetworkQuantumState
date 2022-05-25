import jax
import jax.numpy as jnp
import flax
import netket as nk
import netket.nn as nn
import numpy as np

from scipy.sparse.linalg import eigsh
from netket.operator.spin import sigmaz, sigmax, sigmay
from ray import tune
from tqdm.autonotebook import tqdm


class OurModel(nn.Module):
    
    # Define attributes that can be set with `**kwargs`
    alpha: int = 1
            
    @nn.compact
    def __call__(self, x):

        # INPUTS:
        # x - the number of features 

        # x = x.reshape(-1, 1, x.shape[-1])  # shape for translation symmetry

        # Layers
        re = nk.nn.Dense(
            # symmetries=g.translation_group(),
            features=self.alpha * x.shape[-1],
            # kernel_init=nn.initializers.normal(stddev=0.01)
        )(x)
        re = nk.nn.relu(re) # Defines a rectified linear activation function (ReLU)
        re = jnp.sum(re, axis=-1) # Sums over the real 
        
        im = nk.nn.Dense(
            # symmetries=g.translation_group(),
            features=self.alpha * x.shape[-1],
            # kernel_init=nn.initializers.normal(stddev=0.01)
        )(x)
        im = nn.relu(im)
        im = jnp.sum(im, axis=-1)
        
        return re + 1j * im


def setup_problem(J2=0.1, L=10):

    # Heisenberg model with second nearest neibhbour interactions
    # Source for the model https://arxiv.org/pdf/2112.10526.pdf
    # Another found in netket site, but cannot understand anyhing https://netket.readthedocs.io/en/latest/tutorials/gs-j1j2.html
    #g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True, max_neighbor_order=2)

    J = [1.0, J2]

    # Define custom graph
    edge_colors = []
    for i in range(L):
        edge_colors.append([i, (i+1)%L, 1])
        edge_colors.append([i, (i+2)%L, 2])

    # Define the netket graph object
    g = nk.graph.Graph(edges=edge_colors)

    #Sigma^z*Sigma^z interactions
    sigmaz = [[1, 0], [0, -1]]
    mszsz = (np.kron(sigmaz, sigmaz))

    #Exchange interactions
    exchange = np.asarray([[0, 0, 0, 0], [0, 0, 2, 0], [0, 2, 0, 0], [0, 0, 0, 0]])

    bond_operator = [
        (J[0] * mszsz).tolist(),
        (J[1] * mszsz).tolist(),
        (-J[0] * exchange).tolist(),  
        (J[1] * exchange).tolist(),
    ]
    bond_color = [1, 2, 1, 2]

    hi = nk.hilbert.Spin(s=1/2, total_sz=0, N=g.n_nodes)
    H = nk.operator.GraphOperator(hi, graph=g, bond_ops=bond_operator, bond_ops_colors=bond_color)
    #H = nk.operator.Heisenberg(hilbert=hi, graph=g, J=[1.0, J2])

    return H, hi


def setup_model(H, hi, hyperparams):
    """ Use given hyperparameters and return training loop, or 'driver'."""
    # Init model with hyperparams
    model = OurModel(**hyperparams['model'])

    sampler = nk.sampler.MetropolisLocal(hi)
    vstate = nk.vqs.MCState(sampler, model, n_samples=hyperparams['n_samples'])
    # Define the optimizer

    optimizer = nk.optimizer.Sgd(learning_rate=hyperparams["learning_rate"])
    # Init driver, i.e., training loop
    trainer = nk.driver.VMC(H, optimizer, variational_state=vstate, preconditioner=nk.optimizer.SR(diag_shift=0.1))

    return vstate, model, trainer


def ray_train_loop(hyperparams, checkpoint_dir=None):
    H, hi = setup_problem(J2=0.5)  # TODO Choose this also as a parameter of model
    vstate, model, trainer = setup_model(H, hi, hyperparams)
    log = nk.logging.RuntimeLog()
    
    # TODO precompute this globally
    E_gs_analytic, _ = eigsh(H.to_sparse(), k=2, which="SA")
    E_gs_analytic = E_gs_analytic[0] # Takes the ground state energy

    def _ray_callback(step: int, logdata: dict, driver: "AbstractVariationalDriver") -> bool:
        energy = logdata["Energy"].Mean
        error = abs((energy - E_gs_analytic))
        # with tune.checkpoint_dir(step=step) as checkpoint_dir:
        #     path = os.path.join(checkpoint_dir, "checkpoint")
        # Send current test error for hyprparameter tuning
        tune.report(energy_error=error)
        return True

    trainer.run(n_iter=hyperparams['n_epochs'], callback=_ray_callback, out=log)