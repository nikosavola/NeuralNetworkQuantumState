from functools import lru_cache

import jax.numpy as jnp
import netket as nk
import netket.nn as nn

from scipy.sparse.linalg import eigsh
from netket.operator.spin import sigmaz 
from ray import tune


class OurModel(nn.Module):
    
    # Define attributes that can be set with `**kwargs`
    alpha: int = 1
            
    @nn.compact
    def __call__(self, x):

        # x = x.reshape(-1, 1, x.shape[-1])  # shape for translation symmetry

        # Layers
        re = nk.nn.Dense(
            # symmetries=g.translation_group(),
            features=self.alpha * x.shape[-1],
            # kernel_init=nn.initializers.normal(stddev=0.01)
        )(x)
        re = nk.nn.relu(re)
        re = jnp.sum(re, axis=-1)
        
        im = nk.nn.Dense(
            # symmetries=g.translation_group(),
            features=self.alpha * x.shape[-1],
            # kernel_init=nn.initializers.normal(stddev=0.01)
        )(x)
        im = nn.relu(im)
        im = jnp.sum(im, axis=-1)
        
        return re + 1j * im


def setup_problem():
    L = 15
    g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)
    N = g.n_nodes
    hi = nk.hilbert.Spin(s=1/2, N=N)

    H = sum(sigmaz(hi, i) * sigmaz(hi, j) for i, j in g.edges())
    H += sum(1/3 * sigmaz(hi, i) * sigmaz(hi, j) * sigmaz(hi, i) * sigmaz(hi, j) for i, j in g.edges())

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
    trainer = nk.driver.VMC(H, optimizer, variational_state=vstate,preconditioner=nk.optimizer.SR(diag_shift=0.1))

    return vstate, model, trainer


@lru_cache
def get_ground_state(H):
    " Compute ground state energy of given NetKet Hamiltonian. "
    return eigsh(H.to_sparse(), k=2, which="SA")[0][0]


def ray_train_loop(hyperparams, checkpoint_dir=None):
    H, hi = setup_problem()
    vstate, model, trainer = setup_model(H, hi, hyperparams)
    log = nk.logging.RuntimeLog()
    
    E_gs_analytic = get_ground_state(H)

    def _ray_callback(step: int, logdata: dict, driver: "AbstractVariationalDriver") -> bool:
        energy = logdata["Energy"].Mean
        error = abs((energy - E_gs_analytic))
        # with tune.checkpoint_dir(step=step) as checkpoint_dir:
        #     path = os.path.join(checkpoint_dir, "checkpoint")
        # Send current test error for hyprparameter tuning
        tune.report(energy_error=error)
        return True

    trainer.run(n_iter=hyperparams['n_epochs'], callback=_ray_callback, out=log)