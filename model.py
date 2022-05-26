from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import netket as nk
import netket.nn as nn

from scipy.sparse.linalg import eigsh
from netket.operator.spin import sigmaz, sigmax, sigmay
from ray import tune


GCNN = nk.models.GCNN

class FFN(nn.Module):
    
    # Define attributes that can be set with `**kwargs`
    alpha: int = 1
        
    @nn.compact
    def __call__(self, x):
        n_inputs = x.shape[-1]
        
        x = nk.nn.Dense(features=self.alpha*n_inputs, 
                       use_bias=True, 
                       dtype=jnp.complex128, 
                       kernel_init=jax.nn.initializers.normal(stddev=0.01), 
                       bias_init=jax.nn.initializers.normal(stddev=0.01)
                      )(x)
        x = nk.nn.log_cosh(x)
        
        x = jnp.sum(x, axis=-1)
        return x


def setup_j1j2_problem(L: int = 20, J2: float = 0.8):

    #Couplings J1 and J2
    edge_colors = []
    for i in range(L):
        edge_colors.append([i, (i+1)%L, 1])
        edge_colors.append([i, (i+2)%L, 2])

    # Define the netket graph object
    g = nk.graph.Graph(edges=edge_colors)

    # Found in netket site https://netket.readthedocs.io/en/latest/tutorials/gs-j1j2.html
    J = [1, J2]
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
    # Spin based Hilbert Space
    hi = nk.hilbert.Spin(s=0.5, total_sz=0.0, N=g.n_nodes)
    H = nk.operator.GraphOperator(hi, graph=g, bond_ops=bond_operator, bond_ops_colors=bond_color)
    
    neel = nk.operator.LocalOperator(hi, dtype=complex)
    for i in range(0, L):
        neel += nk.operator.spin.sigmaz(hi, i)*((-1)**(i))/L
    
    structure_factor = nk.operator.LocalOperator(hi, dtype=complex)
    for i in range(0, L):
        for j in range(0, L):
            structure_factor += (nk.operator.spin.sigmaz(hi, i)*nk.operator.spin.sigmaz(hi, j))*((-1)**(i-j))/L
    
    dimer = nk.operator.LocalOperator(hi, dtype=complex)
    
    for i in range(0, L):
        dimer += (nk.operator.spin.sigmap(hi, i)*nk.operator.spin.sigmam(hi, (i+1)%L) + nk.operator.spin.sigmam(hi, i)*nk.operator.spin.sigmap(hi, (i+1)%L))*((-1)**(i))/L
        
    obs = {'Neel': neel, 'sf': structure_factor, 'dimer': dimer}

    return H, hi, g, obs


def setup_aklt_problem(L: int = 10):
    # Sets up the system according to the AKLT model:
    # https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.59.799
    #

    # The model is essentially a spin-1 chain made up of two spin-1/2s pairwise
    # This implies three eigenstates, according to which the system's spin projection
    # operators can be defined.
    g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)
    N = g.n_nodes
    hi = nk.hilbert.Spin(s=1, N=N)
    
    #H0 = nk.operator.Heisenberg(hi,g)
    #H += (H0 + (1/3)*H0*H0)
    # Should we use full spin vectors instead of just sigmaz?
    H = 0
    for i, j in g.edges():
        dotpr = sigmaz(hi, i) * sigmaz(hi, j) #+ sigmax(hi, i) * sigmax(hi, j) + sigmay(hi, i) * sigmay(hi, j)
        H += dotpr + (1/3)*dotpr*dotpr
        
    
    #H = sum(sigmaz(hi, i) * sigmaz(hi, j) for i, j in g.edges())
    #H += sum(1/3 * ( sigmaz(hi, i)*sigmaz(hi, j) * sigmaz(hi, i)*sigmaz(hi, j) ) for i, j in g.edges())
    obs = []
    return H, hi, g, obs


def setup_model(H: nk.operator.AbstractOperator, hi: nk.hilbert.AbstractHilbert,
                g: nk.graph.Graph, model: nn.Module, hyperparams: dict):
    """ Use given hyperparameters and return training loop, or 'driver'."""
    # Init model with hyperparams
    model = model(**hyperparams['model'])

    sampler = nk.sampler.MetropolisExchange(hilbert=hi, graph=g, d_max = 2)
    vstate = nk.vqs.MCState(sampler, model, n_samples=hyperparams['n_samples'])

    # Define the optimizer
    optimizer = nk.optimizer.Sgd(learning_rate=hyperparams["learning_rate"])

    # Init driver, i.e., training loop
    trainer = nk.driver.VMC(H, optimizer, variational_state=vstate,preconditioner=nk.optimizer.SR(diag_shift=0.1))

    return vstate, model, trainer


@lru_cache
def get_ground_state(H: nk.operator.AbstractOperator):
    " Compute ground state energy of given NetKet Hamiltonian. "
    return eigsh(H.to_sparse(), k=2, which="SA")[0][0]


def ray_train_loop(hyperparams: dict, setup_problem=setup_j1j2_problem, model: nn.Module = FFN, checkpoint_dir=None):  # pylint: disable=unused-argument
    H, hi, g, _ = setup_problem()
    _, model, trainer = setup_model(H, hi, g, model, hyperparams)
    log = nk.logging.RuntimeLog()
    
    E_gs_analytic = get_ground_state(H)

    def _ray_callback(step: int, logdata: dict, driver: nk.driver.AbstractVariationalDriver) -> bool:  # pylint: disable=unused-argument
        energy = logdata["Energy"].Mean
        error = abs((energy - E_gs_analytic))
        # with tune.checkpoint_dir(step=step) as checkpoint_dir:
        #     path = os.path.join(checkpoint_dir, "checkpoint")
        # Send current test error for hyprparameter tuning
        tune.report(energy_error=error)
        return True

    trainer.run(n_iter=hyperparams['n_epochs'], callback=_ray_callback, out=log)