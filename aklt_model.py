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

        x = nk.nn.Dense(features=2*x.shape[-1], 
                       use_bias=True, 
                       dtype=jnp.complex128, 
                       kernel_init=nn.initializers.normal(stddev=0.01), 
                       bias_init=nn.initializers.normal(stddev=0.01)
                      )(x)
        x = nk.nn.log_cosh(x)
        x = jnp.sum(x, axis=-1)
        
        return x


def setup_problem():

    # Sets up the system according to the AKLT model:
    # https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.59.799
    #

    # The model is essentially a spin-1 chain made up of two spin-1/2s pairwise
    # This implies three eigenstates, according to which the system's spin projection
    # operators can be defined.
    #
    L = 10
    #g = nk.graph.Chain(length=L, pbc=True)

    edge_colors = []
    for i in range(0, L, 2):
        edge_colors.append([i, (i+1)%L, 1])
        #edge_colors.append([i, (i+2)%L, 2])

    # Define the netket graph object
    g = nk.graph.Graph(edges=edge_colors)

    N = g.n_nodes
    hi = nk.hilbert.Spin(s=1/2, N=N)

    # Should we use full spin vectors instead of just sigmaz?

    sx = [[0, 1], [1, 0]]
    sy = [[0, -1j], [1j, 0]]
    sz = [[1, 0], [0, -1]]
    mszsx = np.kron(sx, sx)
    mszsy = np.kron(sy, sy)
    mszsz = np.kron(sz, sz)
    H0 = np.eye(4)*1/3
    H1 = (mszsx**2 + mszsy**2 + mszsz**2)*1/2
    H2 = H1**2*1/6

    siteops = []
    bondops = [H0+H1+H2]
    bondcols = [1]
    H = nk.operator.GraphOperator(hilbert=hi, graph=g, site_ops=siteops, bond_ops=bondops, bond_ops_colors=bondcols)

    #heis = nk.operator.Heisenberg(hilbert=hi, graph=g)
    #H1 = 0.5*heis
    #H2 = 1/6*heis**2
    #H0 = 1/3*heis**0
    #H = H0 + H1 + H2
    
    #for i, j in g.edges():
    #    dotpr = sigmaz(hi, i)*sigmaz(hi, j+1) + sigmax(hi, i)*sigmax(hi, j+1) + sigmay(hi, i)*sigmay(hi, j+1)
    #    H += 0.5*dotpr + 1/6*dotpr*dotpr + 1/3
    
    #H = sum(sigmaz(hi, i) * sigmaz(hi, j) for i, j in g.edges())
    #H += sum(1/3 * ( sigmaz(hi, i)*sigmaz(hi, j) * sigmaz(hi, i)*sigmaz(hi, j) ) for i, j in g.edges())

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
    obs = [neel, structure_factor, dimer]
    
    return H, hi, g, obs


def setup_model(H, hi, hyperparams):
    """ Use given hyperparameters and return training loop, or 'driver'."""
    # Init model with hyperparams
    #model = OurModel(**hyperparams['model'])
    model = nk.models.RBMMultiVal(alpha=1, n_classes=3)

    # Define the sampler on the Hilbert space
    sampler = nk.sampler.MetropolisLocal(hi)
    
    # Sample state from the given Hilbert space
    vstate = nk.vqs.MCState(sampler, model, n_samples=hyperparams['n_samples'])

    # Define the optimizer
    optimizer = nk.optimizer.Sgd(learning_rate=hyperparams["learning_rate"])

    # Init driver, i.e., training loop
    trainer = nk.driver.VMC(H, optimizer, variational_state=vstate,preconditioner=nk.optimizer.SR(diag_shift=0.1))

    return vstate, model, trainer


def ray_train_loop(hyperparams, checkpoint_dir=None):
    H, hi, obs = setup_problem()
    vstate, model, trainer = setup_model(H, hi, hyperparams)
    log = nk.logging.RuntimeLog()
    
    # TODO precompute this globally
    E_gs_analytic, _ = eigsh(H.to_sparse(), k=2, which="SA")
    E_gs_analytic = E_gs_analytic[0]

    def _ray_callback(step: int, logdata: dict, driver: "AbstractVariationalDriver") -> bool:
        energy = logdata["Energy"].Mean
        error = abs((energy - E_gs_analytic))
        # with tune.checkpoint_dir(step=step) as checkpoint_dir:
        #     path = os.path.join(checkpoint_dir, "checkpoint")
        # Send current test error for hyprparameter tuning
        tune.report(energy_error=error)
        return True

    trainer.run(n_iter=hyperparams['n_epochs'], callback=_ray_callback, out=log)