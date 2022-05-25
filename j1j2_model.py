import jax
import jax.numpy as jnp
import numpy as np
import flax
import netket as nk
import netket.nn as nn

from scipy.sparse.linalg import eigsh
from netket.operator.spin import sigmaz, sigmax, sigmay
from ray import tune
from tqdm.autonotebook import tqdm


#Couplings J1 and J2
L = 20
edge_colors = []
for i in range(L):
    edge_colors.append([i, (i+1)%L, 1])
    edge_colors.append([i, (i+2)%L, 2])

# Define the netket graph object
g = nk.graph.Graph(edges=edge_colors)

class OurModel(nn.Module):
    
    # Define attributes that can be set with `**kwargs`
    alpha: int = 1
        
    @nn.compact
    def __call__(self, x):
        n_inputs = x.shape[-1]
        
        x = nk.nn.Dense(features=self.alpha*n_inputs, 
                       use_bias=True, 
                       dtype=jnp.complex128, 
                       kernel_init=nn.initializers.normal(stddev=0.01), 
                       bias_init=nn.initializers.normal(stddev=0.01)
                      )(x)
        x = nk.nn.log_cosh(x)
        
        x = jnp.sum(x, axis=-1)
        return x


def setup_problem(J2 = 0.8):
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
        
    return H, hi, [neel,structure_factor, dimer]

def setup_model(H, hi, hyperparams):
    # Init model with hyperparams
    model = OurModel(**hyperparams['model'])

    sampler = nk.sampler.MetropolisExchange(hilbert=hi, graph=g, d_max = 2)
    
    vstate = nk.vqs.MCState(sampler, model, n_samples=hyperparams['n_samples'])
    # Define the optimizer

    optimizer = nk.optimizer.Sgd(learning_rate=hyperparams["learning_rate"])
    
    # Init driver, i.e., training loop
    trainer = nk.driver.VMC(H, optimizer, variational_state=vstate, preconditioner=nk.optimizer.SR(diag_shift=0.1))

    return vstate, model, trainer


def ray_train_loop(hyperparams, checkpoint_dir=None):
    H, hi,obs = setup_problem(0.1)  # TODO Choose this also as a parameter of model
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