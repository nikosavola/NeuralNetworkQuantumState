import json
import argparse

import jax.numpy as jnp
import numpy as np
import pandas as pd
import netket as nk

from model import GCNN, setup_j1j2_problem, setup_model, get_ground_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hyperparams",
        type=str,
        required=True,
        help="Hyperparams for model as JSON filename",
    )
    parser.add_argument(
        "--J2_idx",
        type=int,
        default=1.,
        help="Index for value of J2",
    )
    parser.add_argument(
        "--J2_max",
        type=int,
        default=1.,
        help="Max index value of J2",
    )
    args = parser.parse_args()

    
    J2 = np.linspace(0.2, 0.80, args.J2_max)[args.J2_idx - 1]
    H, hi, g, obs = setup_j1j2_problem(J2=J2) 

    # load hyperarams
    with open(args.hyperparams, 'r', encoding='utf-8') as fp:
        hyperparams = json.load(fp)
    hyperparams['model'].update({'symmetries': g})


    vstate, model, trainer = setup_model(H, hi, g, GCNN, hyperparams)
    log = nk.logging.RuntimeLog()
    
    trainer.run(n_iter=hyperparams['n_epochs'], out=log, obs=obs)
    data = log.data
    
    last_means = 3
    # save results to csv
    df = pd.Series({
        'J2': J2,
        'Eacc': get_ground_state(H),
        'neelOPs': jnp.mean(jnp.real(data['Neel'].Mean)[-last_means:-1]),
        'AFOPs': jnp.mean(jnp.real(data['sf'].Mean)[-last_means:-1]),
        'Esim': jnp.mean(jnp.real(data['Energy'].Mean)[-last_means:-1]),
        'sigmas': jnp.mean(jnp.real(data['Energy'].Sigma)[-last_means:-1]),
        'dimOPs': jnp.mean(jnp.real(data['dimer'].Mean)[-last_means:-1]),
    }).to_frame().T
    df.to_csv(f'data/phase/results_J2={J2:.4f}.csv', index=False)
    print(df)
