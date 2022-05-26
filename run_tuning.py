import argparse
from datetime import datetime

import pandas as pd

from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch

from model import GCNN, FFN, setup_j1j2_problem, ray_train_loop


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_samples",
        type=int,
        default=40,
        help="Number of trials for hyperparameter tuning",
    )
    args = parser.parse_args()



    _, _, g, _ = setup_j1j2_problem()
    search_space = {
        'model': {
            # 'alpha': tune.randint(1, 3+1), # last value exclusive
            'symmetries': g,
            'layers': tune.randint(4, 16+1),
            'features': tune.randint(2, 10+1)
        },
        # 'activation': tune.choice(['tanh', 'sigmoid']),
        'learning_rate': tune.uniform(0.005, 0.12),
        'n_epochs': 50, #tune.qrandint(100, 300, 50),
        'n_samples': 1400, #tune.qrandint(100, 1000, 100),
    }

    metric = "energy_error"
    mode = "min"

    hyperopt_search = HyperOptSearch(metric=metric, mode=mode)
    hyper_band_scheduler = tune.schedulers.ASHAScheduler(
        metric=metric,
        mode=mode,
        max_t=search_space['n_epochs'].max_value if isinstance(search_space['n_epochs'], tune.sample.Sampler) else search_space['n_epochs'],
        grace_period=8
    )

    analysis = tune.run(
        tune.with_parameters(ray_train_loop, setup_problem=setup_j1j2_problem, model=GCNN),
        config=search_space,
        progress_reporter=tune.CLIReporter([metric]),
        scheduler=hyper_band_scheduler,
        resources_per_trial={"cpu": 2},
        search_alg=hyperopt_search,
        num_samples=args.num_samples,
        resume='AUTO',
    )
    
    df = analysis.dataframe()
    df_params = df['config/model'].apply(pd.Series).drop(['symmetries'], axis=1)
    df = df_params.join(df['config/learning_rate']).join(df['energy_error'])
    df.to_csv(f'data/ray_tune_output_{datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")}.csv', index_label='uid')

    hyperparams = analysis.get_best_config(metric=metric, mode=mode)
    print(hyperparams)