import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
import numpy as np
from typing import Any, Dict
from main import train, make_env
import json
import os
import wandb

N_TRIALS = 16
N_STARTUP_TRIALS = 5
N_EPISODES = 1000
N_EVAL_EPISODES = 10
n_envs = 1
n_agents = 3

DEFAULT_HYPERPARAMS = {
    "scenario": "cooperative",
    "ensemble_policies": 1,
    "max_steps": 400,
    "evaluate": False,
    "Use_wandb":False,
    "n_episodes": N_EPISODES,
    "H": 0.27364592538346605,                               
    "alpha": 2.9078757970174904e-04,                        
    "batch_size": 512,   # 512
    "beta": 3.547998054264363e-04,                         
    "decay_rate": 0.9984424002596399,                             
    "gamma": 0.9764874896436564,                            
    "learn_every": 10,
    "tau": 0.02543418538368299
}

def sample_maddpg_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sampler for MADDPG hyperparameters."""
    params = {}

    params.update({
        "H": trial.suggest_float("H", 0.02, 0.3, log=True),
        "decay_rate": trial.suggest_float("decay_rate", 0.95, 0.99999),
        "alpha": trial.suggest_float("alpha", 5e-6, 5e-4, log=True),
        "beta": trial.suggest_float("beta", 5e-5, 5e-4, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512, 1024]),
        "learn_every": trial.suggest_categorical("learn_every", [5, 10, 20, 40, 80]),
        "gamma": trial.suggest_float("gamma", 0.9, 0.995),
        "tau": trial.suggest_float("tau", 5e-4, 0.05,log=True),
    })

    for key, val in params.items():
        trial.set_user_attr(key, val)
    return params

def objective(trial: optuna.Trial) -> float:
    kwargs = DEFAULT_HYPERPARAMS.copy()
    sampled_params = sample_maddpg_params(trial)
    kwargs.update(sampled_params)

    env = make_env(
        scenario="balance",
        num_envs=n_envs,
        device="cpu",
        continuous_actions=True,
        max_steps=None,
        seed=None,
        n_agents=n_agents,
        random_package_pos_on_line = False,
    )

    try:
        result = train(env=env, **kwargs)
        best_reward = result.get("best_reward", -np.inf)

        wandb.log({
            "trial": trial.number,
            "best_reward": best_reward,
            **{f"param/{k}": v for k, v in sampled_params.items()}
        }, step=trial.number)

        trial.report(best_reward, step=0)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return best_reward

    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        trial.set_user_attr("error", str(e))
        wandb.log({"trial": trial.number, "status": "failed"}, step=trial.number)
        return float("nan")

def save_best_params(study: optuna.Study, filename: str = "best_params.json"):
    best_params = study.best_params
    best_params.update({
        "best_value": study.best_value,
        "user_attrs": dict(study.best_trial.user_attrs)
    })

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        json.dump(best_params, f, indent=4)

if __name__ == "__main__":
    torch.set_num_threads(1)
    
    wandb.init(
        project="maddpg-tuning",
        name="optuna_study_run",
        config={
            "n_trials": N_TRIALS,
            "n_episodes": N_EPISODES,
            "n_eval_episodes": N_EVAL_EPISODES
        }
    )

    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=10)

    study = optuna.create_study(
        sampler=sampler,
        pruner=pruner,
        storage="sqlite:///db.sqlite3",
        direction="maximize",
        study_name="maddpg_balance_tuning_wandb_2"
    )

    try:
        study.optimize(objective, n_trials=N_TRIALS)#, timeout=3600)
    except KeyboardInterrupt:
        pass

    print("\nNumber of finished trials:", len(study.trials))
    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    save_best_params(study, "tuning_results/best_params.json")