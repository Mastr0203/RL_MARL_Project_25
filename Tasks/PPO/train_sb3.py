from __future__ import annotations

import time
import os
import csv

import wandb

import argparse
import gymnasium as gym
from stable_baselines3 import PPO, SAC

from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy  
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from env import custom_hopper

# Custom Wandb logging callback
class WandbLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        if self.locals.get("ep_info_buffer"):
            wandb.log({
                "timesteps": self.num_timesteps,
                "rollout/ep_rew_mean": self.locals["ep_info_buffer"][-1]['r'],
                "rollout/ep_len_mean": self.locals["ep_info_buffer"][-1]['l'],
            })
        return True
def make_configuration(gamma, learning_rate, batch_size, n_steps, ent_coef, clip_range, gae_lambda, n_epochs):
    return f'g{gamma}_lr{learning_rate}_bs{batch_size}_nSteps{n_steps}_entCoef{ent_coef}_cr{clip_range}_lambda{gae_lambda}_epochs{n_epochs}'

def make_model_name(train_domain:str,udr:str,configuration:str) -> str:
    """
    It returns a filename based on:
	•	the algorithm name (PPO / SAC)
	•	UDR
	•	an optional incremental counter
    """
    name = f"PPO_{train_domain}_{configuration}"
    if udr:
        name += "_udr"   
    return name

# -----------------------------
# 1. Hyperparameters
# -----------------------------

COMMON_HYPERS = {
     "gamma": 0.99,
     "tensorboard_log": "./tb",
}

ALG_HYPERS = {
        "learning_rate": 5e-4,
        "batch_size": 128,
        "n_steps": 1024,
        "clip_range": 0.2,
        "ent_coef": 0.05,
        "n_epochs": 10,
        "gae_lambda": 0.95
}

# -----------------------------
# 2. Environment setup
# -----------------------------
def make_env(domain: str, udr: bool = False) -> gym.Env:
    env_id = f"CustomHopper-{domain}-v0"
    env = gym.make(env_id)
    if hasattr(env.unwrapped, "udr_enabled"):
        env.unwrapped.udr_enabled = udr
    
    return Monitor(env)

# -----------------------------
# 3. Callbacks
# -----------------------------
def create_callbacks(n_steps: int, test_domain, Wandb, name, n_envs):
    best_model_path = f"./best_model/{name}"
    eval_env = make_env(test_domain)
    eval_env.reset(seed=42)

    eval_callback = EvalCallback(
        eval_env = eval_env,
        best_model_save_path=best_model_path,
        eval_freq = n_steps * n_envs,
        deterministic=True,
        render=False,
        verbose=0
    )
    callbacks = [eval_callback]
   
    if Wandb:
        wandb.init(
            project="CustomHopper-RL",
            config={
                "algorithm": 'PPO',
                "n_steps": n_steps,
                **COMMON_HYPERS,
                **ALG_HYPERS,
            },
            name= name,
            sync_tensorboard=True, 
            monitor_gym=True,
        )
        callbacks.append(WandbLoggingCallback())
    return callbacks

def train_model(hypers:dict, train_domain:str, total_timesteps, name, UDR:bool, WandDB:bool, save_csv, configuration, n_envs):
    train_env = make_vec_env(lambda: make_env(train_domain, udr=UDR), n_envs=n_envs)

    model = PPO(
        "MlpPolicy",  # predefined policy network: a 2-layer MLP with Tanh activation
        train_env,
        **hypers
    )

    callbacks = create_callbacks(
        n_steps=hypers.get("n_steps", 1),
        test_domain = train_domain,
        Wandb = WandDB,
        name = name,
        n_envs= n_envs
        )

    start = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList(callbacks)
    )
    end = time.time()
    
    if UDR:
        print("Training with Uniform Domain Randomization (UDR) enabled.")

    filename_best = f"best_model/{name}/best_model.zip"
    best_model = PPO.load(filename_best)

    # --- Evaluation and CSV logging ---
    eval_results = []
    if train_domain == "source":
        for test_dom in ["source", "target"]:
            env = make_env(test_dom)
            env.reset(seed=42)
            mean_reward, std_reward = evaluate_policy(
                best_model,
                env,
                n_eval_episodes=50,
                deterministic=True,
                render=False,
                warn=False,
            )
            eval_results.append((train_domain, test_dom, mean_reward, std_reward, name, configuration))
    else:  # train_domain == "target"
        env = make_env("target")
        env.reset(seed=42)
        mean_reward, std_reward = evaluate_policy(
            best_model,
            env,
            n_eval_episodes=50,
            deterministic=True,
            render=False,
            warn=False,
        )
        eval_results.append((train_domain, "target", mean_reward, std_reward, name, configuration))

    if save_csv:
        # Save to CSV
        csv_path = "results.csv"
        header = ["run_id","train_domain", "test_domain", "mean_reward", "std_reward", "model_name", "configuration"]
        file_exists = os.path.isfile(csv_path)
        if file_exists:
            with open(csv_path, mode="r") as f:
                existing_lines = sum(1 for line in f) - 1
            run_id = existing_lines // 2 + 1
        else:
            run_id = 1
        eval_results_with_id = [
            (run_id, *r) for r in eval_results
        ]
        with open(csv_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(header)
            writer.writerows(eval_results_with_id)
    
    for r in eval_results:
        print(f"[EVAL] Trained on {r[0]} → Tested on {r[1]} | Mean Reward: {r[2]:.2f} | Std: {r[3]:.2f}")

    if WandDB:
        wandb.log({
            "train/time_elapsed": end - start
        })

    print(f"Training time: {end - start:.2f} seconds")

# -----------------------------
# 4. Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Train PPO or SAC on CustomHopper")
    parser.add_argument("--train_domain", choices=["source", "target"], default="source", help="Domain to train on [source, target]")
    parser.add_argument("--WandDB", action="store_true", help="Use WandDB Callback")
    parser.add_argument("--UDR", action="store_true", default=False)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_steps", type=int, default=1024)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.001)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--n_epochs", type=int, default=10)

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--save_csv", action="store_true", help="Enable saving results to CSV")
    parser.add_argument("--n_envs", type=int, default=1)
    
    args = parser.parse_args()

    # collect hyperparams
    ALG_HYPERS["learning_rate"] = args.learning_rate
    ALG_HYPERS["batch_size"] = args.batch_size
    ALG_HYPERS["n_steps"] = args.n_steps
    ALG_HYPERS["clip_range"] = args.clip_range
    ALG_HYPERS["ent_coef"] = args.ent_coef
    ALG_HYPERS["gae_lambda"] = args.gae_lambda
    ALG_HYPERS["n_epochs"] = args.n_epochs

    COMMON_HYPERS["gamma"]= args.gamma

    hypers = {**COMMON_HYPERS, **ALG_HYPERS}

    configuration = make_configuration(args.gamma, args.learning_rate, args.batch_size, args.n_steps, args.ent_coef, args.clip_range, args.gae_lambda, args.n_epochs)
    model_filename = make_model_name(args.train_domain,args.UDR, configuration)

    train_model(hypers, args.train_domain, args.total_timesteps, model_filename, args.UDR, args.WandDB, args.save_csv, configuration, n_envs=args.n_envs)

if __name__ == "__main__":
    main()