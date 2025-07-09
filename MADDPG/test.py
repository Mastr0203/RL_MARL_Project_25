import numpy as np
from MADDPG import MADDPG
import torch as T
from vmas import make_env
import json
import wandb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def test(env, maddpg_agents, episode = 0):
    n_envs = env.num_envs
    score = 0
    obs = env.reset()
    dones = [False] * n_envs
    episode_step = 0
    who_acts = np.zeros(n_agents, dtype=int)

    while not all(dones):
        env.render(mode="human")
        obs_reshaped = np.transpose(obs, [1, 0, 2])
        actions, _ = maddpg_agents.choose_action(obs_reshaped, who_acts,evaluate=True)
        obs_, reward, dones, info = env.step(actions)
        obs = obs_
        score += sum(np.mean(reward, axis=1))
        episode_step += 1

    print(f"Test {episode} score: {score}")
    return score

env = make_env(
        scenario="balance",        
        num_envs= 1,
        device="cpu",              
        continuous_actions=True,
        max_steps= 400,            
        seed=42,                 
        n_agents = 3,         
        udr_enabled = False,        
        test= True,
    )

actor_dims = [env.observation_space[i].shape[0] for i in range(env.n_agents)]
critic_dims = sum(actor_dims)
n_actions = env.action_space[0].shape[0]
n_agents = env.n_agents
n_envs = env.num_envs

with open('tuning_results/best_params.json', 'r') as f:
    config = json.load(f)
par = config["user_attrs"]
hidden_dim = 64
ensemble_policies = 0
n_episodes = 200
chkpt_dir = 'RL_Multi_Agent' 

configuration = {
    "normal": chkpt_dir + "/normal/",
    "UDR": chkpt_dir + "/normal_udr/",
    "Ensemble": chkpt_dir + "/ensemble/",
    "Ensemble_udr": chkpt_dir + "/ensemble_udr/",
    }

config_reward = {}

wandb.login(key="f6c0a7231485fae8fe4c04eade75115805025613")
for key, value in configuration.items():
    if "ensemble" in value:
        ensemble_policies = 3
    maddpg_agents = MADDPG(actor_dims,critic_dims,n_agents,n_actions, 
                                scenario="cooperative",
                                alpha=par["alpha"],beta=par["beta"],
                                fc1=hidden_dim,fc2=hidden_dim,
                                gamma=par["gamma"],tau=par["tau"],
                                chkpt_dir= value,
                                H=par["H"],decay_rate=par["decay_rate"], 
                                ensemble_polices=ensemble_policies,
                                n_envs = n_envs,
                                batch_size = par["batch_size"],
                                fc1_distill = 64, fc2_distill = 64, lr_distill=1e-3)

    maddpg_agents.load_checkpoint()

    wandb.init(project="final-maddpg-balance", name=f"test_{key}", reinit=True)

    score_list = []
    for episode in range(n_episodes):
        test_score = test(env, maddpg_agents, episode)
        score_list.append(test_score)
        if episode >= 100:
            wandb.log({
                        "episode": episode,
                        "avg_score_100": np.mean(score_list[-100:]),
                        "std_score_100": np.std(score_list[-100:]),
                        "episode_score": test_score,
                    })
        config_reward[key] = score_list
    wandb.finish()

data = pd.DataFrame(config_reward)

sns.boxplot(data=data, palette="Pastel1")
plt.title("Compare between different policy")
plt.ylabel("Reward")
plt.xlabel("Policy")
plt.show()