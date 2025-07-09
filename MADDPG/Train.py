import numpy as np
from MADDPG import MADDPG
from MultiAgent import MultiAgentReplayBuffer
import torch as T
import wandb

def train( 
        env,
        H,                      #Hyperparameter (exploration term)
        decay_rate,             #Hyperparameter (decay for H)
        alpha,                  #Hyperparameter (learning rate actor)
        beta,                   #Hyperparameter (learning rate critic)
        batch_size,             #Hyperparameter
        learn_every,            #Hyperparameter
        gamma,                  #Hyperparameter (discount factor)
        tau,                    #Hyperparameter (update factor for weights (both actor and critic))

        scenario = 'cooperative',
        ensemble_policies = 0,   
        buffer_size = 500000,
        hidden_dim = 64,
        name = "normal",

        n_episodes= 7500,
        Use_wandb = True,
        chkpt_dir='RL_Multi_Agent/tmp/maddpg',
        print_interval = 500
        
    ):
    
    n_envs = env.num_envs
    n_agents = env.n_agents
    actor_dims = []
    for i in range(n_agents):
        actor_dims.append(env.observation_space[i].shape[0])            # insert the observation spaces dimensions in a list
    critic_dims = sum(actor_dims)                                       # critic network uses all actor observations (hence we sum them to retrieve dimension of the input layer)
    n_actions = env.action_space[0].shape[0]

    # define the object to manage multiple agents and their updates
    maddpg_agents = MADDPG(actor_dims,critic_dims,n_agents,n_actions, 
                            scenario=scenario,
                            alpha=alpha,beta=beta,
                            fc1=hidden_dim,fc2=hidden_dim,
                            gamma=gamma,tau=tau,
                            chkpt_dir=chkpt_dir,   #change anytime
                            H=H,decay_rate=decay_rate, 
                            ensemble_polices=ensemble_policies,
                            n_envs = n_envs,
                            batch_size = batch_size,
                            fc1_distill = 64, fc2_distill = 64, lr_distill=1e-3)
    
    # define the replay buffer to store past episodes
    memory = MultiAgentReplayBuffer(buffer_size,critic_dims,actor_dims,n_actions,n_agents, env.num_envs, batch_size=batch_size)
 
    total_steps = 0
    score_history = []
    reward_drop_count = 0
    previous_avg_score = 0
    max_drop_patience = 5
    best_score = 0
    
    if Use_wandb:
        wandb.login(key="f6c0a7231485fae8fe4c04eade75115805025613")
        wandb.init(project="maddpg-balance", name=f"train_{name}")

    for i in range(n_episodes):
        obs = env.reset()
        score = 0
        dones = [False] * n_envs                       #As a logic, we must extend each parameter/information for each env
        episode_step = 0

        if ensemble_policies > 0:
            who_acts = np.random.randint(0, ensemble_policies, n_agents)
        else:
            who_acts = np.zeros(n_agents, dtype=int)
            
        while not all(dones):                            

            obs_reshaped = np.transpose(obs, [1, 0, 2])
            actions, actors_idx = maddpg_agents.choose_action(obs_reshaped, who_acts)
            obs_, reward, dones, info = env.step(actions)

            state = obs_list_to_state_vector(obs)                              
            state_ = obs_list_to_state_vector(obs_)

            memory.store_transition(obs,state,actions,reward,obs_,state_,dones, actors_idx)

            if episode_step % learn_every == 0:
                maddpg_agents.learn(memory)

            obs = obs_
            score += sum(np.mean(reward, axis = 1))                               
            total_steps += 1
            episode_step += 1

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if Use_wandb:
            std_score = np.std(score_history[-100:])  
            wandb.log({
                        "episode": i,
                        "avg_score_100": avg_score,
                        "std_score_100": std_score,
                        "episode_score": score,
                        "total_steps": total_steps
                    })

        if avg_score > best_score:
            maddpg_agents.save_checkpoint()
            best_score = avg_score
        
        if i % print_interval == 0 and i > 0:
            delta = avg_score - previous_avg_score
            print('-'*50)
            print(f'Episode: {i}, average_score: {avg_score:.2f}, delta: {delta:.2f}')
            print('-'*50)
            if delta < 0:
                reward_drop_count += 1
            else:
                reward_drop_count = 0
            previous_avg_score = avg_score

            if reward_drop_count > max_drop_patience:
                print("Training stopped early due to sustained reward degradation.")
                break

    # Distillation if using policy ensemble
    if ensemble_policies > 0:
        dataset = memory.collect_joint_data(maddpg_agents.agents, env)
        for i, global_agent in enumerate(maddpg_agents.agents):
            global_agent.distill(dataset, path_file = f'RL_Multi_Agent/tmp/maddpg/distilled/agent_{i+1}')

        for agent in maddpg_agents.agents:
            agent.distilled_mode = True             #Flag activated to specify agent has been distilled
        maddpg_agents.distilled_mode = True         #Specify to MADDPG that distillation must be used

    return {
        'best_reward': best_score,
        'maddpg_agents': maddpg_agents,
        'memory': memory
    }

def obs_list_to_state_vector(obs):
    """
    A function to convert list of observations to tensor
    INPUT:
    - obs : list/tensor of observations
    OUTPUT:
    - obs to tensor
    """
    obs = np.array(obs)
    return np.stack([np.concatenate(obs[:, i, :]) for i in range(obs.shape[1])])