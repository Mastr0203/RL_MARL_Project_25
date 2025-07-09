import numpy as np
from Train import train
from vmas import make_env

def make_name(udr,ensemble):
    match (udr,ensemble):
        case (False,0):
            return "normal"
        case (True,0):
            return "normal_udr"
        case (False,e) if e > 0:
            return "ensemble"
        case (True,e) if e > 0:
            return "ensemble_udr"

def main():
    n_envs = 1
    n_agents = 3
    ensemble_policies = 0
    udr = False

    env = make_env(
        scenario="balance",        
        num_envs=n_envs,
        device="cpu",              
        continuous_actions=True,
        max_steps= 400,            
        seed=None,                 
        n_agents=n_agents,         
        random_package_pos_on_line = False,  
        udr_enabled = udr,         
    )
    
    print(f'number of agents: {n_agents}')
    
    for agent in range(n_agents):
        print(f'Agent{agent}, Observation Space: {env.observation_space[agent].shape}, Action Space: {env.action_space[agent]}')

    print(f"Number of actions: {env.action_space[0].shape[0]}")

    name = make_name(udr,ensemble_policies)
    chkpt_dir = f'RL_Multi_Agent/{name}'   

    kwargs = {                                                  
        "H": 0.27364592538346605,                               
        "alpha": 2.9078757970174904e-04,                        
        "batch_size": 512,
        "beta": 3.547998054264363e-04,                         
        "decay_rate": 0.9984424002596399,                             
        "gamma": 0.9764874896436564,                            
        "learn_every": 10,
        "tau": 0.02543418538368299
    }

    training_result = train(
        env=env,
        ensemble_policies=ensemble_policies,
        chkpt_dir=chkpt_dir,
        print_interval= 20,
        Use_wandb= False,
        name = name,
        **kwargs
    )
    print(f"best_reward: {training_result.get('best_reward', -np.inf)}")

if __name__ == '__main__':
    main()