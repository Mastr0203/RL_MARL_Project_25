import numpy as np
import torch as T
import torch.nn.functional as F
from GlobalAgent import GlobalAgent

class MADDPG:
    """ This class implements the MADDPG algorithm to train multiple agents with 
        centralized critics and decentralized actors. It supports ensemble actor 
        policies and a distilled mode for policy compression. Each agent is modeled 
        as a GlobalAgent and learns in a shared multi-agent environment.

        INPUT:
        - actor_dims : list of individual agent observation dimensions
        - critic_dims : dimension of the global state (concatenated observations)
        - n_agents : number of agents
        - n_actions : number of actions per agent
        - ensemble_polices : number of ensemble actors per agent
        - scenario : name of the scenario (used for checkpoints)
        - alpha, beta : learning rates for actor and critic
        - fc1, fc2 : hidden layer sizes
        - gamma : discount factor
        - tau : soft update rate
        - chkpt_dir : directory for saving models
        - H : entropy coefficient
        - decay_rate : entropy decay factor
        - n_envs : number of parallel environments
        - batch_size : size of each learning batch
    """

    def __init__(self,actor_dims,critic_dims,n_agents,n_actions, ensemble_polices = 1, scenario='simple',
                 alpha=0.01,beta=0.01,fc1=64,fc2=64,gamma=0.99,tau=0.01,
                 chkpt_dir='RL_Multi_Agent/tmp/maddpg/checkpoint',
                 H=0.1,decay_rate=0.999, n_envs = 1, batch_size = 1024, fc1_distill = 64, fc2_distill = 64,lr_distill = 1e-3):
        
        self.n_envs = n_envs
        self.agents = []
        self.n_agent = n_agents
        self.n_actions = n_actions
        self.ensemble_policies = ensemble_polices
        chkpt_dir += scenario
        self.actor_dims = actor_dims
        self.batch_size = batch_size

        self.distilled_mode = False         #Flag

        for agent_idx in range(self.n_agent):                                                      
            agent = GlobalAgent(actor_dims[agent_idx],critic_dims,n_actions,ensemble_polices, 
                                agent_idx, chkpt_dir,self.n_agent,alpha=alpha,beta=beta,gamma=gamma,
                                tau=tau,H=H,decay_rate=decay_rate,fc1_distill = fc1_distill, fc2_distill = fc2_distill, lr_distill = lr_distill)   
            self.agents.append(agent)


    def save_checkpoint(self):
        print('... Saving checkpoint ...')
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        print('... Loading checkpoint ...')
        for agent in self.agents:
            if self.ensemble_policies > 0:
                agent.load_models(student=True)
            else:
                agent.load_models(student=False)

    def choose_action(self,obs, who_acts,evaluate=False):
        """
            Selects an action for each agent based on its current observation and the index of the actor
            to use from the ensemble (if ensemble policies are active). In distilled mode, all agents
            use the first (distilled) policy instead.
            Each agent acts independently using only its local observation. The method returns the selected
            actions and the indices of the actors used, which are necessary for updating the correct ensemble
            members during training.

            INPUT:
            - obs -> np.ndarray : shape [n_envs, n_agents, obs_dim], observations for all agents
            - who_acts -> list[int] : index of the actor to use for each agent (required when using ensembles)

            OUTPUT:
            - actions -> torch.Tensor : shape [n_envs, n_agents] or [n_envs, n_agents, action_dim]
            - actor_idxs -> list[int] : indices of the actors used (for credit assignment during training)
        """

        if self.distilled_mode:
            who_acts = [0 for _ in range(self.n_agent)]     
        actions = []                                        #store one action for each agent who takes action independently
        actor_idxs = []                                     #keeps track on the index of the agent who acted (for the ensemble)

        for agent_idx,agent in enumerate(self.agents):
            agent_obs = obs[:, agent_idx, :]                
            action, actor_idx = agent.choose_action(agent_obs, evaluate, actor_idx = int(who_acts[agent_idx]))
            action_tensor = T.tensor(action, dtype=T.float32, device="cpu").clamp(-1, 1)            #clamp for env action space

            actions.append(action_tensor)
            actor_idxs.append(actor_idx)

        #[n_agents,n_envs,n_actions]
        return T.from_numpy(np.stack(actions, axis = 0)), actor_idxs
    
    def learn(self,memory):
        """ Updates the critics and actors of every agent using a minibatch taken from the
            replay buffer. Training is centralised for the critics (they see the global
            state and the joint actions) but decentralised for the actors (each actor uses 
            only its own observation).

            Workflow
            --------
            1.  Sample batch  
                memory.sample_buffer() returns, for every transition in the batch:
                - local observations (actor_states, actor_new_states)
                - global states (states, states_)
                - actions and rewards for all agents
                - indices of the ensemble actor that actually acted (actors_idx)
                - done flags

            2.  Compute target actions  
                For every agent (and for each ensemble policy if ensemble_policies > 1)
                the method:
                - re-computes the current policy actions π(s)
                - evaluates the target network actions π(s') (detached).

            3.  Critic update (centralised)  
                Using the concatenated future target actions it estimates  
                Q'(s', a') and forms the TD-target  
                y = r + gamma,Q'(s',a') (masked when any env in the vector is done).  
                The critic parameters are updated by minimising the MSE between y and
                Q(s,a).

            4.  Actor update (decentralised / ensemble aware)  
                For every ensemble actor that appeared in the batch the transitions where
                it was selected are extracted, its policy is re-executed, the joint action
                vector is rebuilt, and the deterministic policy-gradient loss.

            5.  Soft target updates  
                After each critic and actor update the corresponding target networks
                are softly updated with agent.update_parameters().
        """
        
        if memory.ready() == False:              #No learning until batch size is fulled up
            return
        
        actor_states,states,actions,rewards,\
        actor_new_states,states_, actors_idx, dones = memory.sample_buffer()    #states and states_ are all states merged, used for the critic (global)
                                                                                #actor_states and actor_states_ are observations of the states for each agent

        device = self.agents[0].actor[0].device
        
        states = T.as_tensor(np.array(states), dtype=T.float32).to(device)
        actions = T.as_tensor(np.array(actions), dtype=T.float32).to(device)
        rewards = T.as_tensor(np.array(rewards), dtype=T.float32).to(device)
        states_ = T.as_tensor(np.array(states_), dtype=T.float32).to(device)
        dones = T.as_tensor(np.array(dones), dtype=T.bool).to(device)

        all_agents_new_actions = []                 #future TARGET actor actions
        all_agents_new_mu_actions = []              #future actor actions
        old_agents_actions = []                     #present actions (placed in the batch)

        for agent_idx,agent in enumerate(self.agents):   
            mu_states = T.tensor(actor_states[agent_idx],dtype=T.float).to(device)         #states s of the current agent                                   
            new_states = T.tensor(actor_new_states[agent_idx],dtype=T.float).to(device)    #transition states s' of the current agent
            idxs = actors_idx[agent_idx]                                                   #which ensemble actor did act
            pi = []
            new_pi = []
            for sample_idx in range(new_states.shape[0]):
                actor_idx = 0 if self.distilled_mode else int(idxs[sample_idx])   #ensemble index for this sample
                state = mu_states[sample_idx]                                     #shape: [n_envs, obs_dim]
                new_state = new_states[sample_idx]                                #shape: [n_envs, obs_dim]

                action = agent.actor[actor_idx].forward(state)
                pi.append(action)
                
                #Compute the new action a' with target network (needed to compute the Q-value)
                target_action = agent.target_actor[actor_idx].forward(new_state).detach()
                new_pi.append(target_action) 

            pi = T.stack(pi, dim = 0) 
            all_agents_new_mu_actions.append(pi)                                #This is the new actor policy

            new_pi = T.stack(new_pi, dim = 0)                                           
            all_agents_new_actions.append(new_pi)                               #This is the new TARGET actor policy

            old_agents_actions.append(actions[agent_idx].clone())               #Keep track of old actions to compute loss

        new_actions = T.cat([acts for acts in all_agents_new_actions],dim=1)    #Merge future actions of target actors (for critic)
        old_actions = T.cat([acts for acts in old_agents_actions],dim=1)        #Merge old actions of actors (for critic)

       #Critic loop and policy updates
        for agent_idx,agent in enumerate(self.agents):  
            idxs = np.array(actors_idx[agent_idx])  
            with T.no_grad():
                #Compute Q'(s',a') using all the agents' obs s' and actions a'
                critic_value_ = agent.target_critic.forward(states_,new_actions).squeeze()
            
            #The ~ is used to invert True with False and v.v.,i.e. if episode is not done you have True, made
            #because Q' must not consider next episode if it is last episode
            done_any = dones.any(dim=1)
            mask = (~done_any).float()                                            
            critic_value_ = critic_value_ * mask

            #Compute Q(s ,a) using all the agents' obs s and actions a and compute target y=r+gamma*Q'(s',a')
            critic_value = agent.critic.forward(states,old_actions).squeeze()
            target = rewards[:, agent_idx] + agent.gamma * critic_value_
            
            critic_loss = F.mse_loss(target,critic_value)
            
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()

            #Since the policies are numerous per agent (if ensemble_policies > 1), we must retrieve all the transitions
            #where each ensemble actor acted to update the actor networks' weights and all the global information regarding
            #the other agents during those transitions (we must pass this last information to the critic)
            
            #Ensemble loop
            for ensemble in np.unique(idxs):
                ensemble = 0 if self.distilled_mode else int(ensemble)
                idxs_ = np.argwhere(idxs == ensemble).flatten()            #Find timesteps indexes when ensemble actor acted

                ensemble_actor_states = T.tensor(actor_states[agent_idx][idxs_], dtype=T.float32).to(device)
                ensemble_actor_actions = agent.actor[ensemble](ensemble_actor_states)          
                ensemble_states = states[idxs_]                                         
                
                #Build the list of actions for every agent at the selected transitions
                #and flatten each tensor so that each sample keeps only the action
                #dimensions (batch_subset, action_dim * n_envs)
                
                mu_actions = [                                      
                    a[idxs_].detach().view(len(idxs_), -1)
                    for a in all_agents_new_mu_actions            
                ]

                # Overwrite the actions of the current agent with the fresh ones
                mu_actions[agent_idx] = ensemble_actor_actions.view(len(idxs_), -1)

                # Concatenate along the agent dimension -> [batch_subset, n_agents * action_dim * n_envs]
                mu_input = T.cat(mu_actions, dim=1)
                
                actor_loss = - agent.critic.forward(ensemble_states, mu_input).mean()
                agent.actor[ensemble].optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                agent.actor[ensemble].optimizer.step()

                agent.update_actor_parameters(actor_idxs = ensemble, target_actor_idxs = ensemble)
            
            agent.update_critic_parameters()
            
