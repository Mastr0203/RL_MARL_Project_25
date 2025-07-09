import numpy as np
import torch as T

class MultiAgentReplayBuffer:
    """ This class creates a replay buffer used to train multiple agent whit MADDPG algorithm
        which is an off-policy algorithm. Therefore aim is to store transitional experiences in 
        batch form sucht that they can be used to update neural networks of agents. Moreover,
        ensemble policy scenario is also handeld.
        
        INPUT:
        - max_size --> int : it is the maximum number of storable transitions in the buffer
        - critic_dims --> int : vector size of global state (i.e. concatenation of all agents observations)
        - actor_dims --> list[int] : list containing dimensions of observations of each agent (individual)
        - n_actions --> int : number of actions that each agent can take
        - n_agents --> int : number of agents
        - n_envs --> int : number of enviroments trained in parallel
        - batch_size --> int : number of transitions that must be done for each batch during training 
    """

    def __init__(self,max_size,critic_dims,actor_dims,
                 n_actions,n_agents, n_envs, batch_size):
        
        self. mem_size = max_size
        self.mem_cntr = 0                                                              #counter for batch filling 
        self.n_agents = n_agents
        self.n_envs = n_envs
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.actor_dims = actor_dims

        self.state_memory = np.zeros((self.mem_size, critic_dims))                     #global state of the enviroment at step t
        self.new_state_memory = np.zeros((self.mem_size, critic_dims))                 #global state of the enviroment at step t+1
        self.terminal_memory = np.zeros((self.mem_size,  n_agents),dtype=bool)         #episode state for each agent (done or not)

        self.init_actor_memory()

    def init_actor_memory(self):
        """ This function initializes all data structures used during MADDPG. In particular it creates
            lists that will contain ,for each agent, numpy Arrays to store observations at time t and 
            t+1, actions and rewards.  
        """

        self.actor_state_memory = []
        self.actor_new_state_memory = []
        self.actor_action_memory = []
        self.actors_idx_memory = []                                                     #Store the indexes of the ensemble policies
        self.reward_memory = np.zeros((self.mem_size, self.n_agents))

        for i in range(self.n_agents):                                              
            self.actor_state_memory.append(np.zeros((self.mem_size, self.actor_dims[i])))         #Update state memory of actor j of agent i
            self.actor_new_state_memory.append(np.zeros((self.mem_size,  self.actor_dims[i])))    #Update new state memory of actor j of agent i
            self.actor_action_memory.append(np.zeros((self.mem_size,  self.n_actions)))           #Update action memory of actor j of agent i
            self.actors_idx_memory.append(np.zeros((self.mem_size, )))

    def store_transition(self,obs,state,action,reward,obs_,state_,done, actors_idx):
        """ Function store transition into replay buffer at each state, i.e. it updates observations,
            actions, rewards, episode state (done) and new states for each agent. Three main indexes:
            - env_idx : it is used if training is performed whitin multiple enviroments in parallel
            - agent_idx : it is used to store the transition of the single agent inside the buffer
            - index : used to subscript in a certain position data inside the buffer (buffer is finite length)    
        """                                             

        for env_idx in range(self.n_envs):
            index = self.mem_cntr % self.mem_size
            for agent_idx in range(self.n_agents):
                self.actor_state_memory[agent_idx][index] = obs[agent_idx][env_idx]
                self.actor_new_state_memory[agent_idx][index] = obs_[agent_idx][env_idx]
                self.actor_action_memory[agent_idx][index] = action[agent_idx][env_idx]
                self.actors_idx_memory[agent_idx][index] = actors_idx[agent_idx]

            self.state_memory[index] = state[env_idx]
            self.new_state_memory[index] = state_[env_idx]

            reward_array = np.array([rew[env_idx].item() if T.is_tensor(rew) else rew[env_idx] for rew in reward])
            self.reward_memory[index] = reward_array
            self.terminal_memory[index] = done[env_idx]
        
            self.mem_cntr += 1                                                   #observation occures, increament the counter

    def sample_buffer(self):
        """ Since MADDPG is off-policy, this function samples a batch of random experiences from replay buffer
            in order to train neural networks of agents. 
            Outputs will have the form (n_agents, batch_size, n_env, action_dim)
        """
        
        max_mem = min(self.mem_cntr,self.mem_size)                              #available transitions in the buffer
        batch = np.random.choice(max_mem,self.batch_size,replace=False)         #retrieve random indexes

        states = self.state_memory[batch]                                       #->(batch_size, n_envs, allStateActions)
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        terminal = self.terminal_memory[batch]

        actor_states = []
        actor_new_states = []
        actors_idx = []
        actions = []

        for agent_idx in range(self.n_agents):
            actor_states.append(self.actor_state_memory[agent_idx][batch])             #Save state of actor of agent i
            actor_new_states.append(self.actor_new_state_memory[agent_idx][batch])     #Save new state of actor of agent i
            actors_idx.append(self.actors_idx_memory[agent_idx][batch])                #Save index of the ensemble actor of agent i
            actions.append(self.actor_action_memory[agent_idx][batch])                 #Save action of actor of agent i

        return actor_states,states,actions,rewards,\
                actor_new_states,states_, actors_idx, terminal
    
    def ready(self):
        """ This function checks whether the buffer is ready for training.
            It returns True only if the buffer contains enough transitions to sample a batch.
        """

        if self.mem_cntr >= self.batch_size:
            return True 
        else:
            return False
        
    def collect_joint_data(self, agents, env, dataset_size = 50_000):
        """ This function is used to generate an offline dataset composed by (obs,actions) pairs
            used for the actors ensemble (one for agent) scenario.

            INPUT:
            - agents -> list[objects] : each object of the list is an agent, which has .actor 
                                        attribute (list of actors of the enseble)
            - env -> enviroment object : is the enviroment in which agents is trained
            - dataset_size -> int : total number of transitions to collect
        """

        dataset = []
        state = env.reset()

        for _ in range(dataset_size):
            joint_actions = []                                         #actions of all agents in this step (until dataset_size)
            obs_all_agents = state

            for agent_idx, agent in enumerate(agents):              
                obs = obs_all_agents[agent_idx]

                #compute each action coming from each actor in the ensemble of the agent and 
                #merge them. Next, takes maximum action (on the mean)
                teacher_actions = T.stack([teacher(obs).detach() for teacher in agent.actor])
                aggregated_action = teacher_actions.mean(dim=0).squeeze(0)

                if len(aggregated_action.shape) == 1:
                    aggregated_action = aggregated_action.unsqueeze(0)       #retrieve batch dimensions compatibility
                joint_actions.append(aggregated_action.cpu())
                
            next_state, reward, done, _ = env.step(joint_actions)
            dataset.append((obs_all_agents, joint_actions))

            if any(done):
                state = env.reset()
            else:
                state = next_state

        return dataset

