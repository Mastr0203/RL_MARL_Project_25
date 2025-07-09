import torch as T
import numpy as np
import os
from Networks import CriticNetwork,ActorNetwork
from Distill import DSTL

class GlobalAgent:
    """
    This class builds an agent in the MADDPG context. It handles all components needed
    to learn and perform the policy of the agent, both in standard mode and ensemble mode.

    INPUTS:
    - actor_dims        -> int or tuple : dimensionality of the agent's observation space
    - critic_dims       -> int or tuple : dimensionality of the critic input (full obs)
    - n_actions         -> int : number of continuous actions the agent can take
    - ensemble_policies -> int : number of actor networks to use in ensemble mode (0 = no ensemble)
    - agent_idx         -> int : index of the agent (used in naming saved models)
    - chkpt_dir         -> str : path to the directory for saving model checkpoints
    - n_agents          -> int : total number of agents in the environment

    HYPERPARAMETERS:
    - alpha       -> float (default=0.01) : learning rate for the actor networks
    - beta        -> float (default=0.01) _ learning rate for the critic network
    - fc1         -> int (default=64) : number of neurons in the first fully connected layer
    - fc2         -> int (default=64) : number of neurons in the second fully connected layer
    - gamma       -> float (default=0.99) : discount factor for future rewards
    - tau         -> float (default=0.01) : soft update parameter for target networks
    - H           -> float (default=0.1) : initial exploration noise level
    - decay_rate  -> float (default=0.999) : decay factor for exploration noise

    The agent includes support for:
    - Actor and target actor networks
    - Critic and target critic networks
    - Ensemble of actor policies (if ensemble_policies > 0)
    - Optional distillation of ensemble into a single student policy
    """

    def __init__(self,actor_dims,critic_dims,n_actions, ensemble_policies, agent_idx,chkpt_dir,
                 n_agents, alpha=0.01,beta=0.01,fc1=64,fc2=64,gamma=0.99,tau=0.01,H=0.1,
                 decay_rate=0.999,fc1_distill = 64, fc2_distill = 64, lr_distill= 1e-3):
        
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.ensemble_policies = ensemble_policies
        self.agent_name = 'agent_%s' % agent_idx        
        self.H = H
        self.decay_rate = decay_rate

        self.distilled_mode = False               #flag indicating if the agent is currently using a distilled (student) policy

        if ensemble_policies == 0:
            self.actor = [ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions,
                                       chkpt_dir=chkpt_dir, name=self.agent_name+'_actor_1')]
            self.target_actor = [ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions,
                                    chkpt_dir=chkpt_dir, name=self.agent_name+'_target_actor_1')]
            self.distiller = None 
        else:
            self.actor = [ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions,
                                       chkpt_dir=chkpt_dir, name=self.agent_name+'_actor_'+str(i+1)) 
                                       for i in range(ensemble_policies)]
            self.target_actor = [ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions,
                                    chkpt_dir=chkpt_dir, name=self.agent_name+'_target_actor_'+str(i+1)) 
                                    for i in range(ensemble_policies)]
            self.distiller = DSTL(lr_distill, actor_dims, fc1_distill, fc2_distill, n_actions, agent_idx,
                                  chkpt_dir=chkpt_dir,name=self.agent_name+'_student')
            
        self.critic = CriticNetwork(beta,critic_dims,fc1,fc2,self.n_agents,n_actions,
                                    chkpt_dir=chkpt_dir,name=self.agent_name+'_critic')
        self.target_critic = CriticNetwork(beta,critic_dims,fc1,fc2,self.n_agents,n_actions,
                                           chkpt_dir=chkpt_dir,name=self.agent_name+'_target_critic')
            
        self.update_critic_parameters()
        self.update_actor_parameters()
    
    def update_critic_parameters(self, tau = None):
        """ This function performs a soft update of the target critic networks using the current critic networks.
            The update is controlled by the parameter tau, which determines how much of the new critic parameters
            should influence the target critic. This is useful to stabilize training by slowly updating the target networks.
            
            PARAMETERS:
            - tau (float, optional): interpolation factor between old and new weights. If None, defaults to self.tau.
        """
        
        if tau == None:
            tau = self.tau

        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()

        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)

        updated_params = {}
        for name in critic_state_dict:
            #Soft update formula
            updated_params[name] = tau * critic_state_dict[name].clone() + (1 - tau) * target_critic_state_dict[name].clone()
            
        self.target_critic.load_state_dict(updated_params)

    def update_actor_parameters(self, tau=None, actor_idxs=None, target_actor_idxs=None):
        """ This function performs a soft update of the target actor networks using the current actor networks.
            The update is controlled by the parameter tau, which determines how much of the new actor parameters
            should influence the target actor. This is useful to stabilize training by slowly updating the target networks.

            PARAMETERS:
            - tau (float, optional): interpolation factor between old and new weights. If None, defaults to self.tau.
            - actor_idxs (list or int, optional): indices of the actor networks to update. If None, all are updated.
            - target_actor_idxs (list or int, optional): corresponding indices of the target actor networks to update.
        """
        
        if tau is None:
            tau = self.tau
        
        # Default: update all actors and their corresponding targets
        if actor_idxs is None:
            actor_idxs = list(range(len(self.actor)))
        if target_actor_idxs is None:
            target_actor_idxs = actor_idxs
        
        # Ensure inputs are lists
        if not isinstance(actor_idxs, (list, np.ndarray)):
            actor_idxs = [actor_idxs]
        if not isinstance(target_actor_idxs, (list, np.ndarray)):
            target_actor_idxs = [target_actor_idxs]
        
        if len(actor_idxs) != len(target_actor_idxs):
            raise ValueError("actor_idxs and target_actor_idxs must have the same length")
        
        for i, (a_idx, t_idx) in enumerate(zip(actor_idxs, target_actor_idxs)):
            actor_state_dict = dict(self.actor[a_idx].named_parameters())
            target_actor_state_dict = dict(self.target_actor[t_idx].named_parameters())

            # Soft update inside the loop because update is made for each actor separately (critic ones)
            updated_params = {}
            for name in actor_state_dict:
                updated_params[name] = tau * actor_state_dict[name].clone() + (1 - tau) * target_actor_state_dict[name].clone()
            
            self.target_actor[t_idx].load_state_dict(updated_params)      
        
    def choose_action(self, obs, evaluate=False, actor_idx=None):
        """
        Selects an action given an observation using one of the actor networks.
        If ensemble_policies > 0, an actor is selected from the ensemble (randomly or by actor_idx).
        If ensemble_policies = 0, the single actor network is used.

        INPUT:
        - obs : The observation (numpy array or sequence of arrays).
        - actor_idx (Optional) : index of actor to use (if None, chosen randomly).

        RRETURNS:
            action -> np.ndarray : The action chosen by the actor network (with exploration noise).
            actor_idx -> int : The index of the actor network used.

        NOTES:
        If ensemble_policies = 0, the agent still wraps the single actor network in a list,
        so this function will still work and always select the only available actor (index 0).
        """

        # Select an actor from the ensemble (randomly or based on actor_idx)
        if actor_idx is None:
            actor_idx = np.random.randint(len(self.actor))  

        actor = self.actor[actor_idx]
        state = (
            T.tensor(obs, dtype=T.float32, device=actor.device).unsqueeze(0)
            if isinstance(obs, np.ndarray)
            else T.tensor(np.concatenate([o.flatten() for o in obs]), 
                          dtype=T.float32, device=actor.device).unsqueeze(0)
        )

        # Compute action and add exploration noise
        actions = actor.forward(state)

        if not evaluate:
            noise = T.randn(self.n_actions).to(actor.device) * self.H
            self.H *= self.decay_rate
        else:
            noise = 0
        action = actions + noise

        return action.detach().cpu().numpy()[0], actor_idx

    def save_models(self):
        for actor, target_actor in zip(self.actor, self.target_actor):
            actor.save_checkpoint()
            target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self, student=False):
        if student and self.distiller is not None:
            self.distiller.student.load_checkpoint()
            self.actor = [self.distiller.student]
            self.target_actor = [self.distiller.student]
            self.distilled_mode = True
            self.critic.load_checkpoint()
            self.target_critic.load_checkpoint()
        elif self.distilled_mode:
            actor_path = f"{self.actor[0].chkpt_dir}/checkpoint_{self.agent_name.split('_')[-1]}_best.pt"
            critic_path = f"{self.critic.chkpt_dir}/checkpoint_{self.agent_name.split('_')[-1]}_critic.pt"
            self.actor[0].load_state_dict(T.load(actor_path))
            self.critic.load_state_dict(T.load(critic_path))
        else:
            for actor, target_actor in zip(self.actor, self.target_actor):
                actor.load_checkpoint()
                target_actor.load_checkpoint()
            self.critic.load_checkpoint()
            self.target_critic.load_checkpoint()
    
    def distill(self, dataset, path_file = 'RL_Multi_Agent/tmp/maddpg/distilled/agent'):
        """
        Distills the ensemble of actor networks into a single student actor network using the provided dataset:
        1) Creates the output directory if it doesn't exist
        2) Performs the distillation using the DSTL object
        3) Replaces the actor and target_actor with the distilled student network
        4) Sets the distilled_mode flag to True
        5) Saves the model to the specified path

        INPUT:
        - dataset : a dataset of observations and actions used for training the student policy via distillation
        - path_file -> str : path where the distilled model will be saved
        """

        os.makedirs(os.path.dirname(path_file), exist_ok=True)
        self.distiller.distill(dataset, path_file)
        self.actor = [self.distiller.student]           #declair as student
        self.target_actor = [self.distiller.student]
        self.distilled_mode = True
        
        self.save_models()
