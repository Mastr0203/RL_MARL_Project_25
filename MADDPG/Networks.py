import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CriticNetwork(nn.Module):
    """ Critic network return the estimated Q(s;a1,...,an) value, i.e. a measure of the quality of the action performed
        by each agent in a joint state. Critic is global, hence takes complete state of the enviroment and returns a value

        INPUT:
        - beta -> float : is the learning rate of the critic (used by Adam optimizer)
        - input_dims -> int : sum of observations of all agents
        - fc1_dims -> int : number of neurons of first fully connected layer
        - fc2_dims -> int : number of neurons of second fully connected layer
        - n_agents -> int : number of agents
        - n_actions -> int : dimension of action vector for each agent
        - name -> str : file name to store weights
        - chkpt_dir -> str : directory of storage
        DIMENSIONS:

        - input dimension = input_dims + n_agents*n_actions (n_actions can differ between agents)
        - output dimension = 1
    """

    def __init__(self,beta,input_dims,fc1_dims,fc2_dims,n_agents,n_actions,name,chkpt_dir):
        super(CriticNetwork,self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir,name)
        
        self.fc1 = nn.Linear(input_dims+n_agents*n_actions,fc1_dims)
        self.fc2 = nn.Linear(fc1_dims,fc2_dims)
        self.q = nn.Linear(fc2_dims,1)

        self.optimizer = optim.Adam(self.parameters(),lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self,state,action):
        x = F.relu(self.fc1(T.cat([state,action],dim=1)))
        x = F.relu(self.fc2(x))
        q = self.q(x)

        return q
    
    def save_checkpoint(self):
        os.makedirs(os.path.dirname(self.chkpt_file), exist_ok=True)
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))


class ActorNetwork(nn.Module):
    """ 
    Actor network that represents the deterministic policy of a single agent. Given the observation of 
    the agent, it outputs a continuous action using a fully connected neural network followed by a tanh 
    activation to ensure the output is within the action bounds.

    INPUT:
    - alpha -> float : learning rate for the actor (used by Adam optimizer)
    - input_dims -> int : dimension of the observation space for the agent
    - fc1_dims -> int : number of neurons in the first fully connected layer
    - fc2_dims -> int : number of neurons in the second fully connected layer
    - n_actions -> int : dimension of the output action vector
    - name -> str : file name to store weights
    - chkpt_dir -> str : directory path where the model will be saved
    
    DIMENSIONS:
    - input dimension = input_dims
    - output dimension = n_actions (bounded in [-1,1] by tanh)
    """

    def __init__(self,alpha,input_dims,fc1_dims,fc2_dims,n_actions,name,chkpt_dir):
        super(ActorNetwork,self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir,name)

        self.fc1 = nn.Linear(input_dims,fc1_dims)
        self.fc2 = nn.Linear(fc1_dims,fc2_dims)
        self.pi = nn.Linear(fc2_dims,n_actions)

        self.optimizer = optim.Adam(self.parameters(),lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self,state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = T.tanh(self.pi(x))          #value, not probability   

        return pi

    def save_checkpoint(self):
        os.makedirs(os.path.dirname(self.chkpt_file), exist_ok=True)
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))