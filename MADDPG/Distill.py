import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torch.utils.data import Dataset, DataLoader

class Net(nn.Module):
    def __init__(self,alpha,input_dims,fc1_dims,fc2_dims,n_actions,chkpt_dir,name):
        super(Net,self).__init__()

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
        pi = T.tanh(self.pi(x))       

        return pi
    
    def save_checkpoint(self):
        os.makedirs(os.path.dirname(self.chkpt_file), exist_ok=True)
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))
    
class DistillationDataset(Dataset):
    def __init__(self, data, agent_idx):
        self.states = T.stack([T.tensor(s[agent_idx], dtype=T.float32) for s, _ in data])
        self.actions = T.stack([T.tensor(a[agent_idx], dtype=T.float32) for _, a in data])

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]


class DSTL():
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, agent_idx,chkpt_dir,name):
        self.student = Net(alpha, input_dims, fc1_dims, fc2_dims, n_actions,chkpt_dir,name)
        self.agent_idx = agent_idx

    def distillation_loss(self, states, target_actions):
        pred_actions = self.student(states)
        return F.mse_loss(pred_actions, target_actions)

    def distill(self, dataset, path_file, n_epochs=50, batch_size=64):
        if dataset is None or len(dataset) == 0:
            return
        distill_dataset = DistillationDataset(dataset, self.agent_idx)
        dataloader = DataLoader(distill_dataset, batch_size=batch_size, shuffle=True)
        self.student.train()

        for epoch in range(n_epochs):
            for states, target_actions in dataloader:
                loss = self.distillation_loss(states, target_actions)
                self.student.optimizer.zero_grad()
                loss.backward()
                self.student.optimizer.step()