from torch import nn
import torch
import numpy as np

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self, input_dim=2, hidden_num=128, output_dim=1, binary=False, positive=False, activation="relu"):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_num, bias=True)
        self.fc2 = nn.Linear(hidden_num, hidden_num//2, bias=True)
        self.fc3 = nn.Linear(hidden_num//2, output_dim, bias=True)
        if activation=="relu":
            self.act = lambda x: torch.relu(x)
        elif activation=="sigmoid":
            self.act = lambda x: torch.sigmoid(x)
        self.binary = binary
        self.positive = positive
    def forward(self, x_input):
        inputs = x_input
        x = self.fc1(inputs)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        if self.binary:
            x = torch.sigmoid(x)
        elif self.positive:
            x = torch.relu(x)
        return x
    

class RectifiedFlow():
    def __init__(self, model=None, num_steps=1000):
        self.model = model
        self.N = num_steps
    def get_train_tuples(self, z0=None, z1=None):
        t = torch.rand((z1.shape[0], 1))
        z_t =  t * z1 + (1.-t) * z0
        target = z1 - z0 
        return z_t, t, target
    @torch.no_grad()
    def sample_ode(self, z0=None, N=None):
        ### NOTE: Use Euler method to sample from the learned flow
        if N is None:
            N = self.N    
        dt = 1./N
        traj = [] # to store the trajectory
        z = z0.detach().clone()
        batchsize = z.shape[0]
        traj.append(z.detach().clone())
        for i in range(N):
            t = torch.ones((batchsize,1)) * i / N
            pred = self.model(torch.cat([z, t], dim=1))
            z = z.detach().clone() + pred * dt
            traj.append(z.detach().clone())
        return traj


def train_rectified_flow(rectified_flow, optimizer, pairs, batchsize, inner_iters):
    loss_curve = []
    for i in range(inner_iters+1):
        optimizer.zero_grad()
        indices = torch.randperm(len(pairs))[:batchsize]
        batch = pairs[indices]
        z0 = batch[:, 0].detach().clone()
        z1 = batch[:, 1].detach().clone()
        z_t, t, target = rectified_flow.get_train_tuples(z0=z0, z1=z1)
        pred = rectified_flow.model(torch.cat([z_t, t], dim=1))
        loss = (target - pred).view(pred.shape[0], -1).abs().pow(2).sum(dim=1)
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        loss_curve.append(np.log(loss.item())) ## to store the loss curve
    return rectified_flow, loss_curve


class ConditionalRectifiedFlow():
    def __init__(self, model=None, num_steps=1000):
        self.model = model
        self.N = num_steps
    def get_train_tuples(self, z0=None, z1=None, c1=None):
        t = torch.rand((z1.shape[0], 1))
        z_t =  t * z1 + (1.-t) * z0
        target = z1 - z0 
        return z_t, t, target
    #@torch.no_grad()
    def sample_conditional_ode(self, z0=None, c1=None, N=None):
        ### NOTE: Use Euler method to sample from the learned flow
        if N is None:
            N = self.N    
        dt = 1./N
        traj = [] # to store the trajectory
        z = z0.clone()
        batchsize = z.shape[0]
        traj.append(z.clone())
        for i in range(N):
            t = torch.ones((batchsize,1)) * i / N
            pred = self.model(torch.cat([z, c1, t], dim=1))
            z = z.clone() + pred * dt
            traj.append(z.clone())
        return traj


def train_conditional_rectified_flow(rectified_flow, optimizer, pairs, batchsize, inner_iters):
    loss_curve = []
    n = pairs[0].shape[0]
    for i in range(inner_iters+1):
        optimizer.zero_grad()
        indices = torch.randperm(n)[:batchsize]
        # batch = pairs[indices]
        z0 = pairs[0][indices].detach().clone()
        z1 = pairs[1][indices].detach().clone()
        c1 = pairs[2][indices].detach().clone()
        z_t, t, target = rectified_flow.get_train_tuples(z0=z0, z1=z1)
        pred = rectified_flow.model(torch.cat([z_t, c1, t], dim=1))
        loss = (target - pred).view(pred.shape[0], -1).abs().pow(2).sum(dim=1)
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        loss_curve.append(np.log(loss.item())) ## to store the loss curve
    return rectified_flow, loss_curve
