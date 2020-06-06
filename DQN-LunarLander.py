import gym
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from collections import deque

cap=100000
batch_size=64
lr=0.01
tau=1e-3
max_ep=4
gamma=0.99
eps=1
n_ep=1500

class dqn(nn.Module):
    def __init__(self,seed,state_size,action_size,fc1_units=64,fc2_units=64):
        super(dqn,self).__init__()
        self.seed=torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
        
class Agent():
    def __init__(self,seed):
        self.memory=deque(maxlen=cap)
        self.qlocal=dqn(seed,state_size=8,action_size=4)
        self.qtarget=dqn(seed,state_size=8,action_size=4)
        self.seed=random.seed(seed)    
        
    def remember(self,state,action,reward,next_state,done):
        if(len(self.memory)>cap):
            self.memory.popleft()
        
            
        self.memory.append([state,action,reward,next_state,done])
        
    def select_action(self,state,eps):
        state = torch.from_numpy(state).float().unsqueeze(0)
        self.qlocal.eval()
        with torch.no_grad():
            action_values = self.qlocal(state)
        if(len(self.memory)%4==0):    
            self.train()
        
        
        if random.random() > eps:
            return (np.argmax(action_values).item())
        else:
            x=random.choice(np.arange(4))
            return x
    
    def sample(self):
        experiences=random.sample(self.memory,k=batch_size)
        states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).long()
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float()
  
        return (states, actions, rewards, next_states, dones)

    def train(self):
        if((len(self.memory)+1) >batch_size):
            states, actions, rewards, next_states, dones = self.sample()
            self.optimizer=optim.SGD(self.qlocal.parameters(),lr,momentum=0.5)
            Q_expected=self.qlocal(states).gather(1, actions)
            Q_next_state=self.qtarget(next_states).detach().max(1)[0].unsqueeze(1)
            Q_target=rewards+(gamma*Q_next_state*(1-dones))
            loss=F.mse_loss(Q_expected,Q_target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.update(self.qlocal,self.qtarget,tau)
            
    def update(self,local,target,tau):
         for target_param, local_param in zip(target.parameters(), local.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
agent=Agent(seed=0)
scores=deque(maxlen=100)
avg_rewards=deque(maxlen=100)
best_avg_reward=0

for n in range(0,n_ep):
    state=env.reset()
    done=False
    score=0
    while True:
        action=agent.select_action(state,eps)
        next_state,reward,done,_=env.step(action)
        score+=reward
        agent.remember(state,action,reward,next_state,done)
        state=next_state
        if done:
            eps=max(0.05,eps*0.995)
            break
    scores.append(score)



    avg_reward = np.mean(scores)
    if(avg_reward>best_avg_reward):
        best_avg_reward=avg_reward

    print("\rEpisode {}/{} || Best average reward {}".format(n+1, n_ep, avg_reward), end="")
    sys.stdout.flush()



            
