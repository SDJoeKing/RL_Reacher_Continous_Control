import numpy as np
import random
from collections import deque

from model import Actor, Critic, tdActor

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
class TD3_agent:
    def __init__(self, states = 33, actions = 4, gamma = 0.99, lr = 0.0003, tau = 0.005, action_noise = 0.1, policy_smooth_noise = 0.2, noise_clip = 0.5, policy_delay = 2, batch_size = 128, update_every = 1):
        
        ## Critics 
        self.qnet1 = Critic(states, actions).to(device)
        self.qtarget1 = Critic(states,actions).to(device)
        self.q1_optimizer = optim.Adam(self.qnet1.parameters(), lr = lr)
        self.soft_update(1, self.qtarget1, self.qnet1)
        
        self.qnet2 = Critic(states, actions).to(device)
        self.qtarget2 = Critic(states,actions).to(device)
        self.q2_optimizer = optim.Adam(self.qnet2.parameters(), lr = lr)
        self.soft_update(1, self.qtarget2, self.qnet2)
        
        ## policies
        self.policy = tdActor(states, actions).to(device)
        self.policy_target = tdActor(states, actions).to(device)
        self.soft_update(1, self.policy_target, self.policy)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr = lr)
        
        self.q_criteria = torch.nn.MSELoss()
        
        ## Replay buffer
        self.replay = EReplay(int(1e4), batch_size = batch_size)
        
        ## uncorrelated gaussian noise 
        self.action_noise = action_noise
        self.policy_smooth_noise = policy_smooth_noise
        
        ## update parameters
        self.action_size = actions
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.steps = 0
        self.update_every = update_every
        self.noise_clip = noise_clip
        
        ## noise samplers
        self.sampler =  Normal(torch.zeros(batch_size, actions), self.policy_smooth_noise * torch.ones(batch_size, actions))
        self.action_sampler = Normal(torch.zeros(actions), self.action_noise * torch.ones(actions))
                
    def act(self, state, noise  =True):
        if type(state) == np.ndarray:
            state = torch.from_numpy(state).float().to(device)
        actions = self.policy(state)
        if noise:
            actions += self.action_sampler.sample().to(device)
            actions = torch.clamp(actions, -1, 1)
        
        return actions
        
    def step(self, states, actions, rewards, next_states, dones):
        # Save experience in replay memory
        self.replay.add(states, actions, rewards, next_states, dones)

        # If enough samples are available in memory, get random subset and learn
        if self.replay.ready():
    
            ## time to gradient descent
            if self.steps % self.update_every == 0:
                cnt = 0
                for _ in range(self.update_every):
                    experiences = self.replay.sample()
                    self.learn(experiences, cnt)    
                    cnt += 1
            self.steps += 1

    def learn(self, experiences, cnt):
        
        states, actions, rewards, next_states, dones = experiences
        
        with torch.no_grad():
            sampled_actions = self.policy_target(next_states)
            sampled_actions =torch.clamp(sampled_actions+torch.clamp(self.sampler.sample().to(device), -self.noise_clip, self.noise_clip), -1, 1) 
            q_target_1 = self.qtarget1(next_states, sampled_actions)
            q_target_2 = self.qtarget2(next_states, sampled_actions)
        
        y = self.gamma * torch.min(q_target_1, q_target_2) * (1-dones) + rewards
        
        ## -------------- update critics --------------- ##
        q_expected_1 = self.qnet1(states, actions)
        q_expected_2 = self.qnet2(states, actions)
        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        
        q1_loss = self.q_criteria(q_expected_1, y)
        q2_loss = self.q_criteria(q_expected_2, y)
        
        q1_loss.backward()
        q2_loss.backward()
        
        ## gradient clipping
        torch.nn.utils.clip_grad_norm_(self.qnet1.parameters(), 1)
        torch.nn.utils.clip_grad_norm_(self.qnet2.parameters(), 1)
        
        self.q1_optimizer.step()
        self.q2_optimizer.step()  
        
        
        ## --------------- update policy ----------------##
        if cnt % self.policy_delay == 0:
            
            self.policy_optimizer.zero_grad()
            
            p_loss = -self.qnet1(states, self.policy(states)).mean()
            
            p_loss.backward()
            
#             torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 3)
            self.policy_optimizer.step()
            
            # soft update
            self.soft_update(self.tau, self.qtarget1, self.qnet1)
            self.soft_update(self.tau, self.qtarget2, self.qnet2)     
            self.soft_update(self.tau, self.policy_target, self.policy)
            
    @staticmethod
    def soft_update(tau, target_net, local_net):
        for target_param, local_param in zip(target_net.parameters(),  local_net.parameters()):
            target_param.data.copy_( tau * local_param.data + (1-tau) * target_param.data)  


## Normal experience replay, to be upgraded to be priority experience replay
class EReplay:
    def __init__(self, size, action_size = 4, batch_size = 256):
        self.memory = deque(maxlen = size)
        self.batch_size = batch_size
        self.action_size = action_size 
    
    
    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self):
        experience = random.sample(self.memory, k = self.batch_size)
        states, actions, rewards, next_states, dones = zip(*experience)
        
        states = torch.from_numpy(np.array(states)).float().to(device) 
        try:
            actions = torch.from_numpy(np.array(actions)).float().to(device) 
            rewards = torch.from_numpy(np.array(rewards)).float().to(device).unsqueeze(1)
            next_states = torch.from_numpy(np.array(next_states)).float().to(device) 
            dones = torch.from_numpy(np.array(dones)).float().to(device).unsqueeze(1)
        except:
            print(actions)        
        return (states, actions, rewards, next_states, dones)
        
    def ready(self):
        return len(self.memory) >= self.batch_size
    
    
    def __len__(self):
        return len(self.memory)


        
        