import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim


WEIGHT_DECAY = 0


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class Agent():
    """
    DDPG agent class
    """ 
    def __init__(self, state_size, action_size,
        num_agents, random_seed = 0, buffer_size = int(1e6), batch_size = 256,
        lr_actor = 1e-4, lr_critic = 1e-4,
        noise_theta = 0.2, noise_sigma = 0.2,
        actor_fc1 = 256, actor_fc2 = 100, actor_fc3 = 32,
        critic_fc1 = 256, critic_fc2 = 100, critic_fc3 = 1,
        update_every = 1, num_updates = 1):
        
        
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(random_seed)
        self.batch_size = batch_size
        self.update_every = update_every
        self.num_updates = num_updates
        self.buffer_learn_size = min(batch_size * 10, buffer_size)

        
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed, actor_fc1, actor_fc2).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed, actor_fc1, actor_fc2).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed, critic_fc1, critic_fc2).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed, critic_fc1, critic_fc2).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=WEIGHT_DECAY)

        
        self.noise = OUNoise((num_agents, action_size), random_seed, theta=noise_theta, sigma=noise_sigma)

        self.memory = ReplayBuffer(action_size, buffer_size, self.batch_size, random_seed)

        # Initialize the time step counter for updating each UPDATE_EVERY number of steps)
        self.t_step = 0
    
    def step(self, states, actions, rewards, next_states, dones, gamma = 0.96, tau = 0.001):
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)
 
        # Learn every update_every time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.buffer_learn_size:
                for _ in range(self.num_updates):
                    experiences = self.memory.sample()      
                    self.learn(experiences, gamma, tau)

    def act(self, state, add_noise=True, noise_scale=1.0):
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += (noise_scale * self.noise.sample())
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma, tau):
        """
        Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, tau)
        self.soft_update(self.actor_local, self.actor_target, tau)                     

    def soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Weighted average. Smaller tau means more of the updated target model is
            weighted towards the current target model.
        local_model: PyTorch model (weights will be copied from)
        target_model: PyTorch model (weights will be copied to)
        tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.size = size
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)

        self.state = x + dx
        return self.state

class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)
    

    
    
def load_and_test(agent, env, actor_weights, critic_weights, n_episodes):
    
    # load the weights from file
    agent.actor_local.load_state_dict(torch.load(actor_weights))
    agent.critic_local.load_state_dict(torch.load(critic_weights))
    
    # Get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    total_score = 0.0
    for i in range(n_episodes):

        # Reset the environment
        env_info = env.reset(train_mode=False)[brain_name] 

        # Get the current state
        state = env_info.vector_observations

        # Initialize the scores
        score = np.zeros(agent.num_agents)
        
        while True:
            
            # Choose actions.
            # Don't forget to make add_noise False when not training.
            action = agent.act(state, add_noise = False)
            
            # Send actions to the environment
            env_info = env.step(action)[brain_name]
            
            # Get the next state
            next_state = env_info.vector_observations
            
            # Get rewards
            reward = env_info.rewards
            
            # Check if the episode is finishised
            done = env_info.local_done
            
            # Add rewards to the scores
            score += reward
            
            # Replace the current state with the next state for the next timestep
            state = next_state
            
            # Exit the loop if the episode is finished
            if np.any(done):
                break

        print("Episode {}\tScore: {:.2f}".format(i + 1, np.mean(score)))
        total_score += np.mean(score)
    print("Average score over {} episodes: {:.2f}".format(n_episodes, total_score / n_episodes))
    
