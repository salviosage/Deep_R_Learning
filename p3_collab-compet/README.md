[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"


# Project 3: Collaboration and Competition

### Introduction

For this project, you will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the DRLND GitHub repository, in the `p3_collab-compet/` folder, and unzip (or decompress) the file. 

### Instructions

Follow the instructions in `Tennis.ipynb` to get started with training your own agent!  

### (Optional) Challenge: Crawler Environment

After you have successfully completed the project, you might like to solve the more difficult **Soccer** environment.

![Soccer][image2]

In this environment, the goal is to train a team of agents to play soccer.  

You can read more about this environment in the ML-Agents GitHub [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#soccer-twos).  To solve this harder task, you'll need to download a new Unity environment.  (**Note**: Udacity students should not submit a project with this new environment.)

You need only select the environment that matches your operating system:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Windows_x86_64.zip)

Then, place the file in the `p3_collab-compet/` folder in the DRLND GitHub repository, and unzip (or decompress) the file.  Next, open `Soccer.ipynb` and follow the instructions to learn how to use the Python API to control the agent.

(_For AWS_) If you'd like to train the agents on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agents without enabling a virtual screen, but you will be able to train the agents.  (_To watch the agents, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)


# Udacity Deep Reinforcement Learning Nanodegree
# Project 3: Collaboration and Competition

As part of the [Udacity Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) 
here you can read the report of the Project 3: Collaboration and Competition.

The challenge of this project is about training two learning agents, represented by rackets, to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. 
If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

![Training an agent to maintain its position at the target location for as many time steps as possible.](tennis.png)

### The Environment

The project environment is similar to, but not identical to the Tennis environment on the [Unity ML-Agents GitHub page](https://github.com/Unity-Technologies/ml-agents).
Unity ML-Agents is an open-source Unity plugin that enables games and simulations to serve as environments for training intelligent agents.

**Note:** The Unity ML-Agent team frequently releases updated versions of their environment. 
In this repository, the v0.4 interface has been used.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. 
Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, 
after taking the maximum over both agents). Specifically:

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. 
- This yields 2 (potentially different) scores. We then take the maximum of these 2 scores. 
- This yields a single score for each episode. The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

In this repository, the tennis environment has been solved with a DDPG algorithm.

## Learning Algorithm

### Agent implementation: Deep Deterministic Policy Gradient (DDPG)

For this project I implemented an *off-policy method* called **Deep Deterministic Policy Gradient**, you can read more about it in this paper: [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971). 

Deep Deterministic Policy Gradient (DDPG) is an algorithm which concurrently learns a Q-function and a policy. It uses off-policy data and the Bellman equation to learn the Q-function, and uses the Q-function to learn the policy.
It combines ideas from DPG (Deterministic Policy Gradient) and DQN (Deep Q-Network). It uses Experience Replay and slow-learning target networks from DQN, and it is based on DPG, which can operate over continuous action spaces.

By combining the actor-critic approach with the Deep Q Network method, the algorithm uses two networks:

- the **Actor** network, which proposes an action given a state
- the **Critic** network, which predicts if the action is good (positive value) or bad (negative value) given a state and an action.

These two networks are deployed alongside 2 more techniques:

- **2 Target networks**, which add stability to training by learning from estimated targets. The Target networks are updated slowly, hence keeping the estimated targets stable.
- **Experience Replay**, by storing list of tuples (state, action, reward, next_state), and instead of learning only from recent experience, the agent learns from sampling all of the experience accumulated so far.

![DDPG](ddpg.png)

In addition to this implementation of the DDPG algorithm, I experimented with and without increasing/decreasing two learning parameters, **gamma** (the discount factor for expected rewards)  and **tau** (the multiplicative factor for the soft update of target weights), inspired by this paper and using their formula explained there:

[How to Discount Deep Reinforcement Learning: Towards New Dynamic Strategies](https://arxiv.org/abs/1512.02011)
> Using deep neural nets as function approximator for reinforcement learning tasks have recently been shown to be very powerful for solving problems approaching real-world complexity. Using these results as a benchmark, we discuss the role that the discount factor may play in the quality of the learning process of a deep Q-network (DQN). When the discount factor progressively increases up to its final value, we empirically show that it is possible to significantly reduce the number of learning steps. When used in conjunction with a varying learning rate, we empirically show that it outperforms original DQN on several experiments. We relate this phenomenon with the instabilities of neural networks when they are used in an approximate Dynamic Programming setting. We also describe the possibility to fall within a local optimum during the learning process, thus connecting our discussion with the exploration/exploitation dilemma.

Increasing gamma after each episode:      gamma(n + 1) = gamma_final + (1 - gamma_rate) * (gamma_final - gamma(n))

Decreasing tau after each episode:     tau(n + 1) = tau_final + (1 - tau_rate) * (tau_final - tau(n))

Learning stability has been way better with this use of dynamic parameters, allowing to get higher and higher average scores even after many episodes of training (over 1500 for example): this is why I used these two formulas in the training function ddpg() in the Jupyter notebook "Tennis.ipynb".

## Code implementation

The code is organized in three files:

**model_ok.py** 

This file contains the **Actor** and the **Critic** class and each of them are then used to implement a "Target" and a "Local" Neural Network for training/learning. 

Here are the Actor and Critic network architectures:

```
Actor NN(
  (fc1): Linear(in_features=24(state size), out_features=164, bias=True, Batch Normlization, relu activation)
  (fc2): Linear(in_features=164, out_features=100, bias=True, relu activation)
  (out): Linear(in_features=100, out_features=2(action size), bias=True, tanh activation)
)
```

```
Critic NN(
  (fc1): Linear(in_features=24(state size), out_features=486, bias=True, Batch Normlization, relu activation)
  (fc2): Linear(in_features=164+2(action size), out_features=100, bias=True, relu activation)
  (out): Linear(in_features=100, out_features=1, bias=True, no activation function)
)
```    

**ddpg_agent_ok.py** 

Here you can find three classes, the (DDPG) Agent, the Noise and the Replay Buffer class, and a function called Load_and_test.

The (DDPG) class Agent contains 5 methods:
- constructor, which initializes the memory buffer and two instances of both Actor's and Critic's NN (target and local).
- step(), which allows to store a step taken by the RL agent (state, action, reward, next_state, done) in the Replay Buffer/Memory. Every four steps, it updates the target NN weights  with the current weight values from the local NN (Fixed Q Targets technique)
- act(), which returns actions for given state as per current policy through an Epsilon-greedy selection in order to balance exploration and exploitation in the Q Learning process
- learn(), which updates value parameters using given batch of experience tuples in the form of (state, action, reward, next_state, done) 
- soft_update(), which is used by the learn() method to softly updates the target NN weights from the local NN weights for both Actor and Critic networks

The ReplayBuffer class consists of Fixed-size buffer to store experience tuples (state, action, reward, next_state, done)  and contains these methods:
- add(), which adds a new experience tuple to memory
- sample(), which randomly sample a batch of experiences from memory for the learning process of the agent
- len(), which returns the current size of internal memory

The OUNoise class implements a Ornstein-Uhlenbeck process.
This is inspired by the [DDPG paper](https://arxiv.org/abs/1509.02971), where the authors use an Ornstein-Uhlenbeck Process to add noise to the action output.

The function Load_and_test allows you to run pretrained agents by loading the actor and critic weights obtained through the training process shown in the jupyter notebook.

**Tennis.ipynb**

This is the Jupyter notebook where I trained the agent. These are the steps taken in it:
  - Importing the necessary packages 
  - Examining the State and Action Spaces
  - Testing random actions in the Environment
  - Training the two DDPG agents
  - Ploting the training scores 

**workspace_utils.py**

In this file, based on the course material of one exercise presented in the [Udacity Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893), there is in particular one function that I used during the training of the agents in the jupyter notebook, which is the "keep_awake" function:
this allows you to iterate through a range (like iterating through the episodes during training) while maintaining the workspace up and running.

### Hyperparameters

Here are the hyperparameters that I used to train the two DDPG agents in the tennis environment:

```
n_episodes: 2000
max_t: 1000
random_seed: 0
gamma: 0.95
gamma_final: 0.99
gammma_rate: 0.02
tau: 0.008
tau_final: 0.001
tau_rate: 0.002
update_every: 15
num_updates: 12
buffer_size: int(1e6)
batch_size: 512
actor_fc1: 164
actor_fc2: 100
critic_fc1: 164
critic_fc2: 100
lr_actor: 1e-3
lr_critic: 2e-3
noise_theta: 0.12
noise_sigma: 0.07
noise_scale: 1
```

## Results

Here is the evolution of the score per episodes:

![Score](reward_scores_graph.png)

After **661 episodes** the 100 period moving average reached a score of **0.5043**, getting above the challenge's goal of at least +0.5.

The highest value of the average score has been achieved after **1923 episodes** with a score of **2.2129**, well beyond the project's requirement.

## Ideas for future work

Here are some ideas on further developments of the algorithm, beyond simply playing around with the presented architecture and hyperparameters tuning.

Other actor-critic algorithms proposed to solve this kind of environment can be found in these links:

[Distributed Distributional Deterministic Policy Gradients](https://openreview.net/pdf?id=SyZipzbCb)
> This work adopts the very successful distributional perspective on reinforcement learning and adapts it to the continuous control setting. We combine this within a distributed framework for off-policy learning in order to develop what we call the Distributed Distributional Deep Deterministic Policy Gradient algorithm, D4PG. We also combine this technique with a number of additional, simple improvements such as the use of N-step returns and prioritized experience replay. Experimentally we examine the contribution of each of these individual components, and show how they interact, as well as their combined contributions. Our results show that across a wide variety of simple control tasks, difficult manipulation tasks, and a set of hard obstacle-based locomotion tasks the D4PG algorithm achieves state of the art performance.

[Sample Efficient Actor-Critic with Experience Replay](https://arxiv.org/abs/1611.01224)
> This paper presents an actor-critic deep reinforcement learning agent with experience replay that is stable, sample efficient, and performs remarkably well on challenging environments, including the discrete 57-game Atari domain and several continuous control problems. To achieve this, the paper introduces several innovations, including truncated importance sampling with bias correction, stochastic dueling network architectures, and a new trust region policy optimization method.

[A2C - Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783v2)
> A2C, or Advantage Actor Critic, is a synchronous version of the A3C policy gradient method. As an alternative to the asynchronous implementation of A3C, A2C is a synchronous, deterministic implementation that waits for each actor to finish its segment of experience before updating, averaging over all of the actors. This more effectively uses GPUs due to larger batch sizes.

[A3C - Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)
> We propose a conceptually simple and lightweight framework for deep reinforcement learning that uses asynchronous gradient descent for optimization of deep neural network controllers. We present asynchronous variants of four standard reinforcement learning algorithms and show that parallel actor-learners have a stabilizing effect on training allowing all four methods to successfully train neural network controllers. The best performing method, an asynchronous variant of actor-critic, surpasses the current state-of-the-art on the Atari domain while training for half the time on a single multi-core CPU instead of a GPU. Furthermore, we show that asynchronous actor-critic succeeds on a wide variety of continuous motor control problems as well as on a new task of navigating random 3D mazes using a visual input.

Alternatively, the Multi Agent RL approach can be explored. In particular, the idea of a multi agent DDPG algorithm as described here:

[Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://papers.nips.cc/paper/2017/file/68a9750337a418a86fe06c1991a1d64c-Paper.pdf)
> We explore deep reinforcement learning methods for multi-agent domains. We begin by analyzing the difficulty of traditional algorithms in the multi-agent case:
Q-learning is challenged by an inherent non-stationarity of the environment, while policy gradient suffers from a variance that increases as the number of agents grows.
We then present an adaptation of actor-critic methods that considers action policies of other agents and is able to successfully learn policies that require complex multiagent coordination. Additionally, we introduce a training regimen utilizing an ensemble of policies for each agent that leads to more robust multi-agent policies. We show the strength of our approach compared to existing methods in cooperative as well as competitive scenarios, where agent populations are able to discover various physical and informational coordination strategies.
