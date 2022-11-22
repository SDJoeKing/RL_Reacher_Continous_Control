## Report

### Learning Algorithm
In this project, a combination of `Double DQN` and `Dueling DQN` is implemented. Double DQN utlises a target neural net (TNN) to evaluate the local neural net (LNN). 
The implementation was following this [paper](https://arxiv.org/abs/1509.06461), such that the esitmated action that generate maximum Q value is selected using LNN, TNN then evaluate the actions generating the targeted Q values for update.

The implementation of `Dueling DQN` follows this [paper](https://arxiv.org/abs/1511.06581), where the underlying NNs have parallel branches generating estimated value function value and action values for each action. The output of the NNs are addition of the two, given the final estimated Q values. However, as described in the paper, the maximum of the advantages (i.e. estimated action values) should be cancelled. This has been implemented.

Other minor considerations include: `delayed updating` TNN but update LNN when replay buffer is full; `gradient clipping` to (-1, 1). Latter has shown significant benefits in practical training.

### Hyper parameters selection
For the agent, I have used the following parameters:
  ```
   batch_size = 64
   gamma = 0.995,
   lr = 1e-4, 
   update_every = 5, 
   tau = 0.01 for soft update
   optimizer = optim.Adam
   loss_function = MSELoss
   Replay_buffer_size = 1e5
  ```
For training, the only selection needs consideration is the decay of epsilon, which I chose 0.995 so that by about middle of the training (1000 eposide) the epsilon has already been reduced to small (0.006) but not completely reject exploration.
The maximum length per episode is limited to 2000 iterations. 
  ```
  episodes = 2000,
  eps = 1, 
  eps_end = 0.001, 
  decay = 0.995
  max_iter = 2000, 
  print_every = 50, 
  term_reward = 23  
  ```
### Neural Network architecture
For the underlying NN models, it is defaulted to have two hidden layers, both `64` dimensions. A `dropout layer` with probability of `0.1` has been added after each activation of hidden layers.
The activation function is `ReLU`.

### Results discussion
The training went very successful, resulting a solved environment in approximately `1200 episodes`, giving maximum undiscounted score of `24`, averaged over 10 episodes reward of `17.2`. This result is significantly improved on the bench mark results of Udacity which solves the environment at approximately 1700 eposides, with less averaged score.

**From the figure of result below, it is evident that the episodes training required to achieve average score of +13 are approximately 500~550 episodes.**

![reward](https://user-images.githubusercontent.com/69092110/201941056-a96298aa-17ab-444e-a22e-f8fb368feec2.png)

### Future improvements
Many improvements can be made, firstly I can implement a priority experience replay buffer, using python native priority queue and rank based priority replay. The rest of improvements method included in `rainbow` can also boost the training performance.
