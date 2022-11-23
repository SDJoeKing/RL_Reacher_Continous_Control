## Report

### Learning Algorithm
In this project, TD3 ([Twin Delayed DDPG](https://spinningup.openai.com/en/latest/algorithms/td3.html)) was implemented. The implementation was following instructions from [Spinning Up](https://spinningup.openai.com/en/latest/algorithms/td3.html)). Some tricks including `Clipped double Q learning`, `Delayed policy update` and `Target policy smoothing` are implemented to stablise the training in comparison to standard DDPG algorithm. 

 Clipped double Q learning resembles the double-q learning concepts which utilises two Q networks, and another two targeted Q nets for each. When evaluating the targeted Q value, the smaller Q values from the targeted Q nets are used to prevent Q nets exploiting ocassional large Q values. Delayed policy udpate entails updating policy nets only after several Q nets update to stablise the policy learning. And finally the targeted policy smoothing refers to adding uncorrelated Gaussian noises on targeted actions when computing the target actions from target policy. Noise is also added to actions generated from policy during training, to encourage exploration.
 
 I have also experimented with `clip_grad_norm_` to limit the Q networks' gradients to prevent gradient explosion. The same with Policy gradient ascent however did not work well during trials. 

### Hyper parameters selection
For the agent, I have used the following parameters:
  ```
states = 33, 
actions = 4, 
gamma = 0.99, 
lr = 0.0003, 
tau = 0.005, 
action_noise = 0.1, 
policy_smooth_noise = 0.2, 
noise_clip = 0.5, 
policy_delay = 2, 
batch_size = 128, 
update_every = 1
  ```
The `update_every` is set to `1` as when experimenting with different value it did not generally work well, perhaps limited by the fact that I didn't implement warm-up episodes for Replay Buffer to save enough experiences. This can be something to explore in the future.

For training, a action noise of `0.1` is added for exploration and `0.2` clipped by `0.5` for target policy smoothing. 

The maximum length per episode is limited to 1000 iterations. 

For training function:
  ```
episodes = 200, 
print_every = 10, 
term_reward = 30 
  ```
### Neural Network architecture
For the underlying Actor NN models, it is defaulted to have two hidden layers, both `256` dimensions. The activation function is `ReLU` in between layers, and a `Tanh` function at the output to limit the policy to action space of (-1, 1).

For Critic NN model it is fairly straight forward, similar to Acotr NN with two `256` dimension hidden layers. 

### Results discussion
The training with a single Agent solved the environment in approximately `68` episodes, giving undiscounted score of `30`, and remained an averaged score of above `30` for over `100` episodes. This result is on par with the bench mark result from Udacity (solved environment in 63 episodes). 

**From the figure of result below, it is evident that the episodes training required to achieve average score of +30 are 68 episodes. The training outputs also valided that**

 

### Future improvements
As discussed above, firstly I can try implementing a warm-up routine to gather sufficient experiences in buffer before update, so that the `update_every` setting can be more meaningful. Besides, I have tried SAC (not included in this repo) but it did not work quite as I would hoped. It is worth finding out the reason (implementation and understanding of the algorithm). I would also like to try the `20` Agent environment and experiment with `PPO` and `TRPO`. 
