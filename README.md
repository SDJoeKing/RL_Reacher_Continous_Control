# Udacity DRL Project 2 - Continuous Control of Reacher

### Introduction

For this project, an agent has been trained to control a 2 joint reacher robotic arm to track and stay in a target area. The simulating was based on [Unity ML-Agents Toolkit](https://github.com/Unity-Technologies/ml-agents). The environment was compiled based on toolkit version 0.4, provided by Udacity.

#### Environment introduction
1. State and action space: The state space is `33`, values corresponding to position, rotation, velocity, and angular velocities of the arm. The action space is `4`, all continuous within range of `(-1, 1)`, corresponding to torques applicable to two joints.

2. Reward is `+0.1` for each step that the agent's hand is in the goal location, the goal location is changing in various patterns
3. `Success criteria`: to solve the environment, the agent must get an average score of +30 over 100 consecutive episodes.
4. Bench implemented by Udacity: the bench agent training performance provided by Udacity indicates maximum socre approximately 37, training episodes at circa 63 eposides. 

### Getting Started adapted from Udacity instructions (this project is based on 1 Agent training)

1. Copy this github repo:   
    ```
    git clone https://github.com/SDJoeKing/Value_based_navigation_reinforcement_learning.git
    ```
3.  Download the environment from one of the links below.  You need only select the environment that matches your operating system:

    - **_Version 1: One (1) Agent_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

    - **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
2. Place the file in the downloaded repo, and unzip (or decompress) the files. 
3. Activate the Python 36 virtual environment;
    ```
    At Linux terminal:
    source ~/your_python_36_env/bin/activation
    pip install numpy matplotlib scipy ipykernel pandas
    ```
5. While with python 36 activated, change to a different folder and setup the dependencies (noted gym environment is not required for this project):
     ```
    cd new_directory
    git clone https://github.com/udacity/deep-reinforcement-learning.git
    cd python
    pip install .
    ```
5. Due to platform and cuda restrictions, the requirement from step 5 of torch==0.4.0 can no longer be satisfied in my machine. Instead I have losen the restriction to any torch versions that are suitable for current machine. The rest of the requirements remain unchanged and satisfied (including the unityagents version requirement). 
6. Check `Navigation_training.ipynb`

### Instructions
In `Continuous_Control.ipynb`, the training process was demonstrated. The `agent.py` and `model.py` contains codes for RL agent and the backend neural net architecture, respectively.  A report.md file is also included explaining the project.

