MARL Example
------
## Prerequisites
- Conda or Miniconda
## Setup
To setup the environment, create a conda environment in Python 3.13:
```
conda create -n marl python=3.13
```
Activate the environment, and then install the required libraries like so:
```
conda activate marl
pip install -r requirements.txt
```
You should now be able to run the scripts includes in the common directory.
## Running Tests
To test that the environment is working, we can run test.py from the common directory. Navigate to common and run
```
python test.py
```
This script will launch a pygame window to display the MARL environment. It defaults to an environment running 3 agents simultaneously, each making randomly sampled actions.
## Running main.py
The main.py script in common is used for both training and evaluation. It can be run with the following arguments:
```
-t/--train
```
Indicates that training should be run. By default, this flag is off, which will run main in evaluate mode.
```
-c/--checkpoint [best/recent]
```
Loads a checkpoint from the specified directory. If included with the -t flag, it is used to resume training from the specified checkpoint. Required if -t is not specified. The flags -n, -k, --fc1, and --fc2 must match the original values for the checkpoint specified.
```
-r/--random_episodes [n]
```
Used to change the number of episodes at the beginning of training in which actions are chosen at random in order to fill the replay buffer. The default is 1000.
```
-d/--duration [n]
```
Used to change the number of episodes over which to train. The default is 50001.
```
-n/--num_agents [n]
```
Used to change the number of agents in the simulation. Set to 3 by default.
```
-k/--k_near_agents [n]
```
Used to change the number of neighbors included in each agent's observation space. This value must be less than num_agents. By default, this is set to 2.
```
-m/--minibatch_size [n]
```
Used to change the number of minibatches used during training. This has no effect if -t is not enabled. The default value is 64.
```
-e/--eval
```
Used for debugging. Will print agent actions to the terminal at each step. By default, this flag is off.
```
--fc1 [n]
```
Used to change the size of the first inner layer of the neural network. Default size is 64.
```
--fc2 [n]
```
Used to change the size of the second inner layer of the neural network. Default size is 64.
```
--alpha [n]
```
Used to change the alpha parameter for MADDPG during training. Default value is 0.01.
```
--beta [n]
```
Used to change the beta parameter for MADDPG during training. Default value is 0.01,
```
--gamma [n]
```
Used to change the gamma parameter for MADDPG during training. Default value is 0.99.
```
--tau [n]
```
Used to change the tau parameter for MADDPG during training. Default value is 0.01.
