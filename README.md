### General
This repo contain a solution for cartpole-v0 from OpenAI, based on simple MLP model using Tensorflow.
A lot of games are played, and only the best one are saved to train the network.

Information about the game environment : https://github.com/openai/gym/wiki/CartPole-v0

I know it's maybe not big deal, but i'm proud that the NN size is configurable and easily readable as a tensorflowgraph


### cartpole-v0_try_[...].py
This repo contain all my different tries in order to solve cartpole-v0 from OpenAI
**/!\ Only the final try, cartpole-v0-try_4(reboot).py resolve this game.**

It's called reboot because I started it from scratch, without using nearly anything from old tries.

I would like to create a new version more object-oriented, with more than 1 file.

### Configure the solution
In order to configure the file please go to line 400
Can be configured :
- number_of_episodes (how many time will be run the game environment)
- max_steps (how many steps will be played in one run)
- learning_rate (of the gradient descent)
- dropout_rate
- io_size (input layer and output layer size)
- hidden_layer_size
- number_of_games (how many time the NN will play)
- win_limit (number of steps to consider a game is won) 
- render (open a window rendering the game)

### conda_env.txt
list all package installed in my conda environment
