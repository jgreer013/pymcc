# pymcc
Python framework to develop models using Halo: MCC on PC

# Goals
Game AI is a fascinating and exciting area to me, but often disappointing. Most Game AI is built off of complex decision trees or Markov State Transitions, but both of those are cumbersome to create and fail to create AI which play similarly to people. Deep Learning and Deep Reinforcement Learning both seem like promising directions to go in, but so far, most environments for training these types of applications are restricted to toy programs. My goal is to build a tool which others can use to abstract the hard/boring parts - pulling the frame and controller data from any Windows PC Gaming application running, saving it, then being able to apply virtual input in real-time later.

To break this down more concretely:
* Create a tool/framework which allows one to record, train, and test models against any PC Game using an Xbox Controller
* Use framework to train models on recorded data
* Create a process for training models which can be applied to any game

# Recording the data
Various libraries exist which create environments for training against some games:
* [Grand Theft Auto V](https://pythonprogramming.net/game-frames-open-cv-python-plays-gta-v/)
* [Minecraft](https://github.com/microsoft/malmo/tree/master/MalmoEnv)
* [Starcraft 2](https://github.com/deepmind/pysc2)

However, there are some notable limitations:
* GTA V's gets very poor performance (12-13 fps at 800x600 resolution). While this can vary with hardware, that's way too low to play with - one could probably record data at that frequency, but it's very difficult to play complex games at such a framerate. Ideally we want to be able to play the game as we normally do (30-60 fps) while still getting a good amount of data (15-30 fps in recorded data, meaning we'll record only occasionally)
* The other libraries are all game-specific, with a lot of additional information being provided in the gym environments that a human wouldn't necessarily have access to.

## Getting a Baseline from someone else
The first approach is a step in the right direction though, so lets see what other libraries we can find...
* [Tensorkart](https://github.com/kevinhughes27/TensorKart)

Ah! Here's a great baseline for what we need - the recording, training, and playing are all nicely laid out; however, this one requires that we add additional software and libraries for playing on the emulator, and it's hard to judge performance when its only running an emulated game. Still, it's a good baseline, and our tool borrows heavily from it.

## Updating it to suite our needs
Once I found the baseline, I 
