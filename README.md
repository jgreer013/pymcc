# pymcc
Python tool/framework to develop models using Halo: MCC on PC

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

Ah! Here's a great baseline for what we need - the recording, training, and playing are all nicely laid out, and since it captures the window on the desktop, it's already able to pull frame data from any Windows application, just as we want, and pulls the controller data as we need it to; however, this one requires that we add additional software and libraries for playing on the emulator, and it's hard to judge performance when its only running an emulated game. Still, it's a good baseline, and our tool borrows heavily from it.

## Updating it to suite our needs
Once I found the baseline, it's time to alter it a bit. Halo is a bit more complex of an input space than Mario Kart, so we'll want to read all of the controller inputs, not just a few of them. There's a few things we need to do:
* Pulling full controller data from the active controller
* Performance Improvements
* Pushing virtual controller data to the active window (i.e. the game application)

Pulling the full controller data is fairly straightforward, as we just need to update the XboxController class to return the full vector on read().

Performance improvements are a bit trickier though. The libraries that TensorKart is using are already fairly low-level and performant on Windows, but we still have a very low sample rate, and our machine can't seem to squeen any more out by increasing it. To get a hint, we need to take a look at the Task Manager to see what's causing the issue:

INSERT IMAGE HERE

Hmm, interesting, it appears that our machine's actually handling things pretty well - our RAM, CPU, GPU, and Disk are all at fairly low levels, even while running Halo and recording data. Well, if the machine's not being overwhelmed, why doesn't increasing the sample rate do anything? If we think about the traditional bottlenecks in software and hardware, IO tends to be a fairly big bottleneck, as reading and writing to disk take the most time, even on an M.2 SSD.

Hope is not lost though - since the CPU and RAM have plenty of room to be used still, we can actually just parallelize the writing of the frame and controller, which is our biggest IO bottleneck rather than the game itself. How do we do that? [Multiprocessing!](https://docs.python.org/3/library/multiprocessing.html)

### Multiprocessing
The fix is fairly straightforward, all we need to do is put our data in an in-memory queue instead of writing it immediately. Then, we need processes which are registered to pull from the queue, which will then dequeue our data and write it to disk.

![image](https://user-images.githubusercontent.com/18727435/125725169-174bdc65-fb01-4c63-a11e-2b6edf3d5395.png)

![image](https://user-images.githubusercontent.com/18727435/125725233-f66ceed2-0cc6-4c07-99eb-6f149c1bb5c5.png)

![image](https://user-images.githubusercontent.com/18727435/125725193-4cd54ff4-aca2-4fcd-bc7a-064c600d0382.png)

![image](https://user-images.githubusercontent.com/18727435/125725207-f71b0bf1-ae81-4595-a2b7-ccd5ce997fae.png)

This allows the recording process to continue with minimal lag, and our sample rate can scale based on the number of processes that are pulling data from the queue and writing it to disk as saved screenshots. Depending on hardware, there would need to be a balance between the number of processes - if there aren't enough processes, or if the processes aren't fast enough, the queue will continue growing, and so too will your RAM, until you've exhausted your resources and start seeing worse performance due to overflowing to your harddrive. Even so, having too many Processes takes up memory and CPU threads of their own, so it's a bit of a balancing act.

With this optimization, I was able to run Halo: MCC at 1080p and 60 fps during gameplay, while recording at a rate of 28-30 fps, close enough to my target!

### Virtual Controller
By default, TensorKart was built with a gym dedicated to running with an emulator, and handles its own controller forwarding as a result. To find a solution to our needs, we need a library that's able to directly communicate with lower-level Windows drivers to create controller input.

Luckily, I didn't have to work very hard, as a library exists!
[Enter pyxinput](https://github.com/bayangan1991/PYXInput)
This gives us exactly what we need - a way to create a virtual controller and updates its state at any point in time immediately. Once I found the correct string mappings to controller inputs, it was just a matter of creating my own controller class:
f![image](https://user-images.githubusercontent.com/18727435/125725863-dbe0213e-b46e-4513-bc63-eeb80a6f474d.png)

### Summary of changes
And with that, we've done it! By making the above changes, we now have a tool that *should* be able to record frames and controller state from any Windows application (i.e. any game on PC). We've also hit our performance targets and are ready to get started with training!

# Training the Model
PyTorch has become very popular recently, and since I've already worked with Tensorflow and Keras previously, it felt like I should try the other major framework to get a feel for it, and I have to say, it's pretty easy to use.

Before we build our models, lets look back to my goal - I want to develop a *process of training models that's easily applicable to any game.* In order for this to hold true, we're left with a major limitation in that we can't use rewards. Traditionally, this type of learning is formulated as some form of reinforcement learning problem, but that's problematic for us because every game has a different reward function. In addition, if I want AIs to be able to play similar to a human, I'm not necessarily optimizing for that game's reward, but am instead trying to create an AI that closely mimics that specific human player (or skill level) as closely as possible. Think something along the lines of a Turing Test for gaming.

That being said, we'll be using RL terminology throughout, as that's where most of the research on Game AI has been for machine learning. To that end, lets look at what we DO have:
* The ability to run the model against an environment
* The ability to 

With that, lets go create our first model:
![image](https://user-images.githubusercontent.com/18727435/125726147-c172b8fa-c390-4411-8213-9a406f41566e.png)

I knew image data generally revolves around using CNNs, so I went ahead and started with that - the
