# pymcc
Python tool/framework to develop models using Halo: MCC on PC

![image](https://github.com/jgreer013/pymcc/blob/9eec4ace6d0a46e8cde4c8bd3bc1357b63399884/bot_1.gif)

# Goals
Game AI is a fascinating and exciting area to me, but often disappointing. Most Game AI is built off of complex decision trees or Markov State Transitions, but both of those are cumbersome to create and fail to create AI which play similarly to people. 

Deep Reinforcement Learning seems like promising directions to go in, but so far, most environments for training these types of applications are restricted to toy programs. My goal is to build a tool which others can use to abstract the hard/boring parts - pulling the frame and controller data from any Windows PC Gaming application running, saving it, then being able to apply virtual input in real-time later.

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
* GTA V's gets very poor performance (12-13 fps at 800x600 resolution). While this can vary with hardware, that's way too low to play with - one could probably record data at that frequency, but it's very difficult to play complex games at such a framerate. 
* Ideally we want to be able to play the game as we normally do (30-60 fps) while still getting a good amount of data (15-30 fps in recorded data, meaning we'll record 25-50% of our frames)
* The other libraries are all game-specific, with a lot of additional information being provided in the gym environments that a human wouldn't necessarily have access to.

## Getting a Baseline from someone else
The first approach is a step in the right direction though, so lets see what other libraries we can find...

![image](https://camo.githubusercontent.com/acddf9f023386781ab74d1eb9a83f232a01be385dfbfb2ef875221fa336bae52/68747470733a2f2f6d656469612e67697068792e636f6d2f6d656469612f313433355676436f7356657a51592f67697068792e676966)

* [Tensorkart](https://github.com/kevinhughes27/TensorKart)

Ah! Here's a great baseline for what we need - the recording, training, and playing are all nicely laid out, and since it captures the window on the desktop, it's already able to pull frame data from any Windows application, just as we want, and pulls the controller data as we need it to; however, this one requires that we add additional software and libraries for playing on the emulator, and it's hard to judge performance when its only running an emulated game. Still, it's a good baseline, and our tool borrows heavily from it.

## Updating it to suit our needs
Once I found the baseline, it's time to alter it a bit. There's a few things we need to do:
* Pulling full controller data from the active controller
* Performance Improvements
* Pushing virtual controller data to the active window (i.e. the game application)

Pulling the full controller data is fairly straightforward, as we just need to update the XboxController class to return the full vector on read().

Performance improvements are a bit trickier though. The libraries that TensorKart is using are already fairly low-level and performant on Windows, but we still have a very low default sample rate, and our machine can't seem to record any faster by increasing it.

After looking at the task manager, our RAM, CPU, GPU, and Disk seemed to be at low levels, even while running Halo and recording data. Well, if the machine's not being overwhelmed, why doesn't increasing the sample rate do anything? If we think about the traditional bottlenecks in software and hardware, IO tends to be a fairly big bottleneck, as reading and writing to disk take the most time, even on an M.2 SSD.

Hope is not lost though - since the CPU and RAM have plenty of room to be used still, we can actually just parallelize the writing of the frame and controller, which is our biggest IO bottleneck rather than the game itself. How do we do that? [Multiprocessing!](https://docs.python.org/3/library/multiprocessing.html)

### Multiprocessing
The fix is fairly straightforward, all we need to do is put our data in an in-memory queue instead of writing it immediately. Then, we need processes which are registered to pull from the queue, which will then dequeue our data and write it to disk.

![image](https://user-images.githubusercontent.com/18727435/125725169-174bdc65-fb01-4c63-a11e-2b6edf3d5395.png)

![image](https://user-images.githubusercontent.com/18727435/125725233-f66ceed2-0cc6-4c07-99eb-6f149c1bb5c5.png)

![image](https://user-images.githubusercontent.com/18727435/125725193-4cd54ff4-aca2-4fcd-bc7a-064c600d0382.png)

![image](https://user-images.githubusercontent.com/18727435/125725207-f71b0bf1-ae81-4595-a2b7-ccd5ce997fae.png)

This allows the recording process to continue with minimal lag, and our sample rate can scale based on the number of processes that are pulling data from the queue and writing it to disk as saved screenshots. 

Depending on hardware, there would need to be a balance between the number of processes - if there aren't enough processes, or if the processes aren't fast enough, the queue will continue growing, and so too will your RAM, until you've exhausted your resources and start seeing worse performance due to overflowing to your harddrive. Even so, having too many Processes takes up memory and CPU threads of their own, so it's a bit of a balancing act.

With this optimization, I was able to run Halo: MCC at 1080p and 60 fps during gameplay, while recording at a rate of 28-30 fps, close enough to my target!

*Note: My hardware is an i9-9900k, NVidia 3090 FE, 32 GB of RAM, and a 1TB M.2 NVME SSD. A relativley high-end machine in terms of consumer hardware, so lower-spec PCs, or more taxing games, may not be able to achieve the same performance.*

### Virtual Controller
By default, TensorKart was built with a gym dedicated to running with an emulator, and handles its own controller forwarding as a result. To find a solution to our needs, we need a library that's able to directly communicate with lower-level Windows drivers to create controller input.

Luckily, I didn't have to work very hard, as a library exists!

[Enter pyxinput](https://github.com/bayangan1991/PYXInput)

This gives us exactly what we need - a way to create a virtual controller and updates its state at any point in time immediately. Once I found the correct string mappings to controller inputs, it was just a matter of creating my own controller class:

![image](https://user-images.githubusercontent.com/18727435/125725863-dbe0213e-b46e-4513-bc63-eeb80a6f474d.png)

And with that, we've done it! By making the above changes, we now have a tool that *should* be able to record frames and controller state from any Windows application (i.e. any game on PC). We've also hit our performance targets and are ready to get started with training!

# Training the Model
PyTorch has become very popular recently, and since I've already worked with Tensorflow and Keras previously, it felt like I should try the other major framework to get a feel for it, and I have to say, it's pretty easy to use.

Before we build our models, lets look back to my goal - I want to develop a *process of training models that's easily applicable to any game.* In order for this to hold true, we're left with a major limitation in that we can't use rewards. 

Traditionally, this type of learning is formulated as some form of reinforcement learning problem, but that's problematic for us because every game has a different reward function. In addition, if I want AIs to be able to play similar to a human, I'm not necessarily optimizing for that game's reward, but am instead trying to create an AI that mimics that specific human player (or skill level) as closely as possible. Think something along the lines of a Turing Test for gaming.

That being said, we'll be using RL terminology throughout, as that's where most of the research on Game AI has been for machine learning. To that end, lets look at what we DO have:
* The ability to run the model against an environment
* Recorded gameplay of "Expert Demonstrations" from myself

Okay, so essentially, what we're trying to do is learn an *agent policy* that most closely matches the *expert policy* based on the *demonstrations alone*. As it turns out, this can be accomplished using *Behavior Cloning*. Essentially, the goal is to train a model using supervised learning, where given a state (the frame at time X, and optionally the controller state at time X), the model must learn the policy, which is defined as its output action (the controller state at time X+1). 

We'll go with this type of formulation as its pretty much our main choice given that we *don't have any rewards to go on.* While I could probably create a reward function for Halo, I'd rather avoid reward tuning for now.

Unfortunately, this formulation is pretty mediocre in performance, as behavior cloning doesn't allow any long-term planning/learning to take place, and minor errors that cause the agent to deviate from the user's path cause cascading/catastrophic failures. There's also been very little advancement in this area - most advancements involve trying to minimize expert demonstrations, whereas we're doing the opposite. While the outlook doesn't look great, lets still see what we can come up with.

With that, lets go create our first model:

![image](https://user-images.githubusercontent.com/18727435/125726147-c172b8fa-c390-4411-8213-9a406f41566e.png)

I knew image data generally revolves around using CNNs, so I went ahead and started with that - the input is a 1920x1080x3 color image of a screenshot of Halo Reach Firefight gameplay. Firefight was chosen because it's a limited mode that's very consistent across playthroughs, unlike multiplayer. In addition, it gives us a low-stress environment to experiment with.

The model is built off of PyTorch's tutorial for CNNs, but it's not clear to me how big the kernel size or step size my convolutions should be. I went with a steadily growing approach, but I struggled to find any consistent literature or good insights here. Unfortunately its a bit less intuitive than NLP to me. That being said, what I do know is that BatchNorm is pretty standard across CNNs because of the issues they tend to get with gradients without it; however, what's not clear is whether to put the batch norm before or after the activation function. [As it turns out, this isn't clear in the community either.](https://forums.fast.ai/t/why-perform-batch-norm-before-relu-and-not-after/81293) I'd suggest you try both approaches for your model before experimenting too much (I got better performance later on by switching their order, which meant I wasted my time with a variety of architectures before I should have).

So, with an architecture like this, what does it learn?

[Link to video](https://twitter.com/i/status/1377847887720161282)

![image](https://github.com/jgreer013/pymcc/blob/9eec4ace6d0a46e8cde4c8bd3bc1357b63399884/bot_1.gif)

Well, it did stuff! It played like a human, if that human were very new to gaming and very *bad* at Halo. While not particularly groundbreaking, it's certainly entertaining to watch! I'll summarize some of the challenges and on-going issues in the next section.

# Looking Deeper
After looking at how my data is distributed, I'm fairly certain my difficulties are in part due to the fact that I have a huge imbalance problem.

Below are the distributions of the thumbstick X and Y values:

![image](https://user-images.githubusercontent.com/18727435/125729124-66442ec8-5ec1-4ae0-a48b-e106e07a06de.png)

Here are distributions for the triggers:

![image](https://user-images.githubusercontent.com/18727435/125729153-3175e226-b0ee-45c6-8940-b666d3aba829.png)

And, here are the major buttons:

![image](https://user-images.githubusercontent.com/18727435/125729186-d724ee21-b8e5-45ec-9afc-663b98d60658.png)

...yeah that's very imbalanced.

So far, I've tried looking at [Focal Loss from Facebook](https://arxiv.org/abs/1708.02002), as that seemed to be a drop-in replacement meant to counteract class imbalance, but alas, not much difference in results. I still need to try oversampling or assigning more weight to the positive weights. Alternatively, I may decide to scope the problem down to just the sticks and the right trigger, as these constitute the primary actions needed (walking, looking, and shooting). Overall though, I'm happy with how far I've come.

# Going Forward
Once I have some time, I plan on cleaning up this repository to make it more readable, as it stands the code is a bit of a jumble mess of different experiments. I've tried different formulations:
* Classification (discretize the continuous parts output space, namely the thumbsticks and triggers)
* Regression
* GANs (admittedly an odd approach, but I mainly just wanted to try my hand with training GANs and becoming more familiar with them)
* Using pre-trained ResNets
* Adding the current frame's action to the input state

Some areas I hope to experiment with in the future:
* Using multiple frames of data
* Using an Inverse Reinforcement Learning formulation (somehow without rewards)
* Gathering more data
* Using sampling methods that help correct the data imbalance problem

I'll try to find some time soon to clean up this repository and make the code more readable/usable for others. I may also release my data at some point, although that might have to be on the Kaggle platform, and with more examples than what I currently have. Until then, thanks for reading!
