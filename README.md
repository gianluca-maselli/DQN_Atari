# DQN_Atari


The directory present the Deep Q-network architecture inspiered by the well-know one developed by [Mnih et al](https://www.nature.com/articles/nature14236/?source=post_page---------------------------). which has achieved expert gamers results in several Atari 2600 games. 

Since training the network on these games could be tremendously time consuming, we decide to use the 'PongNoFrameskip-v4' environment provided by the gym library.  This choice has been taken also to assure that the environement do not present the skipping frame modality present in some gym envs. Therefore, to assure that the model receives the correct input, we implemented the **skip_frames()** function, which samples one out of 4 frames at each step. As previously said, this function is useful in those envs which frame skipping is not present. By contrast in envs as 'Pong-v5' the frame skipping is already implemented in the env itself. 

Another important aspect is to correctly preprocessing the input to fed in the DQN. The solution is to define the functions **process()** and **stack_frames()** which convert each original to a grayscale image and then resize it in order to obtain a frame of dimension $(84,84,1)$. Each preprocessed frame is then added to a queue wich has max length of 4. Therefore, at each step the new frame replace the oldest one in the queue. At this point the frames contained in the queue are stacked together to obtain a single frame of dimension $(84,84,4)$ which is the input to the DQN. This strategy is used in order to capture some of the temporal information in the input sequence. 

After training the network, the result obtained are overall satisfactory as shown in the following plot, where moving average as been applied to depict the trend of the score with respect to the number of games. 

