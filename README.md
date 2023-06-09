# DQN_Atari


The directory contains the Deep Q-network architecture inspiered by the well-know one developed by [Mnih et al](https://www.nature.com/articles/nature14236/?source=post_page---------------------------). which has achieved expert gamers results in several Atari 2600 games. 

Since training the network on these games could be tremendously time consuming, we decide to use the 'PongNoFrameskip-v4' environment provided by the gym library.  This choice has been taken also to assure that the environement do not present the skipping frame modality present in some gym envs. Therefore, to assure that the model receives the correct input, we implemented the **skip_frames()** function, which samples one out of 4 frames at each step. As previously said, this function is useful in those envs which frame skipping is not present.

Another important aspect is to correctly preprocessing the input to fed in the DQN. The solution is to define the functions **process()** and **stack_frames()** which convert each original to a grayscale image and then resize it in order to obtain a frame of dimension $(84,84,1)$. Each preprocessed frame is then added to a queue wich has max length of 4. Therefore, at each step the new frame replace the oldest one in the queue. At this point the frames contained in the queue are stacked together to obtain a single frame of dimension $(84,84,4)$ which is the input to the DQN. This strategy is used in order to capture some of the temporal information in the input sequence. 

The result obtained are overall satisfactory, during the training phase, as shown in the following plot, where moving average as been applied to depict the trend of the score with respect to the number of games. 

<a href="url"><img src="https://github.com/gianluca-maselli/DQN_Atari/blob/main/plot/scores_pong.png" height="370" width="600" ></a>

In addition we provide the trained model on the 'PongNoFrameskip-v4' environment which can be used with the test function to play as many games as we want. During test phase, it is also possibile to render the games to see what is going on. 
