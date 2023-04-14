import gym
import torch
import numpy as np
from example import *
from dqn import DQN
from collections import deque
import copy
from train_test import train_DQN, test_DQN
from utils import *
from tqdm import tqdm

# ---- check device -----#
useGPU = 0
if torch.cuda.is_available(): 
 dev = "cuda:0"
 useGPU = 1
else: 
 dev = "cpu" 
 useGPU = 0
device = torch.device(dev) 

print(device)

all_envs = gym.envs.registry.all()
atari_envs = [env_spec for env_spec in all_envs if 'atari' in env_spec.entry_point]
print([env_spec.id for env_spec in atari_envs])

# ---- generate the environment ----- #
#env = gym.make("SpaceInvaders-v0")
env = gym.make("PongNoFrameskip-v4")
#get the dimension of the env 
height,width,channels = env.observation_space.shape
print(f'height: {height}, width: {width}, channels: {channels}')
#get the available actions
actions = env.action_space.n
print('n. of actions: \n', actions)
actions_name = env.unwrapped.get_action_meanings()
print('Available actions: \n', actions_name)
print('\n')
# ---- usage example ------ # 
usage_example = False
gym_example(usage_example, env, actions)

# ------ DQN ------- #
n_frames = 4 #dimension of the input layer
conv1 = 32 #conv1 dim
k_size1 = 8
stride1=4
conv2 = 64 #conv2 dim
k_size2 = 4
stride2 = 2
conv3 = 64 #conv3 dim
k_size3 = 3
stride3 = 1
fc1_size = 512 # fc1 dim
fc2_size = actions #output dim

nn = DQN(n_frames, conv1, conv2, conv3, k_size1, k_size2, k_size3, stride1, stride2, stride3, fc1_size, fc2_size ).to(device)

criterion = torch.nn.MSELoss() #loss function 
learning_rate = 0.0001
optimizer = torch.optim.Adam(nn.parameters(), lr=learning_rate)
# ---- training parameters ------ #
epsilon_start = 1.0
gamma = 0.99
losses = []
scores_array = []
q_values_array = []
queue = deque(maxlen=n_frames)
use_Boltz = True
render_interval = 5
# --- use pre-trained model ---- #
load_model = False
#trained_model = None
# ----- experince replay parameters ---- #
MAX_FRAME = 5000000
memory_size = 10000 #length of the experince list
replay_start_size = 10000 #the count of frames we wait for before starting training to populate the replay buffer
replay = deque(maxlen=memory_size) #replay buffer
batch_size = 32 #subset size (i.e. batch size dimension)
# ----- target network parameters ---- #
target_network = copy.deepcopy(nn) #Target network copy of the original network
target_network.load_state_dict(nn.state_dict()) #Copies the parameters of the original model
sync_freq = 1000 #Synchronizes the frequency parameter; every 50 steps we will copy the parameters of model into model2
# ----- plot results ----- #
plot_results = True

if load_model == False:
  trained_model, all_losses, all_scores, all_q_values = train_DQN(env=env, max_frame=MAX_FRAME, epsilon_start=epsilon_start, n_frames=n_frames, 
                                                                queue=queue, model=nn, gamma=gamma, criterion=criterion, losses=losses, 
                                                                scores_array=scores_array,target_network=target_network, 
                                                                replay=replay, batch_size=batch_size, sync_freq=sync_freq, 
                                                                q_values_array=q_values_array, replay_start_size=replay_start_size, 
                                                                use_Boltz=use_Boltz, useGPU=useGPU, optimizer=optimizer, render_interval=render_interval, device=device)
  if plot_results:
        #plot losses
    title = 'Plot MSE Loss'
    plot(all_losses,title)
    #plot scores
    title = 'Plot Amount of scores'
    plot(all_scores, title)
    #plot scores
    title = 'Plot AVG Q-values'
    plot(all_q_values, title) 

else:
  print('load the model...')
  trained_model = torch.load('./trained_model/model.pt', map_location=torch.device(device))

  # ----- TEST ----- # 
  print('\n')
  print(' ----- TEST PHASE ----- ')
  print('\n')

  max_games = 1
  t_queue = deque([np.zeros((84, 84), dtype=np.float32)] * n_frames, maxlen=n_frames)
  i = 0
  scores = []
  test_render_interval = 1

  for play_i in tqdm(range(0,max_games)):
    i+=1
    score = test_DQN(env=env, queue=t_queue, n_frames=n_frames, model=trained_model, useGPU=useGPU, device=device, play_i=play_i, render_interval=test_render_interval)
    scores.append(score)
    print(f' Game {i}, Score: {score}')

  print(f'Best score: {max(scores)}')