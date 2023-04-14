import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

#this function is used to capture easy temporal relationship by stacking n frames together
#for more complex temporal relationships 3D convolutions may be required
#following the DeepMind approach we stacked together the last 4 frames to produce a single image
#Each channel in the stacked image corresponds to a single frame at a different point in time, 
#so the CNN can learn to extract features that represent the changes that occur between frames.

def frame_preprocessing(frame):
  if frame.size == 210 * 160 * 3:
    img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
  elif frame.size == 250 * 160 * 3:
    img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
  else:
     assert False, "Unknown resolution."
  img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
  resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
  x_t = resized_screen[18:102, :]
  x_t = np.reshape(x_t, [84, 84, 1])
  return x_t.astype(np.uint8)

def initialize_queue(queue, n_frames, init_frame): 
  queue.clear()
  for i in range(n_frames):
    queue.append(frame_preprocessing(init_frame))
  
  return queue

#since we are using Pong without skipping frame integrated, we built a function to skip 4 frames at each step.
#in addition max-pooling is performed by stacking every k frames together and taking the element-wise maximum.
#then we return the resulting frame as the current observation.
#this function is used since we used the Pong env where no skip frames are considered.
def skip_frames(action,env,skip_frame=4):
  skipped_frame = deque(maxlen=2)
  skipped_frame.clear()
  total_reward = 0.0
  done = None
  for _ in range(skip_frame):
    n_state, reward, done, info = env.step(action)
    skipped_frame.append(n_state)
    total_reward += reward
    if done:
      break
  max_frame = np.max(np.stack(skipped_frame), axis=0)

  return max_frame, total_reward, done, info

def stack_frames(stacked_frames):
  #concatenate the frames 
  frames_stack = np.stack(stacked_frames, axis=0).squeeze()
  #normalize between 0 and 1
  frames_stack = frames_stack.astype(np.float32) / 255.0
  return frames_stack

def e_greedy(epsilon, q_val_, env):
  #by using the e-greey method we select an action (same strategy of n-armed bandit)
  if (random.random() < epsilon): 
  #exploration (i.e. it tries one of the movements at random)
    action = env.action_space.sample()
    #exploitation (i.e. we select the action with the highest q_value)
  else:
    action = np.argmax(q_val_)
  
  return np.asarray([action])

#boltzmann epxploration 
def boltzmann_exploration(q_values, temperature):
  #normalize action values
  q_values = q_values - np.max(q_values)
  # Convert action values to probabilities using Boltzmann distribution
  probs = np.exp(q_values / temperature)
  probs /= np.sum(probs)
  # Sample an action from the probabilities
  action = np.random.choice(len(probs), p=probs)
  
  return [action]

def get_epsilon(frame_idx, epsilon_decay_steps = 100000, min_epsilon=0.01, max_epsilon=1.0):
  #Decay epsilon value based on the current epoch and total number of epochs
  #epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-1.0 * frame_idx / epsilon_decay_steps)
  epsilon = max(min_epsilon, max_epsilon - frame_idx / epsilon_decay_steps)
  return epsilon if epsilon >= min_epsilon else min_epsilon


def get_temperature(frame_idx, temperature_decay_steps = 150000, min_temperature=0.05, max_temperature=2.0):
  #Decay epsilon value based on the current epoch and total number of epochs
  temperature = max(min_temperature, max_temperature - frame_idx / temperature_decay_steps)
  return temperature

def running_mean(x,N=50):
  c = x.shape[0] - N
  y = np.zeros(c)
  conv = np.ones(N)
  for i in range(c):
    y[i] = (x[i:i+N] @ conv)/N
  return y

def plot(array, title):
  fig, ax = plt.subplots(1,1)
  fig.set_size_inches(18.5, 10.5)
  ax.set_title(title)
  y_axis_name = title.split(' ')
  y_a_name = ''
  for i in range(1,len(y_axis_name)):
    y_a_name+=y_axis_name[i]+' '
  ax.set_xlabel("Epochs")
  ax.set_ylabel(y_a_name)
  fig.set_size_inches(10,6)
  plt.plot(running_mean(array,N=10))
  plt.show()