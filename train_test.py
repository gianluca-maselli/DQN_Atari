import torch
import random
import numpy as np
from utils import *
from tqdm import tqdm

# ----- TRAIN ------ #
def train_DQN(env,max_frame, queue, model, epsilon_start, n_frames, gamma, criterion, losses, scores_array, target_network, replay, batch_size, sync_freq, q_values_array, replay_start_size, use_Boltz, device, useGPU, optimizer, render_interval):
    
    model.train()
    target_network.eval()

    frame_waiting = 0 #frames to wait before using experience replay buffer
    frame_idx = 0

    
    game = 1 #count number of games
    mean_reward = 19.0 #mean reward before stopping the training loop
    
    while frame_idx < max_frame:
        #for each epoch we initialized a new game. Here state is a frame. 
        init_state = env.reset()
        #initialize the queue by clear it and adding the same frame 4 times at the beginning of the episode
        # Get the last four frames as the input to the agent (i.e. our CNN)
        frame_queue = initialize_queue(queue, n_frames, init_state)
        #stack the frames together
        input_frames = stack_frames(frame_queue)
        current_state = np.expand_dims(input_frames,0)
        #check the status of the game
        done = False
        tot_score = 0
        tot_loss = 0
        tot_q_values = 0
        total_steps = 0 
        mini_batchGD = 0 #check if we use the mini-batch GD
        count_loss = 0 #count number of loss computations
        
        # ------ GAME START ----- #
        while (done == False) and total_steps < 100000:
            
            frame_idx +=1
            total_steps += 1
            frame_waiting +=1
            
            if use_Boltz == False:
                epsilon = get_epsilon(frame_idx=frame_idx, epsilon_decay_steps = 150000, min_epsilon=0.01, max_epsilon=epsilon_start)
            else:
                temperature = get_temperature(frame_idx, temperature_decay_steps = 150000, min_temperature=0.01, max_temperature=2.0)

            #compute the Q-values for this state given by the CNN
            #the network is updated only when a batch of experiences is sampled from the replay buffer. Not in the predictions of the next state
            with torch.no_grad():
                q_val = model(torch.from_numpy(current_state).to(device)).flatten()
            tot_q_values += torch.max(q_val).item()

            if useGPU==1: q_val_ = q_val.cpu().data.numpy()
            else: q_val_ = q_val.data.numpy()
            
            #get the action with e_greedy approach or boltzmann exploration
            if use_Boltz:
                action = boltzmann_exploration(q_val_, temperature)
            else:
                action = e_greedy(epsilon, q_val_, env)
            #perform action base on its index
            #skip frames
            n_state, reward, done, info = skip_frames(action[0],env=env,skip_frame=4)
            reward = np.asarray([reward])
            #increment the score according to the reward obtained
            tot_score+=reward[0]
            #preprocess new state frame
            new_frame = frame_preprocessing(n_state)
            #add the new frame to the queue
            frame_queue.append(new_frame)
            #create single frame image from the stack using the new frame
            new_input_frames = stack_frames(frame_queue)
            #dimension (1,4,84,84)
            next_state = np.expand_dims(new_input_frames,0)
            #print(new_input_frames_.shape)

            # ------- EXPERIENCE REPLAY --------- #
            #build the tuple (s,a,s_t+1,r_t+1)
            exp = (torch.Tensor(current_state), action, torch.Tensor(next_state), reward, done)
            #add the current experience to the experience replay list
            replay.append(exp)
            #the state S_t+1 becomes our current state S
            current_state = next_state
            #if the replay list is at least as long as the mini-batch size, begins the mini-batch training
            if len(replay) > batch_size and frame_waiting >= replay_start_size:
                mini_batchGD = 1
                #we random select a subset from the replay list of dimension batch size
                minibatch = random.sample(replay, batch_size)
                #separates out the components of each experience into separate mini-batch tensors
                #tensor of current state batch
                state_batch = torch.cat([current_state for (current_state, action, next_state, reward, done) in minibatch])
                #tensor of action batch
                action_batch = torch.Tensor(np.array([action for (current_state, action, next_state, reward, done) in minibatch]))
                #print('action_batch: ', action_batch.shape)
                #tensor of next state batch
                state2_batch = torch.cat([next_state for (current_state, action, next_state, reward, done) in minibatch])
                #tensor of reward batch
                reward_batch = torch.Tensor(np.array([reward for (current_state, action, next_state, reward, done) in minibatch]))
                #tensor of done batch
                done_batch = torch.Tensor([done for (current_state, action, next_state, reward, done) in minibatch])
                done_batch = done_batch.unsqueeze(dim=1)
                #Recomputes Q values with Q-netwrok for the mini- batch of states to get gradients
                q_val1 = model(state_batch.to(device))
                #computes Q values with the Target Network for the mini-batch of next states, but doesn’t compute gradients
                with torch.no_grad():
                    q_val2 = target_network(state2_batch.to(device))

                #target value, i.e. R_t+1 + gamma
                #((1 - done_batch) means 1-True/False, if 1-False = 1-0 = 1, or  1-True = 1-1 = 0
                #we use this strategy to set the right-hand part of the equation to 0 if the game is done
                #If the game is done then we are in a terminal state and there is no next state to take the maximum Q value on, so the target just becomes the reward R_t+1.
                Y = reward_batch.to(device) + gamma * ((1 - done_batch.to(device)) * torch.max(q_val2,dim=1)[0].unsqueeze(dim=1))
                #we use the tensor.gather method to subset the q_val1 tensor  by the action indices
                #in this way we only select the Q-values associated with actions that were actually chosen, resulting in a 100-len vector. 
                #example action=[1,0,3,4, 2], q_val = [0.2, 0.3, 0.1, 0.7, 0.4], q_val.gather(dim=1, index=action) = [0.3, 0.2, 0.7, 0.4, 0.1].
                X = q_val1.gather(dim=1,index=action_batch.long().to(device))

                # ------- END REPLAY --------- #

                #loss computation
                loss = criterion(X.to(device), Y.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tot_loss +=loss.item()
                count_loss +=1
                # ------ TARGET NET SYN ------- #
                if frame_idx % sync_freq == 0:
                    target_network.load_state_dict(model.state_dict())
                    
            #if game % render_interval == 0:
            #    print('\n Rendering... \n')
            #    env.render()
            #env.close()
        #start saving loss values as soon we use experience replau buffer
        if mini_batchGD == 1:
            if use_Boltz: print(f'\n Frame Processed: {frame_idx} Game {game} tot_score: {tot_score} with temperature: {temperature} with MSE Loss {tot_loss/count_loss}')
            else: print(f'\n Frame Processed: {frame_idx} Game {game} tot_score: {tot_score} with epsilon: {epsilon} with MSE Loss {tot_loss/count_loss}')
            losses.append(tot_loss/count_loss)
            scores_array.append(tot_score)
            q_values_array.append(tot_q_values/count_loss)
        
        if len(scores_array) >= 100 and game % 10 == 0:
            avg = np.mean(np.array(scores_array[-100:]))
            print(' \n AVG last 100 games: ', avg)
            #if we reached the desired avg on the last 100 games, break and stop training
            if avg >= mean_reward:
                torch.save(model, './trained_model/model.pkl')
                print("Solved in %d frames!" % frame_idx)
                break
        
        game+=1
    return model, np.array(losses), np.array(scores_array), np.array(q_values_array)



# ----- TEST ------ #

def test_DQN(env, queue, n_frames, model, useGPU, device, play_i, render_interval):
    
    model.eval()
    init_state = env.reset()
    frame_queue = initialize_queue(queue, n_frames, init_state)
    #stack the frames together
    input_frames = stack_frames(frame_queue)
    current_state = np.expand_dims(input_frames,0)
    tot_score = 0
    done = False
    #continue till the game is in progress
    while(done==False):
        #compute the Q-values for this state given by the CNN
        with torch.no_grad():
            q_val = model(torch.from_numpy(current_state).to(device))
        #take the action with the highest q_val
        if useGPU==1: q_val_ = q_val.cpu().data.numpy()
        else: q_val_ = q_val.data.numpy()
        #chose the action with the highest q-val
        action = np.argmax(q_val_)
        next_state, reward, done, info = skip_frames(action,env=env,skip_frame=4)
        #increment the score according to the reward obtained
        tot_score+=reward
        new_frame = frame_preprocessing(next_state)
        frame_queue.append(new_frame)
        new_input_frames = stack_frames(frame_queue)
        next_state = np.expand_dims(new_input_frames,0)
        current_state = next_state
        
        if play_i % render_interval == 0:
            env.render()
        env.close()

    return tot_score
