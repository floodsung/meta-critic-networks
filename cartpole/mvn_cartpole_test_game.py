#-------------------------------------
# Project: Meta Value Network
# Date: 2017.5.25
# All Rights Reserved
#-------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import os
from cartpole import CartPoleEnv
import json

def save_to_json(fname, data):
    with open(fname, 'w') as outfile:
        json.dump(data, outfile)

# CartPole Task Parameter
L_MIN = 0.5   # min length
L_MAX = 5    # max length

# Hyper Parameters
TASK_NUMS = 100
STATE_DIM = 4
ACTION_DIM = 2
TASK_CONFIG_DIM = 3
EPISODE = 10 # test nums per task
SAMPLE_NUMS = 30 #5,10,20
GAMES = 50


class ActorNetwork(nn.Module):

    def __init__(self,input_size,hidden_size,action_size):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,action_size)

    def forward(self,x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.log_softmax(self.fc3(out))
        return out

class MetaValueNetwork(nn.Module):

    def __init__(self,input_size,hidden_size,output_size):
        super(MetaValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,output_size)

    def forward(self,x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

class TaskConfigNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(TaskConfigNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Set initial states
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()
        # Forward propagate RNN
        out, _ = self.lstm(x, (h0, c0))
        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])
        return out

def roll_out(actor_network,task,sample_nums):
    states = []
    actions = []
    rewards = []
    is_done = False
    state = task.state
    result = 0
    for j in range(sample_nums):
        states.append(state)
        log_softmax_action = actor_network(Variable(torch.Tensor([state])).cuda())
        softmax_action = torch.exp(log_softmax_action)
        action = np.random.choice(ACTION_DIM,p=softmax_action.cpu().data.numpy()[0])
        one_hot_action = [int(k == action) for k in range(ACTION_DIM)]
        next_state,reward,done,_ = task.step(action)

        fix_reward = -10 if done else 1
        actions.append(one_hot_action)
        rewards.append(fix_reward)
        final_state = next_state
        state = next_state

        if done:
            task.episodes += 1
            is_done = True
            task.reset()
            #print("result:",result)
            break


    return torch.Tensor(states),torch.Tensor(actions),rewards,is_done,torch.Tensor(final_state)

def discount_reward(r, gamma,final_r):
    discounted_r = np.zeros_like(r)
    running_add = final_r
    for t in reversed(range(0, len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def main():
    # Define dimensions of the networks

    meta_value_input_dim =  STATE_DIM + TASK_CONFIG_DIM # 7
    task_config_input_dim = STATE_DIM + ACTION_DIM + 1 # 7

    # init meta value network with a task config network
    meta_value_network = MetaValueNetwork(input_size = meta_value_input_dim,hidden_size = 80,output_size = 1)
    task_config_network = TaskConfigNetwork(input_size = task_config_input_dim,hidden_size = 30,num_layers = 1,output_size = 3)
    meta_value_network.cuda()
    task_config_network.cuda()

    if os.path.exists("meta_value_network_cartpole.pkl"):
        meta_value_network.load_state_dict(torch.load("meta_value_network_cartpole.pkl"))
        print("load meta value network success")
    if os.path.exists("task_config_network_cartpole.pkl"):
        task_config_network.load_state_dict(torch.load("task_config_network_cartpole.pkl"))
        print("load task config network success")


    task_lengths = np.linspace(L_MIN,L_MAX,TASK_NUMS)

    datas = []

    for task_length in task_lengths:

        data_i = {}
        data_i["task_length"] = task_length
        data_i["results"] = []
        data_i_episode = {}
        for episode in range(EPISODE):
            task = CartPoleEnv(length = task_length)
            task.reset()

            data_i_episode["episode"] = episode

            # ----------------- Training ------------------

            # fetch pre data samples for task config network
            # [task_nums,sample_nums,x+y`]

            actor_network = ActorNetwork(STATE_DIM,40,ACTION_DIM)
            actor_network.cuda()
            actor_network_optim = torch.optim.Adam(actor_network.parameters(),lr = 0.01)
            '''
            if os.path.exists("actor_network.pkl"):
                actor_network.load_state_dict(torch.load("actor_network.pkl"))
                print("load actor_network success")
            '''
            # sample pre state,action,reward for task confi


            pre_states,pre_actions,pre_rewards,_,_ = roll_out(actor_network,task,SAMPLE_NUMS)


            test_results = []
            last_games = 0
            while task.episodes <= GAMES:
                # init task config [1, sample_nums,task_config] task_config size=3
                pre_data_samples = torch.cat((pre_states[-9:],pre_actions[-9:],torch.Tensor(pre_rewards)[-9:]),1).unsqueeze(0)
                task_config = task_config_network(Variable(pre_data_samples).cuda()) # [1,3]

                states,actions,rewards,is_done,final_state = roll_out(actor_network,task,SAMPLE_NUMS)
                final_r = 0
                if not is_done:
                    value_inputs = torch.cat((Variable(final_state.unsqueeze(0)).cuda(),task_config.detach()),1)
                    final_r = meta_value_network(value_inputs).cpu().data.numpy()[0]
                # train actor network
                actor_network_optim.zero_grad()
                states_var = Variable(states).cuda()

                actions_var = Variable(actions).cuda()
                task_configs = task_config.repeat(1,len(rewards)).view(-1,3)
                log_softmax_actions = actor_network(states_var)
                vs = meta_value_network(torch.cat((states_var,task_configs.detach()),1)).detach()
                # calculate qs
                qs = Variable(torch.Tensor(discount_reward(np.float32(rewards),0.99,np.float32(final_r)))).cuda()

                advantages = qs - vs
                actor_network_loss = - torch.mean(torch.sum(log_softmax_actions*actions_var,1)* advantages) #+ entropy #+ actor_criterion(actor_y_samples,target_y)
                actor_network_loss.backward()
                torch.nn.utils.clip_grad_norm(actor_network.parameters(),0.5)

                actor_network_optim.step()

                pre_states = states
                pre_actions = actions
                pre_rewards = rewards

                # testing
                if task.episodes > last_games:
                    # testing
                    result = 0
                    test_task = CartPoleEnv(length = task.length)
                    for test_epi in range(10):
                        state = test_task.reset()
                        for test_step in range(200):
                            softmax_action = torch.exp(actor_network(Variable(torch.Tensor([state])).cuda()))
                            #print(softmax_action.data)
                            action = np.argmax(softmax_action.cpu().data.numpy()[0])
                            next_state,reward,done,_ = test_task.step(action)
                            result += reward
                            state = next_state
                            if done:
                                break
                    aver_result = result/10.0
                    test_results.append(aver_result)
                    last_games = task.episodes

                    print("task length:",task_length,"episode:",episode,"games:",task.episodes,"result:",aver_result)

            data_i_episode["test_results"] = test_results
            data_i["results"].append(data_i_episode)
        datas.append(data_i)

    save_to_json('mvn_cartpole_test_100.json', datas)

if __name__ == '__main__':
    main()
