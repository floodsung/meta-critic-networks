import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import os

import json

def save_to_json(fname, data):
    with open(fname, 'w') as outfile:
        json.dump(data, outfile)

# Hyper Parameters
TASK_NUMS = 100
TEST_NUMS_PER_TASK = 10
ARM_NUMS = 2
STEP = 300
SAMPLE_NUMS = 10


class MultiArmBandit():
    """docstring for MultiArmBandit"""
    def __init__(self,arm_nums,probs):
        self.arm_nums = arm_nums
        self.probs = probs#np.random.dirichlet(np.ones(arm_nums),size=1)[0]

    def step(self,action): # one hot action
        prob = np.sum(self.probs * action)
        if random.random() < prob:
            return 1
        else:
            return 0



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
    actions = []
    rewards = []
    
    for sample in range(5):
            action = 0
            one_hot_action = [int(i == action) for i in range(ARM_NUMS)]
            reward = task.step(one_hot_action)
            actions.append(one_hot_action)
            rewards.append([reward])
    for sample in range(5):
            action = 1
            one_hot_action = [int(i == action) for i in range(ARM_NUMS)]
            reward = task.step(one_hot_action)
            actions.append(one_hot_action)
            rewards.append([reward])
            
    return torch.Tensor([actions]),torch.Tensor([rewards])

def roll_out_actions(actor_network,sample_nums):
    actions = []
    rewards = []
    softmax_action = torch.exp(actor_network(Variable(torch.Tensor([[1]])).cuda()))
    for step in range(sample_nums):
        action = np.random.choice(ARM_NUMS,p=softmax_action.cpu().data.numpy()[0])
        one_hot_action = [int(i == action) for i in range(ARM_NUMS)]
        actions.append(one_hot_action)

    return torch.Tensor([actions])

def main():

    mvn_input_dim = ARM_NUMS + 3
    task_config_input_dim = ARM_NUMS + 1
    # init meta value network with a task config network
    meta_value_network = MetaValueNetwork(input_size = mvn_input_dim,hidden_size = 80,output_size = 1)
    task_config_network = TaskConfigNetwork(input_size = task_config_input_dim,hidden_size = 30,num_layers = 1,output_size = 3)
    
    meta_value_network.cuda()
    task_config_network.cuda()

    if os.path.exists("meta_value_network_arm2.pkl"):
        meta_value_network.load_state_dict(torch.load("meta_value_network_arm2.pkl"))
        print("load meta value network success")
    if os.path.exists("task_config_network_arm2.pkl"):
        task_config_network.load_state_dict(torch.load("task_config_network_arm2.pkl"))
        print("load task config network success")


    # init a task generator for data fetching
    results = []

    total_rewards = 0

    task_probs = json.load(open("tasks_arm2.json"))

    for episode in range(TASK_NUMS):
        res_i = {}
        task_prob = task_probs[episode]["task_probs"]
        task = MultiArmBandit(ARM_NUMS,np.array(task_prob))
        res_i["arm_nums"] = ARM_NUMS
        res_i["task_probs"] = task.probs.tolist()
        res_i["sample_nums"] = SAMPLE_NUMS

        aver_rewards = []
        correct_probs = []
        for test_nums in range(TEST_NUMS_PER_TASK):
            actor_network = ActorNetwork(1,40,ARM_NUMS)
            actor_network_optim = torch.optim.Adam(actor_network.parameters(),lr=0.001)
            actor_network.cuda()
            pre_actions,pre_rewards = roll_out(actor_network,task,SAMPLE_NUMS)
            pre_data_samples = torch.cat((pre_actions,pre_rewards),2)

            task_configs = task_config_network(Variable(pre_data_samples).cuda()).repeat(1,SAMPLE_NUMS).view(-1,3)

            for step in range(STEP):

                inputs = Variable(torch.Tensor([[1]])).cuda() #[1,1]
                actions = roll_out_actions(actor_network,SAMPLE_NUMS)
                actions_var = Variable(actions.view(-1,ARM_NUMS)).cuda()
                actor_data_samples = torch.cat((actions_var,task_configs.detach()),1) #[task_nums,5]
                log_softmax_actions = actor_network(inputs) # [1,2]
                log_softmax_actions = log_softmax_actions.repeat(1,SAMPLE_NUMS).view(-1,ARM_NUMS)
                # train actor network

                actor_network_optim.zero_grad()
                qs = meta_value_network(actor_data_samples)
                actor_network_loss = - torch.mean(torch.sum(log_softmax_actions*actions_var,1)* qs) #+ actor_criterion(actor_y_samples,target_y)
                actor_network_loss.backward()

                actor_network_optim.step()

            choice = torch.exp(actor_network(inputs)).cpu().data[0].numpy()
            aver_reward = np.sum(choice * task.probs)
            optimal_action = np.argmax(task.probs)
            optimal_choice = [int(i == optimal_action) for i in range(ARM_NUMS)]
            correct_prob = np.sum(choice*optimal_choice)

            aver_rewards.append(float(aver_reward))
            correct_probs.append(float(correct_prob))
            total_rewards += aver_reward


        res_i["aver_rewards"] = aver_rewards
        res_i["correct_probs"] = correct_probs

        results.append(res_i)

        print("aver_reward",np.mean(aver_rewards),"correct prob:",np.mean(correct_probs),"task:",task.probs)

    save_to_json('mvn_arm_2_sample_10_last.json', results)
    print("total aver reward:",total_rewards/TASK_NUMS/TEST_NUMS_PER_TASK)




if __name__ == '__main__':
    main()
