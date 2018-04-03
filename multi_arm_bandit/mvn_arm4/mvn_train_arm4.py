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


# Hyper Parameters
TASK_NUMS = 10
ARM_NUMS = 4
EPISODE = 1000
STEP = 300
SAMPLE_NUMS = 10 #5,10,20
TEST_SAMPLE_NUMS = 5



class MultiArmBandit():
    """docstring for MultiArmBandit"""
    def __init__(self,arm_nums):
        self.arm_nums = arm_nums
        self.probs = np.random.dirichlet(np.ones(arm_nums),size=1)[0]

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

def roll_out(actor_network_list,task_list,sample_nums):
    actions = []
    rewards = []

    for i in range(len(task_list)):
        task_actions = []
        task_rewards = []
        softmax_action = torch.exp(actor_network_list[i](Variable(torch.Tensor([[1]])).cuda()))
        for j in range(sample_nums):
            action = np.random.choice(ARM_NUMS,p=softmax_action.cpu().data.numpy()[0])
            one_hot_action = [int(k == action) for k in range(ARM_NUMS)]
            reward = task_list[i].step(one_hot_action)
            task_actions.append(one_hot_action)
            task_rewards.append([reward])
        actions.append(task_actions)
        rewards.append(task_rewards)

    return torch.Tensor(actions),torch.Tensor(rewards)

def main():

    mvn_input_dim = ARM_NUMS + 3
    task_config_input_dim = ARM_NUMS + 1
    # init meta value network with a task config network
    meta_value_network = MetaValueNetwork(input_size = mvn_input_dim,hidden_size = 80,output_size = 1)
    task_config_network = TaskConfigNetwork(input_size = task_config_input_dim,hidden_size = 30,num_layers = 1,output_size = 3)
    meta_value_network.cuda()
    task_config_network.cuda()

    if os.path.exists("meta_value_network_arm4.pkl"):
        meta_value_network.load_state_dict(torch.load("meta_value_network_arm4.pkl"))
        print("load meta value network success")
    if os.path.exists("task_config_network_arm4.pkl"):
        task_config_network.load_state_dict(torch.load("task_config_network_arm4.pkl"))
        print("load task config network success")

    meta_value_network_optim = torch.optim.Adam(meta_value_network.parameters(),lr=0.001)
    task_config_network_optim = torch.optim.Adam(task_config_network.parameters(),lr=0.001)

    # init a task generator for data fetching
    task_list = []
    for i in range(TASK_NUMS):
        task = MultiArmBandit(ARM_NUMS)
        task_list.append(task)

    for episode in range(EPISODE):
        # ----------------- Training ------------------

        if (episode+1) % 10 ==0:
            # renew the tasks
            task_list = []
            for i in range(TASK_NUMS):
                task = MultiArmBandit(ARM_NUMS)
                task_list.append(task)


        # fetch pre data samples for task config network
        # [task_nums,sample_nums,x+y`]
        
        actor_network_list = [ActorNetwork(1,40,ARM_NUMS) for i in range(TASK_NUMS)]
        [actor_network.cuda() for actor_network in actor_network_list]
        actor_network_optim_list = [torch.optim.Adam(actor_network.parameters(),lr = 0.001) for actor_network in actor_network_list]

        pre_actions,pre_rewards = roll_out(actor_network_list,task_list,SAMPLE_NUMS)


        for step in range(STEP):
            # init task config [task_nums*sample_nums,task_config] task_config size=3
            pre_data_samples = torch.cat((pre_actions,pre_rewards),2)
            task_configs = task_config_network(Variable(pre_data_samples).cuda()).repeat(1,SAMPLE_NUMS).view(-1,3)  #[task_nums,3]
            # fetch data samples from task generator
             # [task_nums,sample_nums,x+y`]
            
            # process data samples with actor network list and build the graph
            # 1. for actor network training
            # actor_y_samples: [task_nums*sample_nums,y]
            # actor_x_samples: [task_nums*sample_nums,x]
            # actor_data_samples: [task_nums*sample_nums,x+y+task_config]
            

            inputs = Variable(torch.Tensor([[1]])).cuda() #[1,1]
            
            actions,rewards = roll_out(actor_network_list,task_list,SAMPLE_NUMS)
            actions_var = Variable(actions.view(-1,ARM_NUMS).cuda())
            actor_data_samples = torch.cat((actions_var,task_configs.detach()),1) #[task_nums,5]
            
            log_softmax_actions = actor_network_list[0](inputs) # [1,2]

            for i in range(TASK_NUMS - 1):
                log_softmax_action = actor_network_list[i+1](inputs)
                log_softmax_actions = torch.cat((log_softmax_actions,log_softmax_action),0)

            log_softmax_actions = log_softmax_actions.repeat(1,SAMPLE_NUMS).view(-1,ARM_NUMS)
            # train actor network
            for i in range(TASK_NUMS):
                actor_network_optim_list[i].zero_grad()
            qs = meta_value_network(actor_data_samples).detach()
            actor_network_loss = - torch.mean(torch.sum(log_softmax_actions*actions_var,1)* qs) #+ actor_criterion(actor_y_samples,target_y)
            actor_network_loss.backward()
            for i in range(TASK_NUMS):
                actor_network_optim_list[i].step()
            
            # 2. train meta value network
            # value_data_samples: Variable([task_nums*sample_nums,x+y+task_config])
            # target_values: Variable([task_nums*sample_nums,-(y-y`)^2])
            
            '''
            actions_p = actions.view(-1,2)
            rewards_p = rewards.view(-1,1)
            rewards_p = torch.cat((rewards_p,rewards_p),1)
            results = actions_p * rewards_p
            aver_rewards = torch.mean(results.view(TASK_NUMS,SAMPLE_NUMS,2),1).squeeze(1)
            '''
            #aver_rewards = torch.sum(rewards,1).squeeze(1)

            meta_value_network_optim.zero_grad()
            task_config_network_optim.zero_grad()
            value_data_samples = torch.cat((actions_var,task_configs),1)
            target_values = Variable(rewards).view(-1,1).cuda()
            values = meta_value_network(value_data_samples)
            criterion = nn.MSELoss()
            meta_value_network_loss = criterion(values,target_values)
            meta_value_network_loss.backward()
            meta_value_network_optim.step()
            task_config_network_optim.step()

            task_params = [task.probs for task in task_list]
            if (step+1) % 20 == 0:
                print("episode:",episode,"step:",step+1,"value:",torch.mean(values).data[0],"value loss:",meta_value_network_loss.data[0])
        	
            pre_actions = actions
            pre_rewards = rewards

        
        if (episode+1) % 10 == 0 :
            # Save meta value network
            torch.save(meta_value_network.state_dict(),"meta_value_network_arm4.pkl")
            torch.save(task_config_network.state_dict(),"task_config_network_arm4.pkl")
            print("save networks for episode:",episode)


if __name__ == '__main__':
    main()
