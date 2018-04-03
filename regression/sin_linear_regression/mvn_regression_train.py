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

# Task Parameters
# y = a * sin(x + b)
A_MIN = 1.0
A_MAX = 5.0
B_MIN = 0
B_MAX = math.pi
X_MIN = -5.0
X_MAX = 5.0

# y = ax+b
LINEAR_A_MIN = -3.0
LINEAR_A_MAX = 3.0
LINEAR_B_MIN = -3.0
LINEAR_B_MAX = 3.0

# Hyper Parameters
TASK_NUMS = 100
EPISODE = 2000
STEP = 300
SAMPLE_NUMS = 10
TEST_SAMPLE_NUMS = 5



class TaskGenerator():

    def __init__(self,task_nums):
        # random sample task params a and b for data generation
        self.task_nums = task_nums
        self.task_params = []
        for i in range(task_nums):
            a = random.uniform(A_MIN,A_MAX)
            b = random.uniform(B_MIN,B_MAX)
            linear_a = random.uniform(LINEAR_A_MIN,LINEAR_A_MAX)
            linear_b = random.uniform(LINEAR_B_MIN,LINEAR_B_MAX)
            task_type = i%2
            self.task_params.append([a,b,linear_a,linear_b,task_type])

    def fetch_datas(self,sample_nums):
        x_samples = []
        y_samples = []
        for i in range(self.task_nums):
            x_sample = []
            y_sample = []
            for j in range(sample_nums):
                x = random.uniform(X_MIN,X_MAX)
                x_sample.append([x])
                if self.task_params[i][4] == 0:
                    a = self.task_params[i][0]
                    b = self.task_params[i][1]
                    y = a * math.sin(x + b)    # The Task definition
                else:
                    linear_a = self.task_params[i][2]
                    linear_b = self.task_params[i][3]
                    y = linear_a * x + linear_b
                y_sample.append([y])
            x_samples.append(x_sample)
            y_samples.append(y_sample)
        return torch.Tensor(x_samples),torch.Tensor(y_samples)

class ActorNetwork(nn.Module):

    def __init__(self,input_size,hidden_size,output_size):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,output_size)

    def forward(self,x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
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

def main():

    # init meta value network with a task config network
    meta_value_network = MetaValueNetwork(input_size = 5,hidden_size = 80,output_size = 1)
    task_config_network = TaskConfigNetwork(input_size = 2,hidden_size = 30,num_layers = 1,output_size = 3)
    meta_value_network.cuda()
    task_config_network.cuda()
    if os.path.exists("meta_value_network.pkl"):
        meta_value_network.load_state_dict(torch.load("meta_value_network.pkl"))
        print("load meta value network success")
    if os.path.exists("task_config_network.pkl"):
        task_config_network.load_state_dict(torch.load("task_config_network.pkl"))
        print("load task config network success")

    meta_value_network_optim = torch.optim.Adam(meta_value_network.parameters(),lr=0.001)
    task_config_network_optim = torch.optim.Adam(task_config_network.parameters(),lr=0.001)

    # init a task generator for data fetching
    task_generator = TaskGenerator(TASK_NUMS)
    for episode in range(EPISODE):
        # ----------------- Training ------------------

        if (episode+1) % 10 ==0 :
            task_generator = TaskGenerator(TASK_NUMS)

        # fetch pre data samples for task config network
        # [task_nums,sample_nums,x+y`]
        pre_x_samples,pre_y_samples = task_generator.fetch_datas(SAMPLE_NUMS)
        actor_network_list = [ActorNetwork(1,40,1) for i in range(TASK_NUMS)]
        [actor_network.cuda() for actor_network in actor_network_list]
        actor_network_optim_list = [torch.optim.Adam(actor_network.parameters(),lr = 0.01) for actor_network in actor_network_list]

        for step in range(STEP):
            # init task config [task_nums*sample_nums,task_config] task_config size=3
            pre_data_samples = torch.cat((pre_x_samples,pre_y_samples),2)
            task_configs = task_config_network(Variable(pre_data_samples).cuda()).repeat(1,SAMPLE_NUMS).view(-1,3)
            # fetch data samples from task generator
            x_samples,y_samples = task_generator.fetch_datas(SAMPLE_NUMS) # [task_nums,sample_nums,x+y`]
            
            # process data samples with actor network list and build the graph
            # 1. for actor network training
            # actor_y_samples: [task_nums*sample_nums,y]
            # actor_x_samples: [task_nums*sample_nums,x]
            # actor_data_samples: [task_nums*sample_nums,x+y+task_config]

            actor_y_samples = actor_network_list[0](Variable(x_samples[0,:]).cuda())
            for i in range(TASK_NUMS - 1):
                actor_y_sample = actor_network_list[i+1](Variable(x_samples[i+1,:]).cuda())
                actor_y_samples = torch.cat((actor_y_samples,actor_y_sample),0)

            actor_x_samples = Variable(x_samples).cuda().resize(TASK_NUMS*SAMPLE_NUMS,1)
            actor_data_samples = torch.cat((actor_x_samples,actor_y_samples,task_configs.detach()),1)
            
            # train actor network
            for i in range(TASK_NUMS):
                actor_network_optim_list[i].zero_grad()
            #actor_criterion = nn.MSELoss()
            #target_y = Variable(y_samples).cuda().resize(TASK_NUMS*SAMPLE_NUMS,1)
            #actor_network_loss = actor_criterion(actor_y_samples,target_y)
            actor_network_loss = - torch.sum(meta_value_network(actor_data_samples))/(TASK_NUMS*SAMPLE_NUMS) #+ actor_criterion(actor_y_samples,target_y)
            actor_network_loss.backward()
            for i in range(TASK_NUMS):
                actor_network_optim_list[i].step()
            
            # 2. train meta value network
            # value_data_samples: Variable([task_nums*sample_nums,x+y+task_config])
            # target_values: Variable([task_nums*sample_nums,-(y-y`)^2])
            meta_value_network_optim.zero_grad()
            task_config_network_optim.zero_grad()
            value_data_samples = torch.cat((actor_x_samples,actor_y_samples.detach(),task_configs),1)
            target_values = - (Variable(y_samples).cuda().resize(TASK_NUMS*SAMPLE_NUMS,1) - actor_y_samples.detach()).pow(2)
            values = meta_value_network(value_data_samples)
            criterion = nn.MSELoss()
            meta_value_network_loss = criterion(values,target_values)
            meta_value_network_loss.backward()
            meta_value_network_optim.step()
            task_config_network_optim.step()

            
            if (step+1) % 10 == 0:
                print ('Epoch [%d/%d], Loss: %.4f,value Loss: %.4f'
                    %(step+1, episode, actor_network_loss.data[0],meta_value_network_loss.data[0]))
        	
            pre_x_samples = x_samples
            pre_y_samples = y_samples

        
        if (episode+1) % 100 == 0 :
            # Save meta value network
            torch.save(meta_value_network.state_dict(),"meta_value_network.pkl")
            torch.save(task_config_network.state_dict(),"task_config_network.pkl")
            print("save networks for episode:",episode+1)


if __name__ == '__main__':
    main()
