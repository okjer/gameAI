# coding:UTF-8

"""
网络架构
"""

from __future__ import print_function
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.utils.data as Data
import torchvision

MAX_EPISODE = 100000
N_ACTIONS = 3
MEMORY_SIZE = 50000
MINIBATCH_SIZE = 32
GAMMA = 0.99
INITIAL_EPSILON = 0.07 # 0.1-》0.07 flappy bird -> plane
TARGET_REPLACE_ITER = 200 # should be changed 
LR = 1e-6
CUDA = True

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.conv = nn.Sequential(  # input shape (4,80, 80)
            nn.Conv2d(
                in_channels=4,      # input height
                out_channels=32,    # n_filters
                kernel_size=8,      # filter size
                stride=4,           # filter movement/step
                padding=4,      # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1 padding = (向上取整(H/stride)-1)*stride+kernel_size-H
            ),      # output shape (20, 20，32)
            nn.ReLU(),    # activation
            nn.MaxPool2d(kernel_size=2),    # 在 2x2 空间里向下采样, output shape (10，10，32) 
            nn.Conv2d(32,64,4,2,1),  # output shape (5, 5, 64)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2,padding=1,stride = 2),  # output shape (3, 3, 64)   
            nn.Conv2d(64, 64, 3, 1, 2),  # output shape (3, 3, 64)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2,padding=0),  # output shape (2, 2, 64)
        )
        self.out = nn.Linear(2 * 2 * 64, N_ACTIONS)   # fully connected layer, output action number classes

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output

class DeepQNetwork(nn.Module):
    def __init__(
            self,
            n_actions = N_ACTIONS,
            memory_size = MEMORY_SIZE,
            minibatch_size = MINIBATCH_SIZE,
            gamma = GAMMA,
            epsilon = INITIAL_EPSILON
    ):
        super(DeepQNetwork, self).__init__()
        self.n_actions = N_ACTIONS
        self.memory_size = MEMORY_SIZE
        self.memory_list = []
        self.minibatch_size = MINIBATCH_SIZE
        self.time_step = 0
        self.learn_step_counter = 0
        self.gamma = GAMMA
        self.epsilon = INITIAL_EPSILON#随机选择的概率

        self.current_state = None

        self.eval_net, self.target_net = Net(), Net() #旧网络，目标网络
        
        if (os.path.exists('pytorch_saved_networks/eval_net_params.pkl') and
            os.path.exists('pytorch_saved_networks/target_net_params.pkl')):
            self.eval_net.load_state_dict(torch.load('pytorch_saved_networks/eval_net_params.pkl'))
            self.target_net.load_state_dict(torch.load('pytorch_saved_networks/target_net_params.pkl'))
            print("Successfully loaded:")
        else :print("Could not find old network weights")
        
        if CUDA:
            self.eval_net.cuda()
            self.target_net.cuda()
        
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def reset(self, observation):
        self.current_state = np.stack((observation, observation, observation, observation), axis=0)# shape 4 80 80 把初始图叠成4张
        #print(self.current_state.shape)

    def store_transition(self, observation, action, reward, done, observation_):
        "储存要训练的内容"
        next_state = np.append(observation_, self.current_state[:3,:,:],axis = 0)#取前三个

        self.memory_list.append([self.current_state, action, reward, done, next_state])
        if len(self.memory_list) > self.memory_size:
            del self.memory_list[0]
        self.time_step += 1
        self.current_state = next_state

    def choose_action(self, observation):
        "根据当前状态选择要执行的动作"
        n_actions = self.n_actions

        action = np.zeros(n_actions)
        if random.random() <= self.epsilon:
            action_index = random.randrange(n_actions)
            action[action_index] = 1
        else:
            if CUDA:
                states = torch.FloatTensor(self.current_state).unsqueeze(0).cuda()
                q_value = self.eval_net.forward(states).cpu()
            else:
                states = torch.FloatTensor(self.current_state).unsqueeze(0)
                q_value = self.eval_net.forward(states)
            #这里[1]代表返回下标
            action_index = torch.max(q_value, 1)[1].data.numpy()
            action[action_index] = 1

        # 减去一个很小的数，让训练快结束的时候随机的次数越来越少
        # 这里写了两个奇怪的数字，要考虑能不能写成time_step的函数
        if self.epsilon > 0.0001:
            self.epsilon -= 3.33e-8

        return action

    def learn(self):
        "训练步骤" 
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        
        minibatch = random.sample(self.memory_list, self.minibatch_size)
        
        
        
        if CUDA:
            b_s = torch.FloatTensor([data[0] for data in minibatch]).cuda()
            b_a = torch.FloatTensor(np.array([data[1] for data in minibatch])).cuda()
            b_r = torch.FloatTensor([data[2] for data in minibatch]).cuda()
            b_s_ = torch.FloatTensor([data[4] for data in minibatch]).cuda()
        else:
            b_s = torch.FloatTensor([data[0] for data in minibatch])
            b_a = torch.FloatTensor(np.array([data[1] for data in minibatch]))
            b_r = torch.FloatTensor([data[2] for data in minibatch])
            b_s_ = torch.FloatTensor([data[4] for data in minibatch])
  
        q_eval = torch.sum(torch.mul(self.eval_net(b_s),b_a),1)  # shape (batch, 1)
        #q_eval = torch.sum(torch.mul(self.net(b_s),b_a),1)
        
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate4
        #q_next = self.net(b_s_).detach()
        q_target = b_r + (self.gamma * q_next.max(1)[0])# shape (batch, 1)
       
        
        loss = self.loss_func(q_eval, q_target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.time_step%10000 == 0:
            self.saveNet()
        
    def saveNet(self):
        torch.save(self.eval_net.state_dict(), 'pytorch_saved_networks/eval_net_params.pkl')
        torch.save(self.target_net.state_dict(), 'pytorch_saved_networks/target_net_params.pkl')
        print("Net saved")
