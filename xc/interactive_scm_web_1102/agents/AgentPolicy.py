import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import random
import pickle
import datetime
import numpy as np
from agents.DeepPolicyNetwork import TwoLayersModel

#对话策略模块
class AgentEAR():
    def __init__(self, config, convhis):
        self.convhis = convhis

        self.DPN = TwoLayersModel(config)#选用模型
        self.DPN_model_path = config.DPN_model_path#模型路径
        self.DPN_model_name = config.DPN_model_name#模型名称
        self.aciton_len = config.output_dim

        self.env = None

    def set_env(self, env):
        self.env = env

    def init_episode(self):
        self.DPN.eval()
        pass

    #模型存储
    def save_model(self, is_preatrain):
        if is_preatrain:
            name_suffix = "_PRE"#后缀
        else:
            name_suffix = "_PG"
        torch.save(self.DPN.state_dict(), "/".join([self.DPN_model_path, self.DPN_model_name + name_suffix]))

    #模型加载
    def load_model(self, is_preatrain):
        if is_preatrain:
            name_suffix = "_PRE"
        else:
            name_suffix = "_PG"
        self.DPN.load_state_dict(torch.load("/".join([self.DPN_model_path, self.DPN_model_name + name_suffix])))

    #预训练
    def pretrain_batch_data_input(self, batchdata, is_train):
        if is_train:
            self.DPN.train()#model.train()的作用是启用 Batch Normalization 和 Dropout。
            # 如果模型中有BN层(Batch Normalization）和Dropout，需要在训练时添加model.train()。
            # model.train()是保证BN层能够用到每一批数据的均值和方差。对于Dropout，model.train()是随机取一部分网络连接来训练更新参数。
        else:
            self.DPN.eval()#不启用 Batch Normalization 和 Dropout。
            # 如果模型中有BN层(Batch Normalization）和Dropout，在测试时添加model.eval()。
            # model.eval()是保证BN层能够用全部训练数据的均值和方差，即测试过程中要保证BN层的均值和方差不变。
            # 对于Dropout，model.eval()是利用到了所有网络连接，即不进行随机舍弃神经元。

        output = self.DPN(batchdata.state_list, not is_train)
        if not is_train:#如果不训练，也就是测试
            output = torch.argmax(output, -1)#返回指定维度最大值的序号 最后一个维度会消失
        return output

    #PG（强化学习）一大轮训练
    def PG_train_one_episode(self, t_data):
        self.DPN.train()
        state_pool = []
        action_pool = []
        reward_pool = [] 

        state = self.env.initialize_episode(t_data[0], t_data[1])
        IsOver = False
        while not IsOver:
            attribute_distribution = self.DPN(state, True)
            asked_list = self.convhis.get_asked_list()
            attribute_distribution[asked_list] = -float("inf") 
            c = Categorical(logits = attribute_distribution)
            action = c.sample()
            IsOver, next_state, reward = self.env.step(int(action))

            state_pool.append(state)
            action_pool.append(c.log_prob(action))
            reward_pool.append(reward)
            if not IsOver:
                state = next_state

        return action_pool, reward_pool

    #PG（强化学习）一大轮测试
    def test_one_episode(self, t_data, silence=True):
        self.DPN.eval()
        total_reward = 0.
        turn_count = 0
        success = 0
        action_list = []

        state = self.env.initialize_episode(t_data[0], t_data[1], silence)
        IsOver = False
        while not IsOver:
            turn_count += 1
            attribute_distribution = self.DPN(state, True)
            asked_list = self.convhis.get_asked_list()
            attribute_distribution[asked_list] = 0.
            action = int(attribute_distribution.argmax())
            action_list.append(action)
            IsOver, next_state, reward = self.env.step(int(action))
            total_reward += reward
            if not IsOver:
                state = next_state
            else:
                if reward > 0.:
                    success = 1
                else:
                    success = 0

        return total_reward, turn_count, success, action_list   