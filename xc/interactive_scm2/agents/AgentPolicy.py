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

class AgentPolicy():
    def __init__(self, config, convhis):
        self.convhis = convhis

        self.DPN = TwoLayersModel(config)
        self.DPN_model_path = config.DPN_model_path
        self.DPN_model_name = config.DPN_model_name
        self.aciton_len = config.output_dim
        self.PG_discount_rate = config.PG_discount_rate

        self.env = None

    def train(self):
        self.DPN.train()

    def eval(self):
        self.DPN.eval()

    def set_env(self, env):
        self.env = env

    def init_episode(self):
        self.DPN.eval()
        pass

    def save_model(self, is_preatrain):
        if is_preatrain:
            name_suffix = "_PRE"
        else:
            name_suffix = "_PG"
        torch.save(self.DPN.state_dict(), "/".join([self.DPN_model_path, self.DPN_model_name + name_suffix]))

    def load_model(self, is_preatrain):
        if is_preatrain:
            name_suffix = "_PRE"
        else:
            name_suffix = "_PG"
        self.DPN.load_state_dict(torch.load("/".join([self.DPN_model_path, self.DPN_model_name + name_suffix])))

    def pretrain_batch_data_input(self, batchdata, is_train):
        if is_train:
            self.DPN.train()
        else:
            self.DPN.eval()

        output = self.DPN(batchdata.state_list, not is_train)
        if not is_train:
            output = torch.argmax(output, -1)
        return output

    def PG_train_one_episode(self, A, B):
        self.DPN.train()
        state_pool = []
        action_pool = []
        action_index_pool = []
        reward_pool = [] 

        state = self.env.initialize_episode(A, B)
        IsOver = False
        while not IsOver:
            attribute_logits = self.DPN(state, False)
            asked_list = self.convhis.get_asked_list()
            attribute_logits[asked_list] = -float("inf") 
            c = Categorical(logits = attribute_logits)
            action = c.sample()
            IsOver, next_state, reward = self.env.step(int(action))                
            state_pool.append(state)
            action_index_pool.append(action)
            action_pool.append(c.log_prob(action))
            reward_pool.append(reward)
            if not IsOver:
                state = next_state
        # print("state_pool: ", state_pool)
        # print("action_index_pool: ", action_index_pool)
        # print("reward_pool: ", reward_pool)
        return action_pool, reward_pool

    def test_one_episode(self, A, B, silence=True):
        self.DPN.eval()
        total_reward = 0.
        turn_count = 0
        success = 0
        action_list = []

        state = self.env.initialize_episode(A, B, silence)
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