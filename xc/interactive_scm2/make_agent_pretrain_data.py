import multiprocessing as mp
import time
from queue import Queue
import sys
import os
import random
from tqdm import tqdm
import datetime
import json
import math
import pickle
import torch
from convhis.ConvHis import ConvHis
from convhis.ConvHisConfig import ConvHisConfig
from agents.AgentRule import AgentRule
from agents.AgentRuleConfig import AgentRuleConfig
from recommendersystem.RecModule import RecModule
from recommendersystem.RecModuleConfig import RecModuleConfig
from user.UserSim import UserSim
from user.UserSimConfig import UserSimConfig
from dialoguemanager.DialogueManager import DialogueManager
from dialoguemanager.DialogueManagerConfig import DialogueManagerConfig

random.seed(1534)

def consumer(q):
    # dialogue_state_list = []
    current_state_list = []
    action_list = []
    exit_flag = False
    while not exit_flag:
        while not q.empty():
            res=q.get()
            # print("res:", res)
            if res is None:
                exit_flag = True
                break
            current_state, action = res
            current_state_list.append(current_state)
            action_list.append(action)

    with open('./data/agents/RL_pretrain_data.pkl','wb') as f:
        pickle.dump([current_state_list, action_list], f)

def job(train_pair,q):
    ch = ConvHis(ConvHisConfig())
    agent = AgentRule(AgentRuleConfig(), ch)
    rec = RecModule(RecModuleConfig(), convhis=ch)
    rec.init_eval()
    usersim = UserSim(UserSimConfig())
    dm = DialogueManager(DialogueManagerConfig(), rec, agent, usersim, ch)

    train_index_list = [_ for _ in range(len(train_pair))]
    for index in tqdm(train_index_list, ncols=0):
        input_case, target_case = train_pair[index]

        dm.initialize_dialogue(A, B, True)
        is_over = False
        while not is_over:
            current_state = dm.get_current_agent_state(False)
            is_over, reward, _ = dm.next_turn()
            action = dm.get_current_agent_action()
            q.put([current_state, action])

if __name__=='__main__':
    q = mp.Queue()
    process = []
    process_num = 4

    train_pair = []
    with open("./data/train_id.json", "r") as f:
        train_data_list = json.load(f)
    for A, B, C in train_data_list:
        train_pair.append([A, B])
    # train_pair = train_pair[:100]

    batch_size = math.ceil(len(train_pair)/process_num)
    for index in range(process_num):
        t = mp.Process(target=job,args=(train_pair[batch_size*index:batch_size*(index+1)],q))
        t.start()
        process.append(t)
    cos = mp.Process(target=consumer,args=(q,))
    cos.start()
    for p in process:
        p.join()
    q.put(None)
    cos.join()
