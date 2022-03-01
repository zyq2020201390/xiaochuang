import argparse
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import json
import random
import pickle
import datetime
import numpy as np
from tqdm import tqdm
from utils.LogPrint import Logger
from convhis.ConvHis import ConvHis
from convhis.ConvHisConfig import ConvHisConfig
from agents.AgentRule import AgentRule
from agents.AgentRuleConfig import AgentRuleConfig
from agents.AgentPolicy import AgentPolicy
from agents.AgentPolicyConfig import AgentPolicyConfig
from recommendersystem.RecModule import RecModule
from recommendersystem.RecModuleConfig import RecModuleConfig
from user.UserSim import UserSim
from user.UserSimConfig import UserSimConfig
from dialoguemanager.DialogueManager import DialogueManager
from dialoguemanager.DialogueManagerConfig import DialogueManagerConfig
from collections import deque#双向队列

random.seed(1021)
eps = np.finfo(np.float32).eps.item()
#print(eps)
#非负的最小值
class PretrainBatchData():
    def __init__(self, state_list, label_list, use_gpu, output_check=False):
        batch_size = len(state_list)
        self.state_list = torch.tensor(state_list)
        self.label_list = torch.tensor(label_list)
        if use_gpu:
            self.state_list = self.state_list.cuda()
            self.label_list = self.label_list.cuda()
        if output_check:
            self.output()

    def output(self):
        print("--------------------------")
        print("state_list:", self.state_list)
        print("label_list:", self.label_list)


def load_pretrain_data(pretrain_data_path, pretrain_batch_size, use_gpu=False):
    with open(pretrain_data_path, 'rb') as f:
        dialogue_state_list, label_list = pickle.load(f)
    assert len(dialogue_state_list) == len(label_list)

    all_list = list(zip(dialogue_state_list, label_list)) 
    random.shuffle(all_list)   
    dialogue_state_list, label_list = zip(*all_list)

    data_num = len(dialogue_state_list)
    data_num = data_num // 20

    test_state_list = dialogue_state_list[:data_num:]
    train_state_list = dialogue_state_list[data_num:]
    test_label_list = label_list[:data_num]
    train_label_list = label_list[data_num:]     

    print("train_state_list: {}, test_state_list: {}".format(len(train_state_list), len(test_state_list)))
    return train_state_list, train_label_list, test_state_list, test_label_list

def make_batch_data(state_list, label_list, batch_size, use_gpu):
    all_list = list(zip(state_list, label_list)) 
    random.shuffle(all_list)   
    state_list, label_list = zip(*all_list)

    max_iter = len(state_list)//batch_size
    if max_iter * batch_size < len(state_list):
        max_iter += 1

    batch_data_list = []
    for index in range(max_iter):
        left_index = index * batch_size
        right_index = (index+1) * batch_size
        batch_data = PretrainBatchData(state_list[left_index:right_index], label_list[left_index:right_index], use_gpu)
        batch_data_list.append(batch_data)
    return batch_data_list

def pretrain(agent):
    use_gpu = False
    pretrain_data_path = "./data/agents/RL_pretrain_data.pkl"
    pretrain_epoch_num = 100
    pretrain_weight_decay = 1e-4
    pretrain_lr = 1e-3
    pretrain_optimizer = optim.Adam(agent.DPN.parameters(), lr=pretrain_lr, weight_decay=pretrain_weight_decay)
    label_weight = torch.tensor([1.] * 17 + [2.])
    pretrain_criterion = nn.CrossEntropyLoss(weight = label_weight)
    # pretrain_criterion = nn.CrossEntropyLoss()
    pretrain_batch_size = 64
    pretrain_save_epoch = 1

    date_str = datetime.date.today().isoformat()
    sys.stdout = Logger("pretrain-agentpolicy-{}-lr-{}-reg-{}-bs-{}-weight.log"\
                        .format(date_str,str(pretrain_lr),str(pretrain_weight_decay),str(pretrain_batch_size)))

    print("prepare pretrain data...")
    train_state_list, train_label_list, test_state_list, test_label_list \
        = load_pretrain_data(pretrain_data_path, pretrain_batch_size, use_gpu)
    pretrain_test_data_list = make_batch_data(test_state_list, test_label_list, pretrain_batch_size, use_gpu)
    time_str = datetime.datetime.now().isoformat()
    print("{} start pretraining ...".format(time_str))
    print("lr: {:g}, batch_size: {}".format(pretrain_lr, pretrain_batch_size))
    best_acc = 0.
    best_acc_count = 0

    for _ in range(pretrain_epoch_num):
        print("epoch: ", _)
        loss_list = []
        pretrain_train_data_list = make_batch_data(train_state_list, train_label_list, pretrain_batch_size, use_gpu)
        pretrain_data_index_list = [_ for _ in range(len(pretrain_train_data_list))]
        # random.shuffle(pretrain_data_index_list)
        for pretrain_data_index in tqdm(pretrain_data_index_list, ncols=0):
            t_batch_data = pretrain_train_data_list[pretrain_data_index]
            output = agent.pretrain_batch_data_input(t_batch_data, True)
            loss = pretrain_criterion(output, t_batch_data.label_list)
            loss_list.append(loss.item())
            pretrain_optimizer.zero_grad()
            loss.backward()
            pretrain_optimizer.step()

            time_str = datetime.datetime.now().isoformat()
        epoch_loss = np.mean(np.array(loss_list))
        print("{}: epoch {}, loss {:g}".format(time_str, _, epoch_loss))

        if _ % pretrain_save_epoch == 0:
            print("start evaluation")
            pre_label_list = []
            gt_label_list = []
            for e_batch_data in tqdm(pretrain_test_data_list, ncols=0):
                output = agent.pretrain_batch_data_input(e_batch_data, False)
                pre_label_list.extend(output.tolist())
                gt_label_list.extend(e_batch_data.label_list.tolist())
            cur_acc = accuracy_score(gt_label_list, pre_label_list)
            if cur_acc > best_acc:
                best_acc = cur_acc
                best_acc_count = 0
                agent.save_model(True)
            else:
                best_acc_count += 1
            print("{}: epoch {}, accuracy {:g}, best accuracy {:g}".format(time_str, str(_), cur_acc, best_acc))
            print(classification_report(gt_label_list, pre_label_list))

            if best_acc_count == 5:
                break

def standard_reward(a):
    return (a - np.mean(a)) / (np.std(a) + eps)

def PG_train(agent, load_model_type, dm):
    use_gpu = False
    env = dm
    PG_lr = 0.001
    PG_discount_rate = agent.PG_discount_rate
    # PG_batch_size = 64
    PG_epoch_num = 100
    reward_scale = False
    optim_name = "SGD"

    if optim_name == 'Adam':
        PG_optimizer = optim.Adam(agent.DPN.parameters(), lr=PG_lr,)
    if optim_name == 'RMSprop':
        PG_optimizer = optim.RMSprop(agent.DPN.parameters(), lr=PG_lr,)
    if optim_name == 'SGD':
        PG_optimizer = optim.SGD(agent.DPN.parameters(), lr=PG_lr,)
    PG_save_epoch = 1

    train_data_path = "./data/train_id.json"
    valid_data_path = "./data/valid_id.json"
    test_data_path = "./data/test_id.json"
    with open(train_data_path, 'r') as f:
        PG_train_data_list = json.load(f)
    with open(valid_data_path, 'r') as f:
        PG_valid_data_list = json.load(f)
    with open(test_data_path, 'r') as f:
        PG_test_data_list = json.load(f)

    # PG_train_data_list = PG_train_data_list[:10]
    date_str = datetime.date.today().isoformat()
    sys.stdout = Logger("PG-agentpolicy-{}-lr-{}-disconut-{}-opt-{}-scale-{}-baseline.log"\
                        .format(date_str, str(PG_lr), str(PG_discount_rate), optim_name, str(reward_scale)))
    print("PG_train_data_list: {}, PG_valid_data_list:{}, PG_test_data_list:{}".\
            format(len(PG_train_data_list), len(PG_valid_data_list), len(PG_test_data_list)))

    agent.set_env(env)
    if load_model_type == "pretrain":
        print("load pretrain model ...")
        agent.load_model(True)
    elif load_model_type == "PG":
        print("load PG model ...")
        agent.load_model(False)
    else:
        print("no pretrian model...")

    time_str = datetime.datetime.now().isoformat()
    print("{} start PG ...".format(time_str))
    print("lr: {:g}".format(PG_lr))

    # step = 0
    best_average_reward = -1e9
    best_average_turn = 100.
    best_success_rate = 0.
    best_count = 0
    # each_epoch_len = len(PG_train_data_list) // 10
    each_epoch_len = len(PG_train_data_list)

    max_conv_length = DialogueManagerConfig().turn_limit
    baseline_list = [deque(maxlen=5102)] * max_conv_length
    for _ in baseline_list:
        _.append(0.)

    for _ in range(PG_epoch_num):
        print("epoch: ", _)
        epoch_reward_sum = 0.
        epoch_turn_sum = 0.
        epoch_success_sum = 0.
        PG_data_index_list = [_ for _ in range(len(PG_train_data_list))]
        random.shuffle(PG_data_index_list)
        PG_data_index_list = PG_data_index_list[:each_epoch_len]
        for PG_data_index in tqdm(PG_data_index_list,ncols=0):
            A, B, C = PG_train_data_list[PG_data_index]
            # step += 1
            action_pool, reward_pool = agent.PG_train_one_episode(A, B)
            epoch_reward_sum += sum(reward_pool)
            epoch_turn_sum += len(reward_pool)
            epoch_success_sum += (reward_pool[-1] > 0.)

            total_reward = 0.
            for index in reversed(range(len(reward_pool))):
                total_reward = total_reward * PG_discount_rate + reward_pool[index]
                
                baseline = sum(list(baseline_list[index])) / len(baseline_list[index])
                baseline_list[index].append(total_reward)
                reward_pool[index] = total_reward - baseline

            reward_pool = np.array(reward_pool)
            if reward_scale:
                reward_pool = standard_reward(reward_pool)

            reward_pool_tensor = torch.from_numpy(reward_pool)
            action_pool_tensor = torch.stack(action_pool, 0)
            if use_gpu:
                reward_pool_tensor = reward_pool_tensor.cuda()
                action_pool_tensor = action_pool_tensor.cuda()

            loss = torch.sum(torch.mul(action_pool_tensor, reward_pool_tensor).mul(-1))
            PG_optimizer.zero_grad()
            loss.backward()
            PG_optimizer.step()

        time_str = datetime.datetime.now().isoformat()
        print("{}:train epoch {}, reward {:g}, turn {:g}, success {:g}".\
                format(time_str, _, epoch_reward_sum/len(PG_train_data_list), \
                    epoch_turn_sum/len(PG_train_data_list), epoch_success_sum/len(PG_train_data_list)))

        if (_+1) % PG_save_epoch == 0:
            with torch.no_grad():
                sum_reward = 0.
                sum_turn = 0
                sum_success = 0
                episode_num = 0
                for e_data in tqdm(PG_valid_data_list,ncols=0):
                    A, B, C = e_data
                    reward, turn, success, action_list = agent.test_one_episode(A, B)
                    episode_num += 1
                    sum_reward += reward
                    sum_turn += turn
                    sum_success += success      

                valid_average_reward = float(sum_reward)/episode_num
                valid_average_turn = float(sum_turn)/episode_num
                valid_success_rate = float(sum_success)/episode_num
                time_str = datetime.datetime.now().isoformat()
                print("{}: valid epoch {}, average_reward {:g}, average_turn {:g}, success_rate {:g}"\
                        .format(time_str, _, valid_average_reward, valid_average_turn, valid_success_rate)) 

                sum_reward = 0.
                sum_turn = 0
                sum_success = 0
                episode_num = 0
                for e_data in tqdm(PG_test_data_list,ncols=0):
                    reward, turn, success, action_list = agent.test_one_episode(e_data[0], e_data[1])
                    episode_num += 1
                    sum_reward += reward
                    sum_turn += turn
                    sum_success += success      

                test_average_reward = float(sum_reward)/episode_num
                test_average_turn = float(sum_turn)/episode_num
                test_success_rate = float(sum_success)/episode_num
                print("{}: test epoch {}, average_reward {:g}, average_turn {:g}, success_rate {:g}"\
                        .format(time_str, _, test_average_reward, test_average_turn, test_success_rate))    
    

                if valid_average_reward > best_average_reward:
                    best_average_reward = valid_average_reward
                    best_average_turn = valid_average_turn
                    best_success_rate = valid_success_rate
                    agent.save_model(False) 
                    best_count = 0
                else:
                    best_count += 1 

                time_str = datetime.datetime.now().isoformat()
                print("{}: valid till epoch {}, best_average_reward: {:g}, best_average_turn {:g}, best_success_rate {:g}"\
                        .format(time_str, _, best_average_reward, best_average_turn, best_success_rate))    

                # if best_count == 10:
                #     break


parser = argparse.ArgumentParser(description='train ear agent')
parser.add_argument('--mode', type=str, 
                    help='choose from pretrain or PG')
args = parser.parse_args()

if args.mode == "pretrain":
    # preatrain
    agent = AgentPolicy(AgentPolicyConfig(), None)
    pretrain(agent)
elif args.mode == "PG":
    # PG
    ch = ConvHis(ConvHisConfig())
    agent = AgentPolicy(AgentPolicyConfig(), ch)
    rec = RecModule(RecModuleConfig(), convhis=ch)
    rec.init_eval()
    usersim = UserSim(UserSimConfig())
    dm = DialogueManager(DialogueManagerConfig(), rec, agent, usersim, ch)
    PG_train(agent, "pretrain", dm)
else:
    print("Not support {}. Choose from pretrain or PG".format(args.mode))