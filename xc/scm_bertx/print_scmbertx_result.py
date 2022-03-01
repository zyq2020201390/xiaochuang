import sys
import datetime
import json
import numpy as np
from tqdm import tqdm
import torch
from scmbertx import SCMBertX, SCMBertX3, SCMBertX4
from transformers import AdamW
from build_pretrain_scmbertx_dataloader import build_pretrain_scmbertx_dataloader
from LogPrint import Logger

use_gpu = False
device = torch.device('cpu')
if use_gpu and torch.cuda.is_available():
    device = torch.device('cuda') 
each_test_epoch = 1
epoch_num = 64
batch_size = 1
lr = 1e-6
weight_decay = 0.
vocab_path = "/home/xkr/pretrained_bert/ms/vocab.txt"
pretrained_bert_fold = "/home/xkr/pretrained_bert/ms/"
pretrained_bert_config = "/home/xkr/pretrained_bert/ms/bert_config.json"
train_file_path = "./data/train_seg_3_1008.json"
valid_file_path = "./data/valid_seg_3_1008.json"
test_file_path = "./data/test_seg_3_1008.json"
# checkpoint_path = "./checkpoint/0919-1440/"
# load_epoch_num = 12
# checkpoint_path = "./checkpoint/0921-2309/"
# load_epoch_num = 7
# checkpoint_path = "./checkpoint/0923-0907/"
# load_epoch_num = 15
# checkpoint_path = "./checkpoint/1003-1200/"
# load_epoch_num = 21
checkpoint_path = "./checkpoint/1009-1049/"
load_epoch_num = 12

# date_str = datetime.date.today().isoformat()
# sys.stdout = Logger(f"pretrain-scmberts-{date_str}-lr-{lr}-reg-{weight_decay}-bs-{batch_size}.log")

def load_checkpoint(model, optimizer, trained_epoch):
    filename = checkpoint_path + f"scmbertx4-epoch{trained_epoch}.pkl"
    save_params = torch.load(filename, map_location=torch.device('cpu'))
    model.load_state_dict(save_params["model"])
    optimizer.load_state_dict(save_params["optimizer"])
    # trained_epoch = save_params["trained_epoch"]

def put_data_to_device(data, is_dict_form=True):
    if is_dict_form:
        for key in data.keys():
            data[key] = data[key].to(device)
    else:
        data = data.to(device)
    return data

def load_ori_text(file_path):
    data_list = []
    with open(file_path, "r", encoding="utf8") as f:
        for idx, line in enumerate(f):
            x = json.loads(line)
            # A, B, C = x["A"], x["B"], x["C"]
            data_list.append(x)
    return data_list

train_data_loader = build_pretrain_scmbertx_dataloader(train_file_path, vocab_path, batch_size, shuffle=False, max_seq_len=80)
valid_data_loader = build_pretrain_scmbertx_dataloader(valid_file_path, vocab_path, batch_size, shuffle=False, max_seq_len=80)
test_data_loader = build_pretrain_scmbertx_dataloader(test_file_path, vocab_path, batch_size, shuffle=False, max_seq_len=80)

train_ori_data = load_ori_text(train_file_path)
valid_ori_data = load_ori_text(valid_file_path)
test_ori_data = load_ori_text(test_file_path)

criterion = torch.nn.CrossEntropyLoss()
model = SCMBertX4(pretrained_bert_fold, pretrained_bert_config)
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

epoch_start = 0
if load_epoch_num != None:
    epoch_start = load_epoch_num
    load_checkpoint(model, optimizer, load_epoch_num)
print("epoch_start: ", epoch_start)

print_max_count = 20

model.eval()
with torch.no_grad():
    for idx, batch_data in enumerate(valid_data_loader):
        A_batch, B_batch, C_batch, label_batch = batch_data
        A_batch = put_data_to_device(A_batch)
        B_batch = put_data_to_device(B_batch)
        C_batch = put_data_to_device(C_batch)
        label_batch = put_data_to_device(label_batch, False)
        output_batch, att_A_softmax_1, att_B_softmax, F_AB, att_A_softmax_2, att_C_softmax, F_AC\
            = model.triple_text_input(A_batch, B_batch, C_batch, True)

        print("---------------------")
        if valid_ori_data[idx]['label'] == 'B':
            _, A_index = torch.sort(att_A_softmax_1, descending=True)
            A_index = A_index.tolist()
            _, B_index = torch.sort(att_B_softmax, descending=True)
            B_index = B_index.tolist()
            _, AB_index = torch.sort(F_AB, descending=True)
            AB_index = AB_index.tolist()

            A_ori_text = valid_ori_data[idx]['A']
            B_ori_text = valid_ori_data[idx]['B']

            assert len(A_ori_text) == len(A_index)
            assert len(B_ori_text) == len(B_index)

            print("Top A:", A_index[:5])
            print([A_ori_text[a_idx] for a_idx in A_index[:5]])
            print("Top A pair: ", [AB_index[a_idx][:5] for a_idx in A_index[:5]])
            print([B_ori_text[AB_index[a_idx][0]] for a_idx in A_index[:5]])
            print("Top B:", B_index[:5])
            print([B_ori_text[b_idx] for b_idx in B_index[:5]])


        elif valid_ori_data[idx]['label'] == 'C':
            _, A_index = torch.sort(att_A_softmax_2, descending=True)
            A_index = A_index.tolist()
            _, C_index = torch.sort(att_C_softmax, descending=True)
            C_index = C_index.tolist()
            _, AC_index = torch.sort(F_AC, descending=True)
            AC_index = AC_index.tolist()

            A_ori_text = valid_ori_data[idx]['A']
            C_ori_text = valid_ori_data[idx]['C']

            assert len(A_ori_text) == len(A_index)
            assert len(C_ori_text) == len(C_index)

            print("Top A:", A_index[:5])
            print([A_ori_text[a_idx] for a_idx in A_index[:5]])
            print("Top A pair: ", [AC_index[a_idx][:5] for a_idx in A_index[:5]] )
            print([C_ori_text[AC_index[a_idx][0]] for a_idx in A_index[:5]])
            print("Top C:", C_index[:5])
            print([C_ori_text[c_idx] for c_idx in C_index[:5]])
        
        print_max_count -= 1
        if print_max_count == 0:
            break