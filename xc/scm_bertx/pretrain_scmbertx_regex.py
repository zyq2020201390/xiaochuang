import sys
import datetime
import json
from tqdm import tqdm
import torch
from scmbertx import SCMBertX_regex
from transformers import AdamW
from build_pretrain_scmbertx_dataloader_regex import build_pretrain_scmbertx_dataloader
from LogPrint import Logger


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
each_test_epoch = 1
epoch_num = 64
batch_size = 1
real_batch_size = 1
lr = 2e-6
weight_decay = 0.
vocab_path = "/home/xkr/pretrained_bert/ms/vocab.txt"
pretrained_bert_fold = "/home/xkr/pretrained_bert/ms/"
pretrained_bert_config = "/home/xkr/pretrained_bert/ms/bert_config.json"
checkpoint_path = "./checkpoint/"
train_file_path = "./data/train_seg_3.json"
valid_file_path = "./data/valid_seg_3.json"
test_file_path = "./data/test_seg_3.json"
load_epoch_num = None
max_grad_norm = 1.0

date_str = datetime.date.today().isoformat()
sys.stdout = Logger(f"pretrain-scmbertx_regex-{date_str}-lr-{lr}-reg-{weight_decay}-clip-{max_grad_norm}-bs-{real_batch_size}.log")

def save_checkpoint(model, optimizer, trained_epoch):
    save_params = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "trained_epoch": trained_epoch,
    }
    filename = checkpoint_path + f"scmbertx_regex-epoch{trained_epoch}.pkl"
    torch.save(save_params, filename)

def load_checkpoint(model, optimizer, trained_epoch):
    filename = checkpoint_path + f"scmbertx_regex-epoch{trained_epoch}.pkl"
    save_params = torch.load(filename)
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

train_data_loader = build_pretrain_scmbertx_dataloader(train_file_path, vocab_path, batch_size, max_seq_len=80)
valid_data_loader = build_pretrain_scmbertx_dataloader(valid_file_path, vocab_path, batch_size, max_seq_len=80)
test_data_loader = build_pretrain_scmbertx_dataloader(test_file_path, vocab_path, batch_size, max_seq_len=80)
criterion = torch.nn.CrossEntropyLoss()
model = SCMBertX_regex(pretrained_bert_fold, pretrained_bert_config)
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, correct_bias=False)

epoch_start = 0
if load_epoch_num != None:
    epoch_start = load_epoch_num
    load_checkpoint(model, optimizer, load_epoch_num)

for epoch in range(epoch_start, epoch_num):
    epoch_loss = 0.
    current_step = 0
    model.train()

    total_loss = 0.

    for batch_data in tqdm(train_data_loader, ncols=0):
        A_batch, B_batch, C_batch, A_one_hot_batch, B_one_hot_batch, C_one_hot_batch, label_batch = batch_data
        # print("A_batch: ", A_batch)
        # print("label_batch: ", label_batch)
        A_batch = put_data_to_device(A_batch)
        B_batch = put_data_to_device(B_batch)
        C_batch = put_data_to_device(C_batch)
        A_one_hot_batch = put_data_to_device(A_one_hot_batch, False)
        B_one_hot_batch = put_data_to_device(B_one_hot_batch, False)
        C_one_hot_batch = put_data_to_device(C_one_hot_batch, False)        
        label_batch = put_data_to_device(label_batch, False)
        output_batch = model.triple_text_input(A_batch, B_batch, C_batch, A_one_hot_batch, B_one_hot_batch, C_one_hot_batch)
        loss = criterion(output_batch.unsqueeze(0), label_batch.unsqueeze(0))
        # print(output_batch, label_batch, loss)
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        loss_item = loss.cpu().detach().item()
        epoch_loss += loss_item
        current_step += 1
        # break

        total_loss += loss
        if current_step % real_batch_size == 0:
            ave_loss = total_loss / real_batch_size
            optimizer.zero_grad()
            ave_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            total_loss = 0.


    epoch_loss = epoch_loss / current_step
    time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{time_str} train epoch {epoch} loss {epoch_loss}")

    if (epoch + 1) % each_test_epoch == 0:
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for batch_data in tqdm(valid_data_loader, ncols=0):
                A_batch, B_batch, C_batch, A_one_hot_batch, B_one_hot_batch, C_one_hot_batch, label_batch = batch_data
                A_batch = put_data_to_device(A_batch)
                B_batch = put_data_to_device(B_batch)
                C_batch = put_data_to_device(C_batch)
                A_one_hot_batch = put_data_to_device(A_one_hot_batch, False)
                B_one_hot_batch = put_data_to_device(B_one_hot_batch, False)
                C_one_hot_batch = put_data_to_device(C_one_hot_batch, False)  
                label_batch = put_data_to_device(label_batch, False)
                output_batch = model.triple_text_input(A_batch, B_batch, C_batch, A_one_hot_batch, B_one_hot_batch, C_one_hot_batch)
                _, predicted_output = torch.max(output_batch, -1)
                # total += label_batch.size(0)
                # correct += (predicted_output == label_batch).sum().cpu().item()
                total += 1
                correct += (predicted_output == label_batch).cpu().item()
                # break
            time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{time_str} valid epoch {epoch} acc {correct}/{total}={correct/total}")

            correct = 0
            total = 0
            for batch_data in tqdm(test_data_loader, ncols=0):
                A_batch, B_batch, C_batch, A_one_hot_batch, B_one_hot_batch, C_one_hot_batch, label_batch = batch_data
                A_batch = put_data_to_device(A_batch)
                B_batch = put_data_to_device(B_batch)
                C_batch = put_data_to_device(C_batch)
                A_one_hot_batch = put_data_to_device(A_one_hot_batch, False)
                B_one_hot_batch = put_data_to_device(B_one_hot_batch, False)
                C_one_hot_batch = put_data_to_device(C_one_hot_batch, False) 
                label_batch = put_data_to_device(label_batch, False)
                output_batch = model.triple_text_input(A_batch, B_batch, C_batch, A_one_hot_batch, B_one_hot_batch, C_one_hot_batch)
                _, predicted_output = torch.max(output_batch, -1)
                # total += label_batch.size(0)
                # correct += (predicted_output == label_batch).sum().cpu().item()
                total += 1
                correct += (predicted_output == label_batch).cpu().item()
                # break
            time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{time_str} test epoch {epoch} acc {correct}/{total}={correct/total}")
        save_checkpoint(model, optimizer, epoch)