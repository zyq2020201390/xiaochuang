import sys
import datetime
import json
from tqdm import tqdm
import torch
from scmbertx import SCMBertX, SCMBertX2, SCMBertX3, SCMBertX4
from transformers import AdamW
from build_pretrain_scmbertx_dataloader import build_pretrain_scmbertx_dataloader
from LogPrint import Logger


device = torch.device('cuda:0,1') if torch.cuda.is_available() else torch.device('cpu')

each_test_epoch = 1
epoch_num = 64
batch_size = 1
real_batch_size = 1
lr = 3e-6
weight_decay = 0.#权重衰减
#vocab_path = "/home/kerui_xu/pretrained_bert/ms/vocab.txt"
vocab_path=r"C:\Users\13178\Desktop\xiaochuang\pretrained_bert\ms\vocab.txt" #词汇表
pretrained_bert_fold = r"C:\Users\13178\Desktop\xiaochuang\pretrained_bert\ms" #ms文件夹
pretrained_bert_config = r"C:\Users\13178\Desktop\xiaochuang\pretrained_bert\ms\bert_config.json" #bert_config参数设置
checkpoint_path = "./checkpoint/" #checkpoint文件夹
train_file_path = "./data/train_seg_3_1008.json" #训练集{“B”:["逗号分割后的一句话","",""],"A":["",""],"label": "C", "C": ["",""]}   {“B”:["",""],"A":["",""]}
valid_file_path = "./data/valid_seg_3_1008.json"#训练过程中的测试集{“B”:["逗号分割后的一句话","",""],"A":["",""],"label": "C", "C": ["",""]}{“B”:["",""],"A":["",""]}
test_file_path = "./data/test_seg_3_1008.json" #训练结束后用于评价模型的测试集{“B”:["逗号分割后的一句话","",""],"C":["",""],"label": "C", "C": ["",""]}{“B”:["",""],"A":["",""]}
load_epoch_num = None
max_grad_norm = None

date_str = datetime.date.today().isoformat()
sys.stdout = Logger(f"pretrain-scmbertx4-{date_str}-lr-{lr}-reg-{weight_decay}-clip-{max_grad_norm}-bs-{real_batch_size}.log")

#保存模型权重
def save_checkpoint(model, optimizer, trained_epoch):
    save_params = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "trained_epoch": trained_epoch,
    }
    filename = checkpoint_path + f"scmbertx4-epoch{trained_epoch}.pkl"
    torch.save(save_params, filename)

#加载模型权重
def load_checkpoint(model, optimizer, trained_epoch):
    filename = checkpoint_path + f"scmbertx4-epoch{trained_epoch}.pkl"
    save_params = torch.load(filename)
    model.load_state_dict(save_params["model"])
    optimizer.load_state_dict(save_params["optimizer"])
    # trained_epoch = save_params["trained_epoch"]

#
def put_data_to_device(data, is_dict_form=True):
    if is_dict_form:
        for key in data.keys():
            data[key] = data[key].to(device)
    else:
        data = data.to(device)
    return data

#dataloader类
train_data_loader = build_pretrain_scmbertx_dataloader(train_file_path, vocab_path, batch_size, max_seq_len=80)
valid_data_loader = build_pretrain_scmbertx_dataloader(valid_file_path, vocab_path, batch_size, max_seq_len=80)
test_data_loader = build_pretrain_scmbertx_dataloader(test_file_path, vocab_path, batch_size, max_seq_len=80)

#损失函数
criterion = torch.nn.CrossEntropyLoss()

#模型
model = SCMBertX4(pretrained_bert_fold, pretrained_bert_config)
model = model.to(device)

#优化器
optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

epoch_start = 0

if load_epoch_num != None:
    epoch_start = load_epoch_num
    load_checkpoint(model, optimizer, load_epoch_num)

for epoch in range(epoch_start, epoch_num):#每一轮
    epoch_loss = 0.
    current_step = 0
    model.train()

    total_loss = 0.

    for batch_data in tqdm(train_data_loader, ncols=0): #每一个batch
        A_batch, B_batch, C_batch, label_batch = batch_data
        #print("A_batch: ", A_batch)
        # print("label_batch: ", label_batch)
        A_batch = put_data_to_device(A_batch)
        B_batch = put_data_to_device(B_batch)
        C_batch = put_data_to_device(C_batch)
        label_batch = put_data_to_device(label_batch, False)
        output_batch = model.triple_text_input(A_batch, B_batch, C_batch)
        # print("output_batch: ", output_batch)
        loss = criterion(output_batch.unsqueeze(0), label_batch.unsqueeze(0))
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        loss_item = loss.cpu().detach().item()
        epoch_loss += loss_item
        current_step += 1
        # break

        #loss计算
        total_loss += loss

        if current_step % real_batch_size == 0:
            ave_loss = total_loss / real_batch_size
            optimizer.zero_grad()
            ave_loss.backward()
            if max_grad_norm is not None:
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
                A_batch, B_batch, C_batch, label_batch = batch_data
                A_batch = put_data_to_device(A_batch)
                B_batch = put_data_to_device(B_batch)
                C_batch = put_data_to_device(C_batch)
                label_batch = put_data_to_device(label_batch, False)
                output_batch = model.triple_text_input(A_batch, B_batch, C_batch)
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
                A_batch, B_batch, C_batch, label_batch = batch_data
                A_batch = put_data_to_device(A_batch)
                B_batch = put_data_to_device(B_batch)
                C_batch = put_data_to_device(C_batch)
                label_batch = put_data_to_device(label_batch, False)
                output_batch = model.triple_text_input(A_batch, B_batch, C_batch)
                _, predicted_output = torch.max(output_batch, -1)
                # total += label_batch.size(0)
                # correct += (predicted_output == label_batch).sum().cpu().item()
                total += 1
                correct += (predicted_output == label_batch).cpu().item()
                # break
            time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{time_str} test epoch {epoch} acc {correct}/{total}={correct/total}")
        save_checkpoint(model, optimizer, epoch)
