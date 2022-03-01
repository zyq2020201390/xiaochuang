import sys
import datetime
import json
from tqdm import tqdm
import torch
from scmbertx import SCMBertX4
from transformers import BertTokenizer
import numpy as np
import pickle


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
vocab_path = "/home/xkr/pretrained_bert/ms/vocab.txt"
pretrained_bert_fold = "/home/xkr/pretrained_bert/ms/"
pretrained_bert_config = "/home/xkr/pretrained_bert/ms/bert_config.json"
checkpoint_path = "./checkpoint/1009-1049/"
load_epoch_num = 12
all_text_count = 1649

def load_checkpoint(model, trained_epoch):
    filename = checkpoint_path + f"scmbertx4-epoch{trained_epoch}.pkl"
    save_params = torch.load(filename)
    model.load_state_dict(save_params["model"])
    # optimizer.load_state_dict(save_params["optimizer"])
    # trained_epoch = save_params["trained_epoch"]

def put_data_to_device(data, is_dict_form=True):
    if is_dict_form:
        for key in data.keys():
            data[key] = data[key].to(device)
    else:
        data = data.to(device)
    return data

tokenizer = BertTokenizer.from_pretrained(vocab_path)
model = SCMBertX4(pretrained_bert_fold, pretrained_bert_config)
model = model.to(device)
model.eval()
load_checkpoint(model, load_epoch_num)

with open("./data/id2text_seg_3_1008.json", "r") as f:
	id2text = json.load(f)

text_embed = []
with torch.no_grad():
	for i in tqdm(range(all_text_count)):
		text_id = str(i)
		text_content = id2text[text_id]
		text_input = tokenizer(text_content, padding=True, truncation=True, max_length=80, return_tensors="pt")
		text_input = put_data_to_device(text_input)
		text_rep = model.text_encode(text_input)
		text_rep_array = text_rep.cpu().numpy()
		text_embed.append(text_rep_array)

	att_para = model.att_para.data.cpu().numpy()
	print("att_para: ", att_para.shape)
	np.save('att_para.npy', att_para)

pickle.dump(text_embed, open("text_embed.p", "wb"))