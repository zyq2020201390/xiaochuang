import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.parameter import Parameter
from transformers import BertTokenizer, BertModel, BertConfig
import math
import re

class BertEncoder(nn.Module):
    def __init__(self, pretrained_bert_fold, pretrained_bert_config, vocab_path, max_seq_len, hidden_size):
        super(BertEncoder, self).__init__()
        self.bert_model = BertModel.from_pretrained(pretrained_bert_fold, config=pretrained_bert_config)
        self.tokenizer = BertTokenizer.from_pretrained(vocab_path)
        self.max_seq_len = max_seq_len

        self.bi = nn.Bilinear(hidden_size, hidden_size, 1)

    def text_encode(self, text_content):
        text_input = self.tokenizer(text_content, truncation=True, max_length=self.max_seq_len, return_tensors="pt")
        _, text_embed = self.bert_model(**text_input)
        return text_embed   


class SCMBertX4(nn.Module):
    def __init__(self, pretrained_bert_fold, pretrained_bert_config, vocab_path, max_seq_len=80, hidden_size=768):
        super(SCMBertX4, self).__init__()
        self.bert_model = BertModel.from_pretrained(pretrained_bert_fold, config=pretrained_bert_config)
        self.att_para = Parameter(torch.Tensor(hidden_size))
        self.hidden_size = hidden_size
        # self.scale = math.sqrt(hidden_size)
        self.reset_para()

        self.tokenizer = BertTokenizer.from_pretrained(vocab_path)
        self.max_seq_len = max_seq_len

    def reset_para(self):
        fan_in = self.hidden_size
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.att_para, -bound, bound)

    def text_encode(self, text):
        text1, text2 = text.split(f'\n\n')
        sen_list1 = re.split(r'[，。（）；“”,();"\n]+', text1)
        sen_list2 = re.split(r'[，。：（）；“”,():;"\n]+', text2)
        text_content_list = sen_list1 + sen_list2

        with torch.no_grad():
            text_input = self.tokenizer(text_content_list, padding=True, truncation=True, max_length=self.max_seq_len, return_tensors="pt")
            _, text_embed = self.bert_model(**text_input)
        return text_embed, text_content_list

    # def text_similarity(self, A_sen_embed, B_sen_embed):
    #     F = torch.matmul(A_sen_embed, B_sen_embed.T)
    #     # score_A, match_id_in_B = torch.max(F, dim=1)
    #     # score_B, match_id_in_A = torch.max(F, dim=0)
    #     att_A = torch.sum(A_sen_embed * self.att_para, dim=-1)
    #     att_B = torch.sum(B_sen_embed * self.att_para, dim=-1)
    #     att_A_softmax = torch.softmax(att_A, dim=-1)
    #     att_B_softmax = torch.softmax(att_B, dim=-1)  
    #     AB_att = torch.ger(att_A_softmax, att_B_softmax)
    #     AB_sim = torch.sum(F * AB_att)
    #     return AB_sim, att_A_softmax, att_B_softmax, F