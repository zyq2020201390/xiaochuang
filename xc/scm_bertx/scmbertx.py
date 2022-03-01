import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.parameter import Parameter
from transformers import BertTokenizer, BertModel, BertConfig
import math

class SCMBertX(nn.Module):
    def __init__(self, pretrained_bert_fold, pretrained_bert_config, hidden_size=768, hidden_size2=256):
        super(SCMBertX, self).__init__()
        self.bert_model = BertModel.from_pretrained(pretrained_bert_fold, config=pretrained_bert_config)
        self.weight_matrix = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.weight_para = Parameter(torch.Tensor(hidden_size2, hidden_size))
        self.att_para = Parameter(torch.Tensor(hidden_size2))
        self.reset_para()

    def reset_para(self):
        init.kaiming_uniform_(self.weight_matrix, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_para, a=math.sqrt(5))

        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_para)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.att_para, -bound, bound)

    def text_encode(self, sen_list_input):
        _, sen_embed = self.bert_model(**sen_list_input)
        return sen_embed

    def text_similarity(self, A_sen_embed, B_sen_embed):
        # F = torch.tanh(torch.matmul(torch.matmul(A_sen_embed, self.weight_matrix), B_sen_embed.T))
        # A_sen_para = torch.matmul(A_sen_embed, self.weight_para.T)
        # B_sen_para = torch.matmul(B_sen_embed, self.weight_para.T)
        # H_A = torch.tanh(A_sen_para + torch.matmul(F, B_sen_para))
        # H_B = torch.tanh(B_sen_para + torch.matmul(F.T, A_sen_para))
        # att_A = torch.sum(H_A * self.att_para, dim=-1)
        # att_B = torch.sum(H_B * self.att_para, dim=-1)
        # att_A_softmax = torch.softmax(att_A, dim=-1)
        # att_B_softmax = torch.softmax(att_B, dim=-1)
        # A_embed = torch.matmul(att_A_softmax, A_sen_embed)
        # B_embed = torch.matmul(att_B_softmax, B_sen_embed)
        # AB_sim = torch.sum(A_embed * B_embed, dim=-1)
        # return AB_sim, att_A_softmax, att_B_softmax, F
        ### scm_bertx-1

        # F = torch.tanh(torch.matmul(torch.matmul(A_sen_embed, self.weight_matrix), B_sen_embed.T)) ### scm_bertx-2
        F = torch.matmul(torch.matmul(A_sen_embed, self.weight_matrix), B_sen_embed.T) ### scm_bertx-3
        att_A, _ = torch.max(F, dim=1)
        att_B, _ = torch.max(F ,dim=0)
        att_A_softmax = torch.softmax(att_A, dim=-1)
        att_B_softmax = torch.softmax(att_B, dim=-1)
        A_embed = torch.matmul(att_A_softmax, A_sen_embed)
        B_embed = torch.matmul(att_B_softmax, B_sen_embed)
        AB_sim = torch.sum(A_embed * B_embed, dim=-1)
        return AB_sim, att_A_softmax, att_B_softmax, F


    def pair_text_input(self, A_text, B_text, return_att_weight=False):
        A_embed = self.text_encode(A_text)
        B_embed = self.text_encode(B_text)
        AB_sim, att_A_softmax, att_B_softmax, F = self.text_similarity(A_embed, B_embed)
        if return_att_weight:
            return AB_sim, att_A_softmax, att_B_softmax, F
        else:
            return AB_sim

    def triple_text_input(self, A_text, B_text, C_text, return_att_weight=False):
        A_embed = self.text_encode(A_text)
        B_embed = self.text_encode(B_text)
        C_embed = self.text_encode(C_text)
        AB_sim, att_A_softmax_1, att_B_softmax, F_AB = self.text_similarity(A_embed, B_embed)
        AC_sim, att_A_softmax_2, att_C_softmax, F_AC = self.text_similarity(A_embed, C_embed)
        if return_att_weight:
            return torch.stack([AB_sim, AC_sim], dim=-1), att_A_softmax_1, att_B_softmax, F_AB, att_A_softmax_2, att_C_softmax, F_AC
        else:
            return torch.stack([AB_sim, AC_sim], dim=-1)


class SCMBertX2(nn.Module):
    def __init__(self, pretrained_bert_fold, pretrained_bert_config, hidden_size=768):
        super(SCMBertX2, self).__init__()
        self.bert_model = BertModel.from_pretrained(pretrained_bert_fold, config=pretrained_bert_config)
        self.two_layer = nn.Sequential(nn.Linear(4*hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))
        self.hidden_size = hidden_size
        self.reset_para()


    def text_encode(self, sen_list_input):
        _, sen_embed = self.bert_model(**sen_list_input)
        return sen_embed

    def text_similarity(self, A_sen_embed, B_sen_embed):
        F = torch.matmul(A_sen_embed, B_sen_embed.T)
        att_A, _ = torch.max(F, dim=1)
        att_B, _ = torch.max(F, dim=0)
        att_A_softmax = torch.softmax(att_A, dim=-1)
        att_B_softmax = torch.softmax(att_B, dim=-1)
        A_embed = torch.matmul(att_A_softmax, A_sen_embed)
        B_embed = torch.matmul(att_B_softmax, B_sen_embed)
        pair_input = torch.cat([A_embed, B_embed, A_embed - B_embed, A_embed * B_embed])
        AB_sim = self.two_layer(pair_input)
        AB_sim = AB_sim.squeeze(-1)
        return AB_sim, att_A_softmax, att_B_softmax, F


    def pair_text_input(self, A_text, B_text, return_att_weight=False):
        A_embed = self.text_encode(A_text)
        B_embed = self.text_encode(B_text)
        AB_sim, att_A_softmax, att_B_softmax, F = self.text_similarity(A_embed, B_embed)
        if return_att_weight:
            return AB_sim, att_A_softmax, att_B_softmax, F
        else:
            return AB_sim

    def triple_text_input(self, A_text, B_text, C_text, return_att_weight=False):
        A_embed = self.text_encode(A_text)
        B_embed = self.text_encode(B_text)
        C_embed = self.text_encode(C_text)
        AB_sim, att_A_softmax_1, att_B_softmax, F_AB = self.text_similarity(A_embed, B_embed)
        AC_sim, att_A_softmax_2, att_C_softmax, F_AC = self.text_similarity(A_embed, C_embed)
        if return_att_weight:
            return torch.stack([AB_sim, AC_sim], dim=-1), att_A_softmax_1, att_B_softmax, F_AB, att_A_softmax_2, att_C_softmax, F_AC
        else:
            return torch.stack([AB_sim, AC_sim], dim=-1)


class SCMBertX3(nn.Module):
    def __init__(self, pretrained_bert_fold, pretrained_bert_config, hidden_size=768):
        super(SCMBertX3, self).__init__()
        self.bert_model = BertModel.from_pretrained(pretrained_bert_fold, config=pretrained_bert_config)
        self.att_para = Parameter(torch.Tensor(hidden_size))
        self.hidden_size = hidden_size
        self.scale = math.sqrt(hidden_size)
        self.reset_para()

    def reset_para(self):
        fan_in = self.hidden_size
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.att_para, -bound, bound)

    def text_encode(self, sen_list_input):
        _, sen_embed = self.bert_model(**sen_list_input)
        return sen_embed

    def text_similarity(self, A_sen_embed, B_sen_embed):
        F = torch.matmul(A_sen_embed, B_sen_embed.T)
        # F = torch.matmul(A_sen_embed, B_sen_embed.T) / self.scale
        score_A, match_id_in_B = torch.max(F, dim=1)
        score_B, match_id_in_A = torch.max(F, dim=0)
        att_A = torch.sum(A_sen_embed * self.att_para, dim=-1)
        att_B = torch.sum(B_sen_embed * self.att_para, dim=-1)
        att_A_softmax = torch.softmax(att_A, dim=-1)
        att_B_softmax = torch.softmax(att_B, dim=-1)  

        # A_sim = torch.sum(score_A * att_A_softmax, dim=-1) # SCMBertX3-1
        # B_sim = torch.sum(score_B * att_B_softmax, dim=-1) # SCMBertX3-1

        # A_sim = torch.sum(score_A * att_A_softmax * att_B_softmax[match_id_in_B], dim=-1) # SCMBertX3-2
        # B_sim = torch.sum(score_B * att_B_softmax * att_A_softmax[match_id_in_A], dim=-1) # SCMBertX3-2

        # AB_sim = A_sim + B_sim

        # SCMBertX3-3
        AB_att = torch.ger(att_A_softmax, att_B_softmax)
        AB_sim = torch.sum(F * AB_att)

        return AB_sim, att_A_softmax, att_B_softmax, F


    def pair_text_input(self, A_text, B_text, return_att_weight=False):
        A_embed = self.text_encode(A_text)
        B_embed = self.text_encode(B_text)
        AB_sim, att_A_softmax, att_B_softmax, F = self.text_similarity(A_embed, B_embed)
        if return_att_weight:
            return AB_sim, att_A_softmax, att_B_softmax, F
        else:
            return AB_sim

    def triple_text_input(self, A_text, B_text, C_text, return_att_weight=False):
        A_embed = self.text_encode(A_text)
        B_embed = self.text_encode(B_text)
        C_embed = self.text_encode(C_text)
        AB_sim, att_A_softmax_1, att_B_softmax, F_AB = self.text_similarity(A_embed, B_embed)
        AC_sim, att_A_softmax_2, att_C_softmax, F_AC = self.text_similarity(A_embed, C_embed)
        if return_att_weight:
            return torch.stack([AB_sim, AC_sim], dim=-1), att_A_softmax_1, att_B_softmax, F_AB, att_A_softmax_2, att_C_softmax, F_AC
        else:
            return torch.stack([AB_sim, AC_sim], dim=-1)


class SCMBertX_regex(nn.Module):
    def __init__(self, pretrained_bert_fold, pretrained_bert_config, hidden_size=768, regex_hidden_size=17):
        super(SCMBertX_regex, self).__init__()
        self.bert_model = BertModel.from_pretrained(pretrained_bert_fold, config=pretrained_bert_config)
        self.hidden_size = hidden_size
        self.regex_hidden_size = regex_hidden_size
        self.all_size = hidden_size + regex_hidden_size
        self.att_para = Parameter(torch.Tensor(self.all_size))
        self.reset_para()

    def reset_para(self):
        fan_in = self.all_size
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.att_para, -bound, bound)

    def text_encode(self, sen_list_input, sen_regex_feature):
        _, sen_embed = self.bert_model(**sen_list_input)
        sen_embed = torch.cat([sen_embed, sen_regex_feature], dim=-1)
        return sen_embed

    def text_similarity(self, A_sen_embed, B_sen_embed):
        F = torch.matmul(A_sen_embed, B_sen_embed.T)
        score_A, match_id_in_B = torch.max(F, dim=1)
        score_B, match_id_in_A = torch.max(F, dim=0)
        att_A = torch.sum(A_sen_embed * self.att_para, dim=-1)
        att_B = torch.sum(B_sen_embed * self.att_para, dim=-1)
        att_A_softmax = torch.softmax(att_A, dim=-1)
        att_B_softmax = torch.softmax(att_B, dim=-1)  

        # A_sim = torch.sum(score_A * att_A_softmax, dim=-1) # SCMBertX3-1
        # B_sim = torch.sum(score_B * att_B_softmax, dim=-1) # SCMBertX3-1

        # A_sim = torch.sum(score_A * att_A_softmax * att_B_softmax[match_id_in_B], dim=-1) # SCMBertX3-2
        # B_sim = torch.sum(score_B * att_B_softmax * att_A_softmax[match_id_in_A], dim=-1) # SCMBertX3-2

        # AB_sim = A_sim + B_sim

        # SCMBertX3-3
        AB_att = torch.ger(att_A_softmax, att_B_softmax)
        AB_sim = torch.sum(F * AB_att)

        return AB_sim, att_A_softmax, att_B_softmax, F


    def pair_text_input(self, A_text, B_text, A_regex, B_regex, return_att_weight=False):
        A_embed = self.text_encode(A_text, A_regex)
        B_embed = self.text_encode(B_text, B_regex)
        AB_sim, att_A_softmax, att_B_softmax, F = self.text_similarity(A_embed, B_embed)
        if return_att_weight:
            return AB_sim, att_A_softmax, att_B_softmax, F
        else:
            return AB_sim

    def triple_text_input(self, A_text, B_text, C_text, A_regex, B_regex, C_regex, return_att_weight=False):
        A_embed = self.text_encode(A_text, A_regex)
        B_embed = self.text_encode(B_text, B_regex)
        C_embed = self.text_encode(C_text, C_regex)
        AB_sim, att_A_softmax_1, att_B_softmax, F_AB = self.text_similarity(A_embed, B_embed)
        AC_sim, att_A_softmax_2, att_C_softmax, F_AC = self.text_similarity(A_embed, C_embed)
        if return_att_weight:
            return torch.stack([AB_sim, AC_sim], dim=-1), att_A_softmax_1, att_B_softmax, F_AB, att_A_softmax_2, att_C_softmax, F_AC
        else:
            return torch.stack([AB_sim, AC_sim], dim=-1)


class SCMBertX4(nn.Module):
    def __init__(self, pretrained_bert_fold, pretrained_bert_config, hidden_size=768):
        super(SCMBertX4, self).__init__()
        self.bert_model = BertModel.from_pretrained(pretrained_bert_fold, config=pretrained_bert_config)#加载模型
        self.att_para = Parameter(torch.Tensor(hidden_size))#可训练参数
        self.hidden_size = hidden_size
        # self.scale = math.sqrt(hidden_size)
        self.reset_para()#参数初始化

    #权重初始化
    def reset_para(self):
        fan_in = self.hidden_size #第i层神经元个数
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.att_para, -bound, bound)#初始化

    ''' #文本编码
    def text_encode(self, sen_list_input):
        #print(sen_list_input)
        _, sen_embed = self.bert_model(**sen_list_input)#字典转化后调用
        #print("_:",_) #模型最后一层输出的隐藏状态
        print("sen_embed:",sen_embed) #序列的第一个token(classification token)的最后一层的隐藏状态，它是由线性层和Tanh激活函数进一步处理的。（通常用于句子分类，至于是使用这个表示，还是使用整个输入序列的隐藏状态序列的平均化或池化，视情况而定）
        return sen_embed'''

    def text_encode(self, sen_list_input):
        #print(sen_list_input)
        x = self.bert_model(**sen_list_input)#字典转化后调用

        #print("_:",_) #模型最后一层输出的隐藏状态
        print("x.pooler_output.shape:",x.pooler_output.shape) #序列的第一个token(classification token)的最后一层的隐藏状态，它是由线性层和Tanh激活函数进一步处理的。（通常用于句子分类，至于是使用这个表示，还是使用整个输入序列的隐藏状态序列的平均化或池化，视情况而定）
        return x.pooler_output

    #ppt上相似度计算公式
    def text_similarity(self, A_sen_embed, B_sen_embed):
        F = torch.matmul(A_sen_embed, B_sen_embed.T)
        # score_A, match_id_in_B = torch.max(F, dim=1)
        # score_B, match_id_in_A = torch.max(F, dim=0)
        att_A = torch.sum(A_sen_embed * self.att_para, dim=-1)
        att_B = torch.sum(B_sen_embed * self.att_para, dim=-1)
        att_A_softmax = torch.softmax(att_A, dim=-1)
        att_B_softmax = torch.softmax(att_B, dim=-1)  
        AB_att = torch.ger(att_A_softmax, att_B_softmax)
        AB_sim = torch.sum(F * AB_att)
        return AB_sim, att_A_softmax, att_B_softmax, F

    def pair_text_input(self, A_text, B_text, return_att_weight=False):
        A_embed = self.text_encode(A_text)
        B_embed = self.text_encode(B_text)
        AB_sim, att_A_softmax, att_B_softmax, F = self.text_similarity(A_embed, B_embed)
        if return_att_weight:
            return AB_sim, att_A_softmax, att_B_softmax, F
        else:
            return AB_sim

    #ABC返回AB，AC相似度
    def triple_text_input(self, A_text, B_text, C_text, return_att_weight=False):
        A_embed = self.text_encode(A_text)
        B_embed = self.text_encode(B_text)
        C_embed = self.text_encode(C_text)
        AB_sim, att_A_softmax_1, att_B_softmax, F_AB = self.text_similarity(A_embed, B_embed)
        AC_sim, att_A_softmax_2, att_C_softmax, F_AC = self.text_similarity(A_embed, C_embed)
        if return_att_weight:
            return torch.stack([AB_sim, AC_sim], dim=-1), att_A_softmax_1, att_B_softmax, F_AB, att_A_softmax_2, att_C_softmax, F_AC
        else:
            return torch.stack([AB_sim, AC_sim], dim=-1)