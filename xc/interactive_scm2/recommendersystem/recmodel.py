import json
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from recommendersystem.bertencoder import SCMBertX4

class myrecmodel(nn.Module):
    def __init__(self, config):
        super(myrecmodel, self).__init__()

        # self.gpu = config.use_gpu
        self.alpha = torch.tensor(config.alpha)
        self.hidden_dim = config.hidden_dim
        self.item_num = config.item_num
        self.attribute_num = config.attribute_num
        self.text_01_sim_mode = config.text_01_sim_mode

        self.text_embed = pickle.load(open(config.text_embed_path, "rb"))
        for idx in range(len(self.text_embed)):
            self.text_embed[idx] = torch.from_numpy(self.text_embed[idx])

        with open(config.text_01feature_path, "r") as f:
            text_01_dict = json.load(f)
        self.text_01feature = []
        for idx in range(self.item_num):
            self.text_01feature.append(text_01_dict[str(idx)])
        self.text_01feature = torch.tensor(self.text_01feature, dtype=torch.float)

        self.init_for_new(config)

        if config.load_similar_matrix:
            self.load_similar_matrix = config.load_similar_matrix
            self.similar_matrix = torch.from_numpy(np.load(config.similar_matrix_path))

    def init_for_new(self, config):
        # add empty list and zero vector for unseen case
        self.text_embed.append([])

        text_01feature_dim = self.text_01feature.size()[-1]
        new_01feature = torch.zeros(1, text_01feature_dim)
        self.text_01feature = torch.cat([self.text_01feature, new_01feature])

        # add encoder
        self.bert_encoder = SCMBertX4(config.pretrained_bert_fold, config.pretrained_bert_config, \
                                        config.vocab_path, config.max_seq_len, config.bert_embed_size)
        bert_finetuned_params = torch.load(config.finetuned_model_path)
        self.bert_encoder.load_state_dict(bert_finetuned_params["model"], strict=False)
        self.bert_encoder.eval()

        # self.regex_encoder = RegexEncoder(config.text_01feature_size, config.feature2id_file_path)

    def text_embed_encode(self, text_id):
        return self.text_embed[text_id]

    def text_embed_similarity(self, A_sen_embed, B_sen_embed):
        with torch.no_grad():
            F = torch.matmul(A_sen_embed, B_sen_embed.T)
            att_A = torch.sum(A_sen_embed * self.bert_encoder.att_para, dim=-1)
            att_B = torch.sum(B_sen_embed * self.bert_encoder.att_para, dim=-1)
            att_A_softmax = torch.softmax(att_A, dim=-1)
            att_B_softmax = torch.softmax(att_B, dim=-1)  
            AB_att = torch.ger(att_A_softmax, att_B_softmax)
            AB_sim = torch.sum(F * AB_att)
        return AB_sim, att_A_softmax, att_B_softmax, F

    def text_01_encode(self, text_id):
        return self.text_01feature[text_id]

    def text_01_similarity(self, A_01, B_01):
        if self.text_01_sim_mode == "dot":
            sim_score = torch.sum(A_01 * B_01, dim=-1)
        elif self.text_01_sim_mode == "xor":
            sim_score = - torch.sum(torch.abs(A_01 - B_01), dim=-1)
        else:
            print("text_01_similarity error!!!")
        return sim_score

    def pair_text_input(self, A_text_id, B_text_id, pos_att, neg_att):
        if self.load_similar_matrix:
            text_rep_sim = self.similar_matrix[A_text_id][B_text_id]
        else:
            A_embed = self.text_embed_encode(A_text_id)
            B_embed = self.text_embed_encode(B_text_id)
            text_rep_sim, att_A_softmax, att_B_softmax, F = self.text_embed_similarity(A_embed, B_embed)
        # print("text_rep_sim: ", text_rep_sim.size())

        A_01 = self.text_01_encode(A_text_id)
        if pos_att is not None:
            # assert len(A_01) == len(pos_att)
            # for index in range(len(A_01)):
            #     A_01[index][pos_att[index]] = 1
            A_01[pos_att] = 1
        if neg_att is not None:
            # assert len(A_01) == len(neg_att)
            # for index in range(len(A_01)):
            #     A_01[index][neg_att[index]] = 0
            A_01[neg_att] = 0

        B_01 = self.text_01_encode(B_text_id)
        text_01_sim = self.text_01_similarity(A_01, B_01)

        return text_rep_sim + self.alpha * text_01_sim