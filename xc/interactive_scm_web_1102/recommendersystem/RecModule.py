import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import datetime
import random
random.seed(9001)
import numpy as np
from recommendersystem.recmodel import myrecmodel, myrecmodelx

class RecModule():
    def __init__(self, config, convhis=None):
        self.item_num = config.item_num
        self.attribute_num = config.attribute_num
        self.parent_attribute_num = config.parent_attribute_num
        self.myrecmodel = myrecmodel(config)
        self.convhis = convhis
        self.max_rec_item_num = config.max_rec_item_num

    def init_eval(self):
        self.myrecmodel.eval()

    def get_item_preference(self, input_case=None, pos_attribute=None, neg_attribute=None,candidate_list=None):
        self.init_eval()
        if input_case == None:
            input_case = self.convhis.get_input_case()
            # print("Here: ", input_case)
        if pos_attribute == None:
            pos_attribute = self.convhis.get_pos_attribute()
        if neg_attribute == None:
            neg_attribute = self.convhis.get_neg_attribute()
        if candidate_list == None:
            candidate_list = self.convhis.get_candidate_list()

        input_case = [input_case] * len(candidate_list)
        input_case = torch.tensor(input_case)
        item_list = torch.tensor(candidate_list)
        
        # print("input_case: ", input_case.size())
        # print("item_list: ", item_list.size())
        # print("pos_attribute: ", pos_attribute.size())
        # print("neg_attribute: ", neg_attribute.size())

        if len(pos_attribute) == 0:
            pos_attribute = None
        else:
            pos_attribute = [pos_attribute] * len(candidate_list)
            pos_attribute = torch.tensor(pos_attribute)

        if len(neg_attribute) == 0:
            neg_attribute = None
        else:
            neg_attribute = [neg_attribute] * len(candidate_list)
            neg_attribute = torch.tensor(neg_attribute)

        # print("input_case: ", input_case)
        return self.myrecmodel.pair_text_input(input_case, item_list, pos_attribute, neg_attribute)

    def get_recommend_item_list(self, candidate_list=None):
        if candidate_list == None:
            candidate_list = self.convhis.get_candidate_list()
        if len(candidate_list) == 0:
            return []
        item_score_list = self.get_item_preference(candidate_list=candidate_list)
        # print("item_score_list: ", item_score_list.size())
        values, indices = item_score_list.sort(descending=True)

        indices = indices.tolist()[:self.max_rec_item_num]
        item_list = []
        for i in indices:
            item_list.append(candidate_list[i])
        return item_list

    def get_text_rep(self, input_case):
        return self.myrecmodel.text_embed_encode(input_case)

    def add_new_case(self, case_content):
        self.myrecmodel.add_new_case(case_content)


class RecModuleX():
    def __init__(self, config, convhis=None):
        self.item_num = config.item_num
        self.attribute_num = config.attribute_num
        self.parent_attribute_num = config.parent_attribute_num
        self.myrecmodel = myrecmodelx(config)
        self.convhis = convhis
        self.max_rec_item_num = config.max_rec_item_num
        self.max_sen_selected = config.max_sen_selected
        self.match_sen_num = config.match_sen_num

    def init_eval(self):
        self.myrecmodel.eval()

#获取案例属性
    def get_item_preference(self, input_case=None, pos_attribute=None, neg_attribute=None,candidate_list=None):
        self.init_eval()
        if input_case == None:
            input_case = self.convhis.get_input_case()
            # print("Here: ", input_case)
        if pos_attribute == None:
            pos_attribute = self.convhis.get_pos_attribute()
        if neg_attribute == None:
            neg_attribute = self.convhis.get_neg_attribute()
        if candidate_list == None:
            candidate_list = self.convhis.get_candidate_list()

        score_list = [-float("inf")] * min(self.max_rec_item_num, len(candidate_list))
        item_id_list = [-1] * min(self.max_rec_item_num, len(candidate_list))
        rec_case_sen_id = [[]] * min(self.max_rec_item_num, len(candidate_list))
        input_case_sen_id = []

        if len(pos_attribute) == 0:
            pos_attribute = None
        else:
            # pos_attribute = [pos_attribute]
            pos_attribute = torch.tensor(pos_attribute)

        if len(neg_attribute) == 0:
            neg_attribute = None
        else:
            # neg_attribute = [neg_attribute]
            neg_attribute = torch.tensor(neg_attribute)

        with torch.no_grad():
            for item in candidate_list:
                AB_sim, att_A_softmax, att_B_softmax, F_AB = \
                    self.myrecmodel.pair_text_input(input_case, item, pos_attribute, neg_attribute)
                AB_sim = AB_sim.item()

                current_min = min(score_list)
                if AB_sim > current_min:
                    current_min_index = score_list.index(current_min)
                    score_list[current_min_index] = AB_sim
                    item_id_list[current_min_index] = item

                    _, A_index = torch.sort(att_A_softmax, descending=True)
                    A_index = A_index.tolist()
                    # _, B_index = torch.sort(att_B_softmax, descending=True)
                    # B_index = B_index.tolist()
                    _, AB_index = torch.sort(F_AB, descending=True)
                    AB_index = AB_index.tolist()

                    if len(input_case_sen_id) == 0:
                        input_case_sen_id = A_index[:self.max_sen_selected]
                    else:
                        assert input_case_sen_id == A_index[:self.max_sen_selected]
                    # rec_case_sen_id[current_min_index] = [AB_index[a_idx][:self.match_sen_num] for a_idx in A_index[:self.max_sen_selected]]
                    # print("current_min_index: ", current_min_index)
                    # print("len rec_case_sen_id: ", len(rec_case_sen_id))
                    # print("A_index[:self.max_sen_selected]: ", A_index[:self.max_sen_selected])
                    # print("AB_index: ", len(AB_index), len(AB_index[0]))
                    rec_case_sen_id[current_min_index] = [AB_index[a_idx][0] for a_idx in A_index[:self.max_sen_selected]]

        return item_id_list, input_case_sen_id, rec_case_sen_id

#获取case的list
    def get_recommend_item_list(self, candidate_list=None):
        if candidate_list == None:
            candidate_list = self.convhis.get_candidate_list()
        # if len(candidate_list) == 0:
        #     return []
        item_id_list, input_case_sen_id, rec_case_sen_id = self.get_item_preference(candidate_list=candidate_list)
        # print("item_score_list: ", item_score_list.size())
        return item_id_list, input_case_sen_id, rec_case_sen_id

#案件编码
    def get_text_rep(self, input_case):
        return self.myrecmodel.text_embed_encode(input_case)

#增加新案例
    def add_new_case(self, case_content):
        text_content_list = self.myrecmodel.add_new_case(case_content)
        return text_content_list