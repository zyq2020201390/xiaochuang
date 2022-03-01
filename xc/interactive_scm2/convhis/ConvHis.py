import json
import random
from math import log, e
from collections import defaultdict
import numpy as np
from scipy.stats import entropy

class ConvHis():
    def __init__(self, config):
        self.item_num = config.item_num
        self.attribute_num = config.attribute_num
        self.parent_attribute_num = config.parent_attribute_num
        self.att_pos_state = float(config.att_pos_state)
        self.att_neg_state = float(config.att_neg_state)
        self.item_neg_state = float(config.item_neg_state)
        self.init_state = float(config.init_state)
        self.max_conv_length = config.max_conv_length

        with open(config.attribute_tree_path, 'r') as f:
            self.attribute_tree = json.load(f)
        new_attribute_tree = {}
        for parent in self.attribute_tree:
            new_attribute_tree[int(parent)] = set(self.attribute_tree[parent])
        self.attribute_tree = new_attribute_tree     

        with open(config.item_info_path, 'r') as f:
            self.item_info = json.load(f)
        new_item_info = {}
        for item in self.item_info:
            new_item_info[int(item)] = set(self.item_info[item])
        self.item_info = new_item_info

        self.input_case = None
        self.target_case = None
        self.candidate_list = None
        self.pos_attribute = None
        self.neg_attribute = None
        self.attribute_description = None
        # self.negative_item_list1 = None
        # self.negative_item_list2 = None
        self.target_attribute = None
        self.not_target_attribute = None
        self.conv_neg_item_list = None
        self.convhis_vector = None
        self.conv_lenth = None
        self.asked_list = None

    def init_conv(self, input_case, target_case, init_pos_attribute_list, init_neg_attribute_list, init_parent_attribute):
        self.input_case = input_case
        self.target_case = target_case
        self.attribute_description = [0.] * self.attribute_num
        for _ in self.item_info[input_case]:
            self.attribute_description[_] = 1.
        self.candidate_list = []

        for _ in init_pos_attribute_list:
            self.attribute_description[_] = 1.

        for _ in init_neg_attribute_list:
            self.attribute_description[_] = 0.

        init_pos_attribute_set = set(init_pos_attribute_list)
        init_neg_attribute_set = set(init_neg_attribute_list)
        all_item_set = set([_ for _ in range(self.item_num)])
        candidate_set = set()
        for i in all_item_set:
            if len(init_pos_attribute_set - self.item_info[i]) == 0 and \
                len(init_neg_attribute_set & self.item_info[i]) == 0:
                candidate_set.add(i)
        self.candidate_list = list(candidate_set)

        all_att_set = set([_ for _ in range(self.attribute_num)])
        target_attribute_set = self.item_info[self.target_case]
        self.not_target_attribute = list(all_att_set - target_attribute_set)

        self.pos_attribute = init_pos_attribute_set
        self.neg_attribute = init_neg_attribute_set
        self.target_attribute = list(self.item_info[target_case])
        self.conv_neg_item_list = set()
        self.convhis_list = [self.init_state] * self.max_conv_length
        # self.convhis_list[0] = self.att_pos_state
        self.conv_lenth = 1
        self.asked_list = []
        if init_parent_attribute != None:
            self.asked_list = [init_parent_attribute]

    def add_new_attribute(self, pos_attribute_set, neg_attribute_set, parent_attribute):
        self.pos_attribute = self.pos_attribute.union(pos_attribute_set)
        self.neg_attribute = self.neg_attribute.union(neg_attribute_set)

        for _ in pos_attribute_set:
            self.attribute_description[_] = 1.
        for _ in neg_attribute_set:
            self.attribute_description[_] = 0.

        new_candidate_set = set()
        for item in self.candidate_list:
            if (self.attribute_tree[parent_attribute] & self.item_info[item]) == pos_attribute_set:
            # if pos_attribute_set.issubset(self.item_info[item]):
                new_candidate_set.add(item)
        self.candidate_list = list(new_candidate_set)

    def update_conv_his(self, pos, parent_attribute):
        if self.conv_lenth == self.max_conv_length:
            return 
        if pos:
            self.convhis_list[self.conv_lenth] = self.att_pos_state 
        else:
            self.convhis_list[self.conv_lenth] = self.att_neg_state
        self.conv_lenth += 1
        self.asked_list.append(parent_attribute)

    def add_conv_neg_item_list(self, neg_item_list):
        if self.conv_lenth == self.max_conv_length:
            return 
        for item in neg_item_list:
            self.conv_neg_item_list.add(item)
        
        neg_item_set = set(neg_item_list)
        new_candidate_list = set(self.candidate_list) - neg_item_set
        self.candidate_list = list(new_candidate_list)
        
        self.convhis_list[self.conv_lenth] = self.item_neg_state
        self.conv_lenth += 1

    def get_attribute_entropy(self):
        attribute_count = defaultdict(int)
        for item in self.candidate_list:
            for att in self.item_info[item]:
                attribute_count[att] += 1

        parent_attribute_entropy_list = []
        for i in range(self.parent_attribute_num):
            attribute_entropy_list = [attribute_count[_] for _ in self.attribute_tree[i]]
            if len(attribute_entropy_list) == 1:
                attribute_entropy_list.append(len(self.candidate_list)-attribute_entropy_list[0])
            parent_attribute_entropy_list.append(entropy(attribute_entropy_list))
        return parent_attribute_entropy_list

    def get_convhis_vector(self):
        return self.convhis_list

    def get_length_vector(self):
        length_vector = [0.] * 8
        if len(self.candidate_list) <= 5:
            length_vector[0] = 1.
        if len(self.candidate_list) > 5 and len(self.candidate_list) <= 15:
            length_vector[1] = 1.
        if len(self.candidate_list) > 15 and len(self.candidate_list) <= 30:
            length_vector[2] = 1.
        if len(self.candidate_list) > 30 and len(self.candidate_list) <= 50:
            length_vector[3] = 1.
        if len(self.candidate_list) > 50 and len(self.candidate_list) <= 100:
            length_vector[4] = 1.
        if len(self.candidate_list) > 100 and len(self.candidate_list) <= 300:
            length_vector[5] = 1.
        if len(self.candidate_list) > 300 and len(self.candidate_list) <= 1000:
            length_vector[6] = 1.
        if len(self.candidate_list) > 1000:
            length_vector[7] = 1.
        return length_vector

    def get_target_item(self):
        return self.target_case

    def get_pos_attribute(self):
        return list(self.pos_attribute)

    def get_neg_attribute(self):
        return list(self.neg_attribute)

    def get_target_attribute(self):
        return self.item_info[self.target_case]

    # def get_not_target_attribute(self, neg_sample_num=None):
    #     if neg_sample_num == None or len(self.not_target_attribute) <= neg_sample_num:
    #         return self.not_target_attribute
    #     else:
    #         return random.sample(set(self.not_target_attribute), neg_sample_num)

    def get_conv_neg_item_list(self):
        return list(self.conv_neg_item_list)

    def get_conv_length(self):
        return self.conv_lenth

    def get_candidate_list_len(self):
        return len(self.candidate_list)

    def get_candidate_list(self):
        #print("self.candidate_list:", self.candidate_list)
        return self.candidate_list

    def get_asked_list(self):
        return self.asked_list

    def get_attribute_description(self):
        return self.attribute_description

    def get_input_case(self):
        return self.input_case