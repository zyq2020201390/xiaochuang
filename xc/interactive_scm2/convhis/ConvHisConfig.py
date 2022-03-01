import os

root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

class ConvHisConfig():
    def __init__(self):
        self.item_num = 1649
        self.attribute_num = 24
        self.parent_attribute_num = 17
        self.att_pos_state = 1.
        self.att_neg_state = -1.
        self.item_neg_state = -2.
        self.init_state = 0
        self.max_conv_length = 10
        self.item_info_path = root_path + "/data/item_info.json"
        self.attribute_tree_path = root_path + "/data/attribute_tree_dict.json"