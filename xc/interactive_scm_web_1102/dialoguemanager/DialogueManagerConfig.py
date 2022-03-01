import os

root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

class DialogueManagerConfig():
    def __init__(self):
        self.index2attribute_path = root_path + "/data/zh_index2attribute.json"
        self.index2text_path = root_path + "/data/id2text.json"
        self.index2textseg_path = root_path + "/data/id2text_seg_3_1008.json"
        self.attribute_tree_path = root_path + "/data/attribute_tree_dict.json"
        self.rec_action_index = 17
        self.rec_success_reward = 3
        self.pos_attribute_reward = 0. 
        self.user_quit_reward = -1.
        self.every_turn_reward = -0.1
        self.turn_limit = 20