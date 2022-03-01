import os

root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

class AgentRuleConfig():
    def __init__(self):
        self.rec_prob_para = 10
        self.parent_attribute_num = 17