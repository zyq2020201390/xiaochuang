import json
import torch

def trans_index(origin_dict):
    new_dict = {}
    for key in origin_dict:
        new_dict[int(key)] = origin_dict[key]
    return new_dict

class DialogueManager:
    def __init__(self, config, rec, agent, user, convhis):      
        with open(config.index2attribute_path, 'r', encoding='utf-8') as f:
            self.index2attribute = json.load(f)
        self.index2attribute = trans_index(self.index2attribute)

        with open(config.index2text_path, 'r', encoding='utf8') as f:
            self.index2text = json.load(f)      
        self.index2text = trans_index(self.index2text)

        with open(config.index2textseg_path, 'r', encoding='utf8') as f:
            self.index2textseg = json.load(f)      
        self.index2textseg = trans_index(self.index2textseg)

        with open(config.attribute_tree_path, 'r') as f:
            self.attribute_tree = json.load(f)
        self.attribute_tree = trans_index(self.attribute_tree)

        self.rec_action_index = config.rec_action_index
        self.rec_success_reward = config.rec_success_reward
        self.pos_attribute_reward = config.pos_attribute_reward
        self.user_quit_reward = config.user_quit_reward
        self.every_turn_reward = config.every_turn_reward
        self.turn_limit = config.turn_limit

        self.rec = rec
        self.agent = agent
        self.user = user
        self.convhis = convhis

        self.input_case = None
        self.target_case = None
        self.silence = True
        self.turn_num = None
        self.current_agent_action = None

    def add_new_case(self, new_case_id, text_content, text_content_list):
        # assert new_case_id not in self.index2text
        # assert new_case_id not in self.index2textseg        

        duplicate_case_id = None
        for idx in self.index2text:
            if idx != new_case_id and text_content.strip() == self.index2text[idx].strip():
                duplicate_case_id = idx
                break

        self.index2text[new_case_id] = text_content
        self.index2textseg[new_case_id] = text_content_list

        return duplicate_case_id

    def output_user_attribute(self, attribute_list):
        if len(attribute_list) == 0:
            user_utt = "no"
        else:
            output_attribute_list = list(map(lambda x: self.index2attribute[x], attribute_list))
            user_utt = "contain " + ",".join(output_attribute_list)
        print("turn {} user: {}".format(str(self.turn_num), user_utt))

    def output_user_item(self, like):
        if like:
            user_utt = "find similar case: " + str(self.target_case)
        else:
            user_utt = "not find similar case"
        print("turn {} user: {}".format(str(self.turn_num), user_utt))

    def output_agent_attribute(self, action_index):
        attribute_list = self.attribute_tree[action_index]
        attribute_list = map(lambda x: self.index2attribute[x], attribute_list)
        agent_utt = "choose attribute you need: " + ','.join(attribute_list)
        print("turn {} agent: {}".format(str(self.turn_num), agent_utt))

    def output_agent_item(self, rec_item_list):
        item_list = map(lambda x: str(x) + ": " + self.index2text[x][:10]+'...', rec_item_list)
        agent_utt = "recommend cases: " + ','.join(item_list)
        print("turn {} agent: {}".format(str(self.turn_num), agent_utt))

    def output_text_content(self, text_id):
        print("text id {}:\n {}".format(str(text_id), self.index2text[text_id][:10]+"..."))

    def output_user_start(self, input_case):
        print("user start, input case")
        self.output_text_content(input_case)

    def get_current_agent_action(self):
        return self.current_agent_action

    def initialize_dialogue(self, input_case, target_case, silence):
        self.input_case = input_case
        self.target_case = target_case
        self.silence = silence
        self.turn_num = 1
        self.current_agent_action = None

        self.agent.init_episode()
        self.user.init_episode(input_case, target_case)
        self.convhis.init_conv(input_case, target_case, [], [], None)

        if not self.silence:
            self.output_user_start(self.input_case)

    def agent_turn(self, action_index):
        self.current_agent_action = action_index
        if action_index == self.rec_action_index:
            candidate_list = self.convhis.get_candidate_list()
            if self.rec is None:
                rec_item_list = candidate_list[:1]
            else:
                rec_item_list = self.rec.get_recommend_item_list(candidate_list)
            if not self.silence:
                self.output_agent_item(rec_item_list)
            return None, rec_item_list
        else:
            # ask_attribute_list = self.attribute_tree[action_index]
            if not self.silence:
                self.output_agent_attribute(action_index)
            return action_index, None


    def user_turn(self, ask_attribute_list, ask_item_list):
        self.turn_num += 1
        if ask_attribute_list != None:
            attribute_list = self.user.next_turn(ask_attribute_list)
            if not self.silence:
                self.output_user_attribute(attribute_list)
            return attribute_list
        if ask_item_list != None:
            item_list = []
            for item in ask_item_list:
                if item == self.target_case:
                    item_list.append(item)
                    break
            if not self.silence:
                self.output_user_item(len(item_list)>0)
            return item_list

    def get_current_agent_state(self, return_tensor=True):
        state_entropy = self.convhis.get_attribute_entropy().copy()
        state_convhis = self.convhis.get_convhis_vector().copy()
        state_len = self.convhis.get_length_vector().copy()
        attribute_description = self.convhis.get_attribute_description().copy()
        text_rep = self.rec.get_text_rep(self.input_case).tolist()

        dialogue_state = state_entropy + state_convhis + state_len + attribute_description
        if return_tensor:
            dialogue_state = torch.tensor(dialogue_state)
        return dialogue_state

    def next_turn(self, action_index=None):
        if action_index == None:
            action_index = self.get_agent_action_index()
        # if self.turn_num == self.turn_limit + 1:
        #     return True, self.user_quit_reward, None
        IsOver, reward, return_list = None, None, None
        action_index, ask_item_list = self.agent_turn(action_index)
        if action_index != None:
            ask_attribute_list = self.attribute_tree[action_index]
            attribute_list = self.user_turn(ask_attribute_list, ask_item_list)
            pos_attribute_set = set(attribute_list)
            neg_attribute_set = set(ask_attribute_list) - pos_attribute_set
            self.convhis.add_new_attribute(pos_attribute_set, neg_attribute_set, action_index)
            self.convhis.update_conv_his(len(pos_attribute_set)>0, action_index)
            if len(attribute_list) > 0:
                IsOver, reward, return_list = False, self.every_turn_reward + self.pos_attribute_reward, None
            else:
                IsOver, reward, return_list = False, self.every_turn_reward, None
        if ask_item_list != None:
            item_list = self.user_turn(action_index, ask_item_list)
            if len(item_list) > 0:
                IsOver, reward, return_list = True, self.rec_success_reward, ask_item_list
            else:
                self.convhis.add_conv_neg_item_list(ask_item_list)
                IsOver, reward, return_list = False, self.every_turn_reward, ask_item_list

        if self.turn_num == self.turn_limit + 1:
            if not IsOver:
                IsOver = True
                reward = self.user_quit_reward

        return IsOver, reward, return_list

    def get_agent_action_index(self):
        dialogue_state = self.get_current_agent_state()
        action_index = self.agent.choose_action(dialogue_state)
        return action_index

    def initialize_episode(self, input_case, target_case, silence=True):
        self.initialize_dialogue(input_case, target_case, silence)
        return self.get_current_agent_state()

    def step(self, action_index):
        IsOver, reward, return_list = self.next_turn(action_index)
        return IsOver, self.get_current_agent_state(), reward