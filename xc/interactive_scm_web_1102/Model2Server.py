import numpy as np
import json
import random
from tqdm import tqdm
from convhis.ConvHis import ConvHis #
from convhis.ConvHisConfig import ConvHisConfig #
from agents.AgentRule import AgentRule #
from agents.AgentRuleConfig import AgentRuleConfig #
# from agents.AgentEAR import AgentEAR
# from agents.AgentEARConfig import AgentEARConfig
from recommendersystem.RecModule import RecModule, RecModuleX #
from recommendersystem.RecModuleConfig import RecModuleConfig #
from recommendersystem.textmarker import TextMarker #
from user.UserSim import UserSim #
from user.UserSimConfig import UserSimConfig #
from dialoguemanager.DialogueManager import DialogueManager#
from dialoguemanager.DialogueManagerConfig import DialogueManagerConfig

class Model2Server(object):
	def __init__(self) -> object:
		self.ch = ConvHis(ConvHisConfig())
		self.agent = AgentRule(AgentRuleConfig(), self.ch)
		# self.rec = RecModule(RecModuleConfig(), convhis=self.ch)
		self.rec = RecModuleX(RecModuleConfig(), convhis=self.ch)
		self.rec.init_eval()
		self.rec_item_num = 5
		self.rec.max_rec_item_num = self.rec_item_num
		# agent.set_rec_model(rec)
		self.usersim = UserSim(UserSimConfig())
		self.dm = DialogueManager(DialogueManagerConfig(), self.rec, self.agent, self.usersim, self.ch)
		self.marker = TextMarker(RecModuleConfig().text_01feature_size, RecModuleConfig().feature2id_file_path)

	#根据conv_id返回二级字典
	def get_action_from_conv_his(self, conv_his_dict, return_action, return_rec_items):
		print("conv_his_dict: ", conv_his_dict)
		input_case = conv_his_dict['input_case']
		pos_attribute_list = conv_his_dict['pos_attribute_list']
		neg_attribute_list = conv_his_dict['neg_attribute_list']
		parent_attribute_list = conv_his_dict['parent_attribute_list']
		rejected_item_list = conv_his_dict['rejected_item_list']
		assert len(pos_attribute_list) == len(neg_attribute_list)
		assert len(pos_attribute_list) == len(parent_attribute_list)

		self.dm.initialize_dialogue(input_case, 0, True)

		# self.ch.init_conv(input_case, 0, [], [], None)
		# self.agenft.init_episode()
		return_dict = {}

		for pos_att, neg_att, parent_att in zip(pos_attribute_list, neg_attribute_list, parent_attribute_list):
			self.ch.add_new_attribute(set(pos_att), set(neg_att), parent_att)
		self.ch.add_conv_neg_item_list(rejected_item_list)
		current_pos_att_list = self.ch.get_pos_attribute()
		
		if return_action:
			current_agent_state = self.dm.get_current_agent_state()
			action_index = self.agent.choose_action(current_agent_state)
			if action_index == self.dm.rec_action_index:
				return_dict['action_type'] = 'rec'
			else:
				return_dict['action_type'] = 'ask'
				asked_att_id_list = self.dm.attribute_tree[action_index]
				asled_att_name_list = [self.dm.index2attribute[_] for _ in asked_att_id_list]
				return_dict['asked_parent_att'] = action_index
				return_dict['asked_att_id_list'] = asked_att_id_list
				return_dict['asked_att_name_list'] = asled_att_name_list

		if return_rec_items:
			# candidate_list = self.ch.get_candidate_list()
			# rec_item_id_list = self.rec.get_recommend_item_list(candidate_list)
			# rec_item_text_list = [self.dm.index2text[_] for _ in rec_item_id_list]
			# rec_item_text_mark_list = [self.marker.get_marked_sen(_, current_pos_att_list) for _ in rec_item_text_list]
			# if len(rec_item_id_list) < self.rec_item_num:
			# 	rec_item_id_list = rec_item_id_list + ['null'] * (self.rec_item_num - len(rec_item_id_list))
			# 	rec_item_text_list = rec_item_text_list + ['null'] * (self.rec_item_num - len(rec_item_text_list))
			# 	rec_item_text_mark_list = rec_item_text_mark_list + ['null'] * (self.rec_item_num - len(rec_item_text_mark_list))
			# assert len(rec_item_id_list) == len(rec_item_text_list)
			# assert len(rec_item_id_list) == len(rec_item_text_mark_list)
			# return_dict['rec_item_id_list'] = rec_item_id_list
			# return_dict['rec_item_text_list'] = rec_item_text_list
			# return_dict['rec_item_text_mark_list'] = rec_item_text_mark_list

			candidate_list = self.ch.get_candidate_list()
			rec_item_id_list, input_case_sen_id, rec_case_sen_id = self.rec.get_recommend_item_list(candidate_list)
			rec_item_text_list = [self.dm.index2text[_] for _ in rec_item_id_list]
			rec_item_text_mark_list = [self.marker.get_marked_sen(self.dm.index2textseg[_], current_pos_att_list) for _ in rec_item_id_list]
			input_item_text_select_list = [self.dm.index2textseg[input_case][_] for _ in input_case_sen_id]
			rec_item_text_select_list = [list(set([self.dm.index2textseg[idx][sen_id] for sen_id in sen_id_list])) \
												for idx, sen_id_list in zip(rec_item_id_list, rec_case_sen_id)]

			if len(rec_item_id_list) < self.rec_item_num:
				rec_item_id_list = rec_item_id_list + ['null'] * (self.rec_item_num - len(rec_item_id_list))
				rec_item_text_list = rec_item_text_list + ['null'] * (self.rec_item_num - len(rec_item_text_list))
				rec_item_text_mark_list = rec_item_text_mark_list + ['null'] * (self.rec_item_num - len(rec_item_text_mark_list))
				input_item_text_select_list = input_item_text_select_list + ['null'] * (self.rec_item_num - len(input_item_text_select_list))
				rec_item_text_select_list = rec_item_text_select_list + ['null'] * (self.rec_item_num - len(rec_item_text_select_list))

			assert len(rec_item_id_list) == len(rec_item_text_list)
			assert len(rec_item_id_list) == len(rec_item_text_mark_list)
			# assert len(rec_item_id_list) == len(input_item_text_select_list)
			assert len(rec_item_id_list) == len(rec_item_text_select_list)

			return_dict['rec_item_id_list'] = rec_item_id_list
			return_dict['rec_item_text_list'] = rec_item_text_list
			return_dict['rec_item_text_mark_list'] = rec_item_text_mark_list
			return_dict['input_item_text_select_list'] = input_item_text_select_list
			return_dict['rec_item_text_select_list'] = rec_item_text_select_list

			# print("input_item_text_select_list: ", input_item_text_select_list)
			# print("rec_item_text_select_list: ", rec_item_text_select_list)

		return return_dict

	def add_new_case(self, input_content):
		# self.rec.add_new_case(input_content)
		text_content_list = self.rec.add_new_case(input_content)
		duplicate_case_id = self.dm.add_new_case(RecModuleConfig().item_num, input_content, text_content_list)
		return duplicate_case_id