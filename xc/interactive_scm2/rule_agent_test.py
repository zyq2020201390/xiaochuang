import numpy as np
import json
import random
from tqdm import tqdm
from convhis.ConvHis import ConvHis
from convhis.ConvHisConfig import ConvHisConfig
from agents.AgentRule import AgentRule
from agents.AgentRuleConfig import AgentRuleConfig
# from agents.AgentEAR import AgentEAR
# from agents.AgentEARConfig import AgentEARConfig
from recommendersystem.RecModule import RecModule
from recommendersystem.RecModuleConfig import RecModuleConfig
from user.UserSim import UserSim
from user.UserSimConfig import UserSimConfig
from dialoguemanager.DialogueManager import DialogueManager
from dialoguemanager.DialogueManagerConfig import DialogueManagerConfig

random.seed(2154)

ch = ConvHis(ConvHisConfig())
agent = AgentRule(AgentRuleConfig(), ch)
rec = RecModule(RecModuleConfig(), convhis=ch)
rec.init_eval()
# agent.set_rec_model(rec)
usersim = UserSim(UserSimConfig())
dm = DialogueManager(DialogueManagerConfig(), rec, agent, usersim, ch)
# agent.set_env(dm)

trainid_file_path = "./data/train_id.json"
validid_file_path = "./data/valid_id.json"
testid_file_path = "./data/test_id.json"

def conv_eva(file_path):
	success_count = 0
	turn_count = 0
	target_rank_count = 0

	with open(file_path, "r") as f:
		conv =  json.load(f)
	# conv = conv[:1]
	for A, B, C in tqdm(conv):
	# for A, B, C in conv:
		dm.initialize_dialogue(A, B, True)
		IsOver = False
		while not IsOver:
			IsOver, reward, return_list = dm.next_turn()
		success = 0
		target_rank = None
		if reward > 0:
			success = 1
			target_rank = return_list.index(B)
		turn_num = dm.turn_num - 1
		# print(f"success: {success}, turn_num: {turn_num}, target_rank: {target_rank}")
		success_count += success
		turn_count += turn_num
		if success:
			target_rank_count += target_rank

	print(f"success rate: {success_count}/{len(conv)}={success_count/len(conv)}, mean turn: {turn_count/len(conv)}, mean target rank: {target_rank_count/len(conv)}")

print("-----train------------")
conv_eva(trainid_file_path)
print("-----valid------------")
conv_eva(validid_file_path)
print("-----test------------")
conv_eva(testid_file_path)