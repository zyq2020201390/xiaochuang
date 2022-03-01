import numpy as np
import json
import random
import torch
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

item_num = 1649
sim_matrix = [[0.] * item_num] * item_num

with torch.no_grad():
	for i in tqdm(range(item_num)):
		for j in range(i, item_num):
			A_embed = rec.myrecmodel.text_embed_encode(i)
			B_embed = rec.myrecmodel.text_embed_encode(j)
			text_rep_sim, att_A_softmax, att_B_softmax, F = rec.myrecmodel.text_embed_similarity(A_embed, B_embed)
			text_rep_sim = text_rep_sim.item()
			sim_matrix[i][j] = sim_matrix[j][i] = text_rep_sim

sim_matrix = np.array(sim_matrix)
np.save('./data/sim_matrix.npy', sim_matrix)