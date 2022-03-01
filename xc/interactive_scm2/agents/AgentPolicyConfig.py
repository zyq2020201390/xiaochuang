import os

root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

class AgentPolicyConfig():
    def __init__(self):
        self.input_dim = 17 + 10 + 8 + 24
        self.hidden_dim = 32
        self.output_dim = 17 + 1
        self.dp = 0.2
        self.DPN_model_path = root_path  + "/agents/agent_model"
        self.DPN_model_name = "TwoLayer"
        self.PG_discount_rate = 0.99