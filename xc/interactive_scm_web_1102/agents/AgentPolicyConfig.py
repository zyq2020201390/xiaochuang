import os

root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

class AgentEARConfig():
    def __init__(self):
    	self.use_gpu = 
    	self.input_dim = 
    	self.hidden_dim = 
    	self.output_dim = 
    	self.dp = 0.2
    	self.DPN_model_path = root_path 
    	self.DPN_model_name = "TwoLayer"
    	self.PG_discount_rate = 