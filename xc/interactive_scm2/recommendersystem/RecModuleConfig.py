import os

root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

class RecModuleConfig():
    def __init__(self):
        self.text_embed_path = root_path + "/data/text_embed.p"
        self.bilinear_weight_path = root_path + "/data/bilinear_weight.npy"
        self.bilinear_bais_path = root_path + "/data/bilinear_bais.npy"
        self.text_01feature_path = root_path + "/data/text_01feature2.json"
        self.text_01_sim_mode = "dot"
        self.hidden_dim = 768
        self.item_num = 1649
        self.attribute_num = 24
        self.parent_attribute_num = 17
        self.alpha = 1.
        self.max_rec_item_num = 5
        self.max_sen_selected = 3
        self.match_sen_num = 1

        self.vocab_path = root_path + "/data/bert/vocab.txt"
        self.pretrained_bert_fold = root_path + "/data/bert/"
        self.pretrained_bert_config = root_path + "/data/bert/bert_config.json"
        self.finetuned_model_path = root_path + "/data/bert/finetuned_model.pkl"
        self.max_seq_len = 80 #512
        self.bert_embed_size = 768

        self.text_01feature_size = 24
        self.feature2id_file_path = root_path + "/data/regex/feature2id.json"

        self.load_similar_matrix = True
        self.similar_matrix_path = root_path + '/data/sim_matrix.npy'