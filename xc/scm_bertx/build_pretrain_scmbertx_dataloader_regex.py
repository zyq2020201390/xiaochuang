import json
import torch
import re
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

def get_plantiff_features(text):
    def f_plantiff_is_company(x):
        r = re.search(r"原告(.*?)被告", x)
        if r:
            plantiff = r.group(1)
            return "法定代表人" in plantiff or "公司" in plantiff
        else:
            return False

    # text = text.split('\\n\\n')[0]
    reg = re.compile(r"原告")
    num = len(reg.findall(text))
    is_company = f_plantiff_is_company(text)

    if num <= 1:
        num_level = 0
    elif num <= 4:
        num_level = 1
    else:
        num_level = 2

    # return_dict = {"plantiff_num": num_level, "plantiff_is_company": is_company}
    # return return_dict
    return_list = []
    if is_company:
        return_list.append("plantiff_is_company")
    return return_list

# 返回被告人数等级
def get_defendant_features(text):
    def f_defandent_noreply(text):
        if any(
            ss in text
            for ss in ["未答辩", "拒不到庭", "未到庭", "未做答辩", "未应诉答辩", "未作出答辩", "未出庭"]
        ):
            return True
        return False
    # print("text:", text)
    # text1 = text.split('\\n\\n')[0]
    # print("text1:", text1)
    # text2 = text.split('\\n\\n')[1]
    reg = re.compile(r"被告.*?法定代表人|公司.*?。")
    is_company = len(reg.findall(text)) > 0
    no_reply = f_defandent_noreply(text)
    reg = re.compile(r"被告")
    num = len(reg.findall(text))

    if num <= 1 :
        num_level = 0
    elif num <= 4:
        num_level = 1
    else:
        num_level = 2

    # return_dict = {"defandent_num": num_level, "defandent_is_company": is_company, "defandent_no_reply": no_reply}
    # return return_dict
    return_list = []
    if is_company:
        return_list.append("defandent_is_company")
    if no_reply:
        return_list.append("defandent_no_reply")
    return return_list


def get_guarantor_features(text):
    # text = text.split('\\n\\n')[1]
    reg = re.compile(r"担保")
    is_guarantee = len(reg.findall(text)) > 0
    reg = re.compile(r"抵押")
    is_mortgage = len(reg.findall(text)) > 0

    # return_dict = {"is_guarantee": is_guarantee, "is_mortgage": is_mortgage}
    # return return_dict
    return_list = []
    if is_guarantee:
        return_list.append('is_guarantee')
    if is_mortgage:
        return_list.append('is_mortgage')
    return return_list


def get_interest_features(text):
    def do_lixi(text):
        m_reg = re.compile(r"(月利息|月息|月利率|月利息率)按?(\d+(\.\d{1,2})?)(％|分)")
        mm = m_reg.search(text)

        m2_reg = re.compile(r"(月利息|月息|月利率|月利息率)按?(\d+(\.\d{1,2})?)毛")
        mm2 = m2_reg.search(text)

        m3_reg = re.compile(r"月(\d+(\.\d{1,2})?)％(利息|息|利率|利息率)")
        mm3 = m3_reg.search(text)

        y_reg = re.compile(r"(年利息|年息|年利率|年利息率)按?(\d+(\.\d{1,2})?)(％|分)")
        ym = y_reg.search(text)

        y2_reg = re.compile(r"(年利息|年息|年利率|年利息率)按?(\d+(\.\d{1,2})?)毛")
        ym2 = y2_reg.search(text)

        y3_reg = re.compile(r"年(\d+(\.\d{1,2})?)％(利息|息|利率|利息率)")
        ym3 = y3_reg.search(text)

        count = 0

        if mm:
            count = round(float(mm.group(2)) * 12, 2)
        elif mm2:
            count = round(float(mm2.group(2)) * 10 * 12, 2)
        elif mm3:
            count = round(float(mm3.group(1)) * 12, 2)
        elif ym:
            count = float(ym.group(2))
        elif ym2:
            count = round(float(ym2.group(2)) * 10, 2)
        elif ym3:
            count = float(ym3.group(1))
        else:
            count = 0

        if count == 0:
            return 0, count
        elif count < 24:
            return 1, count
        elif count < 36:
            return 2, count
        else:
            return 3, count

    # text = text.split('\\n\\n')[1]
    reg = re.compile(r"约定利息|约定月利息|年息|月息|利息|利率")
    is_interest = len(reg.findall(text)) > 0
    level, num = do_lixi(text)

    # return_dict = {"is_interest": is_interest, "interest_level": level, "interest_num": num}
    # return_dict = {"is_interest": is_interest, "interest_level": level}

    return_list = []
    if is_interest:
        return_list.append('is_interest') 
    if level != 0:
        return_list.append(f'interest_level{level}')

    return return_list


def get_agreement_features(text):
    # text = text.split('\\n\\n')[1]
    reg = re.compile(r"借条|欠条|借据")
    is_receipt = len(reg.findall(text)) > 0
    reg = re.compile(r"合同")
    is_contract = len(reg.findall(text)) > 0
    reg = re.compile(r"聊天记录")
    is_records = len(reg.findall(text)) > 0

    # return_dict = {"is_receipt": is_receipt, "is_contract": is_contract, "is_records": is_records}
    # return return_dict
    return_list = []
    if is_receipt:
        return_list.append('is_receipt')
    if is_contract:
        return_list.append('is_contract')
    if is_records:
        return_list.append('is_records')
    return return_list


def get_repayment_features(text):
    # text = text.split('\\n\\n')[1]
    reg = re.compile(r"(剩余.{0,20}(未|不))|((未|不).{0.20}剩余)|尚欠")
    is_part = len(reg.findall(text)) > 0

    # return_dict = {"is_part": is_part}
    # return return_dict

    return_list = []
    if is_part:
        return_list.append('is_part')
    return return_list

def get_borrowing_features(text):
    # text = text.split('\\n\\n')[1]
    reg = re.compile(r"原告.{0,30}(现金).{0,20}(交|支)付")
    is_cash = len(reg.findall(text)) > 0

    reg = re.compile(r"原告.{0,30}(银行转账).{0,20}(交|支)付")
    is_bank_transfer = len(reg.findall(text)) > 0

    reg = re.compile(r"原告.{0,30}(银行卡).{0,20}(交|支)付")
    is_authorization = len(reg.findall(text)) > 0

    reg = re.compile(r"原告.{0,30}(支付宝|微信).{0,20}(交|支)付")
    is_online = len(reg.findall(text)) > 0

    # return_dict = {"is_cash": is_cash, "is_bank_transfer": is_bank_transfer, \
    #                 "is_authorization": is_authorization, "is_online": is_online}
    # return return_dict
    return_list = []
    if is_cash:
        return_list.append('is_cash')
    if is_bank_transfer:
        return_list.append('is_bank_transfer')
    if is_authorization:
        return_list.append('is_authorization')
    if is_online:
        return_list.append('is_online')
    return return_list

def load_ori_text(file_path):
    data_list = []
    with open(file_path, "r", encoding="utf8") as f:
        for idx, line in enumerate(f):
            x = json.loads(line)
            # A, B, C = x["A"], x["B"], x["C"]
            data_list.append(x)
    return data_list



class RegexEncoder(object):
    def __init__(self):
        feat_list = ["plantiff_is_company", "defandent_is_company", "defandent_no_reply", \
                        "is_guarantee", "is_mortgage", \
                        "is_interest", "interest_level1", "interest_level2", "interest_level3", \
                        "is_receipt", 'is_contract', 'is_records', 'is_part', \
                        "is_cash", "is_bank_transfer", "is_authorization", "is_online"]
        self.feat2id = {}
        for idx, feat in enumerate(feat_list):
            self.feat2id[feat] = idx

    def get_feture(self, text, return_tensor=True):
        all_list = []
        all_list += get_plantiff_features(text)
        all_list += get_defendant_features(text)
        all_list += get_guarantor_features(text)
        all_list += get_interest_features(text)
        all_list += get_agreement_features(text)
        all_list += get_repayment_features(text)
        all_list += get_borrowing_features(text)

        one_hot_rep = [0.] * len(self.feat2id)
        for feat in all_list:
            one_hot_rep[self.feat2id[feat]] = 1.    
        if return_tensor:
            one_hot_rep = torch.tensor(one_hot_rep)
        return one_hot_rep

    def get_batch_feture(self, text_list, return_tensor=True):
        one_hot_rep_list = []
        for text in text_list:
            one_hot_rep_list.append(self.get_feture(text, False))
        if return_tensor:
            one_hot_rep_list = torch.tensor(one_hot_rep_list)
        return one_hot_rep_list


class PretrainSCMBertDataGenerator(Dataset):
    def __init__(self, jsonfile_path):
        super(PretrainSCMBertDataGenerator, self).__init__()
        self.data_list = []
        with open(jsonfile_path, "r", encoding="utf8") as f:
            for idx, line in enumerate(f):
                x = json.loads(line)
                A, B, C = x["A"], x["B"], x["C"]
                if "label" in x.keys():
                    label = int(x["label"] == "C")
                else:
                    label = 0
                self.data_list.append([A, B, C, label])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        A, B, C, label = self.data_list[index]
        return A, B, C, label

class Collate:
    def __init__(self, vocab_path, max_seq_len):
        self.tokenizer = BertTokenizer.from_pretrained(vocab_path)
        self.regex_encoder = RegexEncoder()
        self.max_seq_len = max_seq_len

    def __call__(self, batch):
        A_batch, B_batch, C_batch, label_batch = zip(*batch)
        assert len(A_batch) == 1 
        # print("A_batch: ", A_batch)
        A_one_hot_batch = self.regex_encoder.get_batch_feture(A_batch[0])
        B_one_hot_batch = self.regex_encoder.get_batch_feture(B_batch[0])
        C_one_hot_batch = self.regex_encoder.get_batch_feture(C_batch[0])
        A_batch = self.tokenizer(A_batch[0], padding=True, truncation=True, max_length=self.max_seq_len, return_tensors="pt")
        B_batch = self.tokenizer(B_batch[0], padding=True, truncation=True, max_length=self.max_seq_len, return_tensors="pt")
        C_batch = self.tokenizer(C_batch[0], padding=True, truncation=True, max_length=self.max_seq_len, return_tensors="pt")
        label_batch = torch.tensor(label_batch[0])
        return A_batch, B_batch, C_batch, A_one_hot_batch, B_one_hot_batch, C_one_hot_batch, label_batch

def build_pretrain_scmbertx_dataloader(jsonfile_path, vocab_path, batch_size, shuffle=True, num_workers=0, max_seq_len=512):
    data_generator = PretrainSCMBertDataGenerator(jsonfile_path)
    collate = Collate(vocab_path, max_seq_len)
    return DataLoader(
        data_generator,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate
    )