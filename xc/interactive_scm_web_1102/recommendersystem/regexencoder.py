import re
import random
import json

# 返回原告人数等级
def get_plantiff_features(text):
    def f_plantiff_is_company(x):
        r = re.search(r"原告(.*?)被告", x)
        if r:
            plantiff = r.group(1)
            return "法定代表人" in plantiff or "公司" in plantiff
        else:
            return False

    text = text.split(f'\n\n')[0]
    reg = re.compile(r"原告")
    num = len(reg.findall(text))
    is_company = f_plantiff_is_company(text)

    if num <= 1:
        num_level = 0
    elif num <= 4:
        num_level = 1
    else:
        num_level = 2

    return_dict = {"plantiff_num": num_level, "plantiff_is_company": is_company}
    return return_dict

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
    text1 = text.split(f'\n\n')[0]
    # print("text1:", text1)
    text2 = text.split(f'\n\n')[1]
    reg = re.compile(r"被告.*?法定代表人|公司.*?。")
    is_company = len(reg.findall(text1)) > 0
    no_reply = f_defandent_noreply(text2)
    reg = re.compile(r"被告")
    num = len(reg.findall(text1))

    if num <= 1 :
        num_level = 0
    elif num <= 4:
        num_level = 1
    else:
        num_level = 2

    return_dict = {"defandent_num": num_level, "defandent_is_company": is_company, "defandent_no_reply": no_reply}
    return return_dict


def get_guarantor_features(text):
    text = text.split(f'\n\n')[1]
    reg = re.compile(r"担保")
    is_guarantee = len(reg.findall(text)) > 0
    reg = re.compile(r"抵押")
    is_mortgage = len(reg.findall(text)) > 0

    return_dict = {"is_guarantee": is_guarantee, "is_mortgage": is_mortgage}
    return return_dict


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

    text = text.split(f'\n\n')[1]
    reg = re.compile(r"约定利息|约定月利息|年息|月息|利息|利率")
    is_interest = len(reg.findall(text)) > 0
    level, num = do_lixi(text)

    # return_dict = {"is_interest": is_interest, "interest_level": level, "interest_num": num}
    return_dict = {"is_interest": is_interest, "interest_level": level}
    return return_dict


def get_agreement_features(text):
    text = text.split(f'\n\n')[1]
    reg = re.compile(r"借条|欠条|借据")
    is_receipt = len(reg.findall(text)) > 0
    reg = re.compile(r"合同")
    is_contract = len(reg.findall(text)) > 0
    reg = re.compile(r"聊天记录")
    is_records = len(reg.findall(text)) > 0

    return_dict = {"is_receipt": is_receipt, "is_contract": is_contract, "is_records": is_records}
    return return_dict


def get_repayment_features(text):
    text = text.split(f'\n\n')[1]
    reg = re.compile(r"(剩余.{0,20}(未|不))|((未|不).{0.20}剩余)|尚欠")
    is_part = len(reg.findall(text)) > 0

    return_dict = {"is_part": is_part}
    return return_dict

def get_borrowing_features(text):
    text = text.split(f'\n\n')[1]
    reg = re.compile(r"原告.{0,30}(现金).{0,20}(交|支)付")
    is_cash = len(reg.findall(text)) > 0

    reg = re.compile(r"原告.{0,30}(银行转账).{0,20}(交|支)付")
    is_bank_transfer = len(reg.findall(text)) > 0

    reg = re.compile(r"原告.{0,30}(银行卡).{0,20}(交|支)付")
    is_authorization = len(reg.findall(text)) > 0

    reg = re.compile(r"原告.{0,30}(支付宝|微信).{0,20}(交|支)付")
    is_online = len(reg.findall(text)) > 0

    return_dict = {"is_cash": is_cash, "is_bank_transfer": is_bank_transfer, \
                    "is_authorization": is_authorization, "is_online": is_online}
    return return_dict


class RegexEncoder(object):
    def __init__(self, feature_size, feature2id_file_path):
        self.feature_size = feature_size
        with open(feature2id_file_path, "r", encoding="utf-8") as f:
            self.feature2id = json.load(f)

    def get_01feature(self, text):
        total_dict = dict(list(get_plantiff_features(text).items()) + list(get_defendant_features(text).items())
                      + list(get_guarantor_features(text).items()) + list(get_interest_features(text).items())
                      + list(get_agreement_features(text).items()) + list(get_repayment_features(text).items())
                      + list(get_borrowing_features(text).items()))

        feature_list = []
        for feature, value in total_dict.items():
            if isinstance(self.feature2id[feature], dict):
                feature_list.append(self.feature2id[feature][str(value)])
            else:
                if value == True:
                    feature_list.append(self.feature2id[feature])

        enc_array = []
        for i in range(self.feature_size):
            if i in feature_list:
                enc_array.append(1)
            else:
                enc_array.append(0)

        return enc_array