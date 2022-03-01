# coding:utf-8
import re
import json

print(False or True)
def guoqi(text):
    dongshi = "董事" in text
    qiye = "企业" in text
    jingli = "经理" in text
    gongsi = "公司" in text
    fuzeren = "负责人" in text
    guoyou = "国有" in text
    if dongshi or qiye or jingli or gongsi or fuzeren or guoyou == True :
        i = 1
    else:
        i = 0
    return i

def jiguan(text):
    guzhang = "股长" in text
    shengzhang = "省长" in text
    zhiwei = "支委" in text
    zhizhang = "支长" in text
    guojia = "国家" in text
    xiangzhang = "乡长" in text
    tongzhi = "同志" in text
    xianzhang = "县长" in text
    keyuan = "科员" in text
    zhuren = "主任" in text
    diaoyanyuan = "调研员" in text
    kezhang = "科长" in text
    chuzhang = "处长" in text
    buzhang = "部长" in text
    juzhang = "局长" in text
    tingzhang = "厅长" in text
    zhenzhang = "镇长" in text
    if guzhang or shengzhang or kezhang or zhiwei or zhizhang or guojia or xiangzhang or tongzhi or xianzhang or keyuan or zhuren or diaoyanyuan or chuzhang or buzhang or juzhang or tingzhang or zhenzhang == True:
        i = 1
    else:
        i = 0
    return i



def weituo(text):
    guoyoucanchang="国有财产" in text
    weituoguanli= "委托管理" in text
    weituojingyan = "委托经营" in text
    if guoyoucanchang or weituojingyan or weituoguanli == True:
        i = 1
    else:
        i = 0
    return i

def yishen(text):
    yuanshen = "原审" in text
    zaishen = "再审" in text
    ershen = "二审" in text
    if yuanshen or zaishen or ershen == True:
        i = 0
    else:
        i = 1
    return i

def ershen(text):
    yuanshen = "原审" in text
    zaishen = "再审" in text
    ershen = "二审" in text
    if yuanshen or zaishen or ershen == True:
        i = 1
    else:
        i = 0
    return i

def zhiwu(text):
    liyongzhiwu = "利用职务" in text
    zhiwubianli = "职务便利" in text
    if liyongzhiwu or zhiwubianli == True:
        i = 1
    else:
        i = 0
    return i

def huizui(text):
    rushigongshu = "如实供述" in text
    huizui = "悔罪" in text
    if rushigongshu or huizui == True:
        i = 1
    else:
        i = 0
    return i

class regexencoder(object):
    def __init__(self):
        pass
    def get_01feature(self,text):
        label_dict={}
        get_label=[guoqi,jiguan,weituo,yishen,ershen,zhiwu,huizui]
        label_name=['guoqi','jiguan','weituo','yishen','ershen','zhiwu','huizui']
        for i in range(len(label_name)):
            label_dict[label_name[i]]=[get_label[i](text)]
        return label_dict

text="经本院审理,认为原判认定上诉人周某某、阳某某构成贪污罪的犯罪事实清楚,所依据的证据确实、充分,且定案证据均经开庭审理举证、质证,具有合法性、客观性及关联性,本院予以确认。在本院审理过程中,周某某及其辩护人,阳某某及其辩护人均未提出新的证据。故,本院查明的犯罪事实与原判查明的一致。对于上诉人的上诉理由及辩护人的辩护意见,本院综合评判如下:一、对于周某某的上诉理由及辩护人的辩护意见。经查,1、现有证据已充分证实涉案的5804946元是由周某某与阳某某二人通过虚假的工程合同且共同在付款报告上签字同意,该款才得以顺利转出到中意公司;2、该款转出后主要由周某某实际控制与使用,没有证据证实桂林市地产公司及平乐县国土资源局作出将该款作为“小金库”使用的决定;3、发放桂林市地产公司员工福利的钱是从周某某等人挂名在下属公司所领取的工资及奖金中支出,并非出自本案套取的5804946元。上述事实有施工合同、付款报告、银行凭证及流水、桂林市地产公司情况说明及记账凭证、奖金福利发放单据等书证证实,证人石某、沈某、莫某、赵某、黄某、胡某、易某、靳某的证言亦均能证实周某某事前并未将该事汇报单位,单位也未曾决定将该款套取后作为单位“小金库”用于发放员工福利及单位开支的事实,且周某某及其辩护人在二审期间亦未能提供相应的新证据。故在案证据足以证实周某某伙同阳某某贪污公款5804946元的事实,不支持周某某及其辩护人提出的无罪辩解和辩护意见。二、对于阳某某的上诉理由及辩护人的辩护意见。经查,1、阳某某在与周某某共同贪污涉案款项的过程中,参与共同虚构桂乐公司与中意公司虚假工程合同,并在必须有其和周某某签字才有效的付款报告上签字,事后又采用开具税务发票的方式将该款入账、平账,使该款最终得以由二人实际控制、使用。故阳某某在这其中与周某某同起主要作用,同为主犯,但作用确实相对犯意提出者及中意公司的实际控制人周某某而言较轻;2、桂林市秀峰区人民检察院于2008年对阳某某伙同他人共同贪污公款30多万元的犯罪事实进行了指控,同年秀峰区法院也对该案进行了判决。根据阳某某在纪委问话阶段就已如实交代该犯罪事实的情况,结合当时适用的1998年《最高人民法院关于处理自首和立功具体应用法律若干问题的解释》规定,阳某某是在未被司法机关发觉,或者讯问,未被采取强制措施时就已经供述,符合自首条件。公诉机关及一、二审法院均认定阳某某构成自首。现在案证据证实,阳某某不仅在2007年纪委问话时交代了当时起诉的犯罪事实,而且也已经将其与周某某二人在2004年把桂乐公司的5804946元公款以虚假施工合同的名义套取到中意公司并进行平账的行为进行了交代。按此逻辑,阳某某当时在纪委已经交代了其涉及本案的主要犯罪事实。依照《最高人民法院、最高人民检察院关于适用刑事司法解释时间效力问题的规定》,本案中阳某某应适用1998年《最高人民法院关于处理自首和立功具体应用法律若干问题的解释》规定,也构成自首,可从轻处罚。但其与周某某是共同犯罪,揭发周某某只是其应如实供述的内容,依法并不构成立功。"
a=regexencoder()
print(a.get_01feature(text))