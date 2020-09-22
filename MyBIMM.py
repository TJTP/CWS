import numpy as np 
import time 
import MyHMM
from MyHMM import viterbi, seg_sentence, seg_substr_f, seg_substr_b
from MyUtils import get_median

#测试集路径
testSetPath = "./test.txt"
#模型版本号
modelVer = "2020-09-20-11_23_01"
#中文数字, 匹配时合在一起
quanChars = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九', '十', '廿', \
                '百','千', '万', '亿']
#这类字符在他们前面出现阿拉伯或中文数字时, 合在一起
dateChars = ['时', '日', '天', '月', '年', '后', '件', '个']
#当阿拉伯数字中间出现这些字符时, 连在一起
digiChars = ['.', ':']
#单字的概率阈值, 如果低于它, 则用HMM检查, 看能否组成一个词
probThreshold = -10.00
#单元测试时的句子
unitTestSentence = \
    ["可口可乐在校园里设立了一个冰箱，提供新产品：如果你试图拧开瓶盖，你会发现这是一件不可能完成的事情。", \
    "《经济日报》产经新闻部主任崔书文：负面清单管理制度的推出和修订，提高了开放度和透明度。", \
    "（4）月宫一号志愿者105天实验后出舱，氧气、水、食物实现自循环。", \
    "现在儿子电话响应速度快多了，妈妈又开始研发iPhone版…", \
    "光看此举，中国队赢了，虽比赛以平局告终，但却绝对是正能量。"]

#导入字典, windowSize
def import_models():
    d = np.load("./Models/" + modelVer + "-Dic.npy", allow_pickle=True).item()
    w = np.load("./Models/" + modelVer + "-WS.npy", allow_pickle=True).item()

    return d, w

#正向匹配
def fmm(Dic, windowSize, sentence, A, B, Pi):
    sentenceLength = len(sentence)
    idx = 0
    segSentenceList = []
    sentencePreLabels = viterbi(sentence, A, B, Pi)

    while idx < sentenceLength:
        match = False
        #当出现阿拉伯数字时
        if sentence[idx].isdigit():
            j = idx + 1
            while j < sentenceLength:
                if sentence[j].isdigit() == False and sentence[j] not in digiChars:
                    break
                j += 1
            if j < sentenceLength and (sentence[j] in quanChars or sentence[j] in dateChars):
                segSentenceList.append(sentence[idx: j + 1])
                idx = j + 1
                match = True 
                continue
            segSentenceList.append(sentence[idx: j])
            idx = j 
            match = True
            continue
        
        #当出现中文的数量词时
        if sentence[idx] in quanChars:
            j = idx + 1
            while j < sentenceLength:
                if sentence[j] not in quanChars:
                    break
                j += 1
            if j < sentenceLength and sentence[j] in dateChars:
                segSentenceList.append(sentence[idx: j + 1])
                idx = j + 1
                match = True
                continue
            segSentenceList.append(sentence[idx: j])
            idx = j 
            match = True
            continue
        
        #当出现英文字母时
        if sentence[idx].encode("UTF-8").isalpha():
            j = idx + 1
            while j < sentenceLength:
                if sentence[j].encode("UTF-8").isalpha() == False and sentence[j].isdigit() == False \
                        and sentence[j] != '.':
                    break
                j += 1
            segSentenceList.append(sentence[idx: j])
            match = True
            idx = j 
            continue

        if windowSize > sentenceLength - idx: #动态调整windowSize大小
            windowSize = sentenceLength - idx
        for i in range(windowSize, 0, -1):
            #以WindowSize为起始宽度按照字典滑动匹配
            subStr = sentence[idx: idx + i]
            if subStr in Dic:
                match = True
                if len(subStr) == 1 and Dic[subStr] < probThreshold:
                    end = seg_substr_f(sentencePreLabels, idx)
                    segSentenceList.append(sentence[idx: end])
                    idx = end 
                    break
                segSentenceList.append(subStr)
                idx += i
                break

        #如果各种规则以及字典都无法匹配    
        if not match: 
            end = seg_substr_f(sentencePreLabels, idx)
            segSentenceList.append(sentence[idx: end])
            idx = end 
    return segSentenceList

#反向匹配
def bmm(Dic, windowSize, sentence, A, B, Pi):
    sentenceLength = len(sentence)
    idx = sentenceLength - 1
    segSentenceList = []
    sentencePreLabels = viterbi(sentence, A, B, Pi)

    while idx >= 0:
        match = False
        #当出现阿拉伯数字时, 分为有时间指示词和纯阿拉伯数字两种情况
        if sentence[idx] in dateChars and idx -1 >= 0and sentence[idx - 1].isdigit():
            j = idx - 1
            while j >= 0:
                if sentence[j].isdigit() == False and sentence[j] not in digiChars:
                    break
                j -= 1
            segSentenceList.append(sentence[j + 1: idx + 1])
            match = True
            idx = j
            continue
        if sentence[idx].isdigit():
            j = idx
            while j >= 0:
                if sentence[j].isdigit() == False and sentence[j] not in digiChars:
                    break
                j -= 1
            segSentenceList.append(sentence[j + 1: idx + 1])
            match = True
            idx = j
            continue

        #当出现中文数字时, 分为有时间指示词和纯中文数字两种情况
        if sentence[idx] in dateChars and sentence[idx - 1] in quanChars:
            j = idx - 1
            while j >= 0:
                if sentence[j] not in quanChars:
                    break
                j -= 1
            segSentenceList.append(sentence[j + 1: idx + 1])
            match = True
            idx = j
            continue
        if sentence[idx] in quanChars:
            j = idx 
            while j >= 0:
                if sentence[j] not in quanChars:
                    break
                j -= 1
            segSentenceList.append(sentence[j + 1: idx + 1])
            match = True
            idx = j
            continue

        #当出现英文字母时
        if sentence[idx].encode("UTF-8").isalpha():
            j = idx 
            while j >= 0:
                if sentence[j].encode("UTF-8").isalpha() == False and sentence[j].isdigit() == False \
                    and sentence[j] != '.':
                    break
                j -= 1
            segSentenceList.append(sentence[j + 1: idx + 1])
            match =True
            idx = j
            continue

        #按照WindowSize大小滑动匹配
        if windowSize > idx - 0 + 1:
            windowSize = idx - 0 + 1
        for i in range(windowSize - 1, 0 - 1, -1):
            subStr = sentence[idx - i: idx + 1]
            if subStr in Dic:
                match = True
                if len(subStr) == 1 and Dic[subStr] < probThreshold:
                    begin = seg_substr_b(sentencePreLabels, idx)
                    segSentenceList.append(sentence[begin: idx + 1])
                    idx = begin - 1
                    break 
                segSentenceList.append(subStr)               
                idx = (idx - 1) - i
                break
        if not match:
            begin = seg_substr_b(sentencePreLabels, idx)
            segSentenceList.append(sentence[begin: idx + 1])
            idx = begin - 1
    
    segSentenceList.reverse()
    return segSentenceList

#双向匹配
def bimm_enhance(Dic, windowSize, sentence, A, B, Pi):
    fSegSentenceList = fmm(Dic, windowSize, sentence, A, B, Pi)
    bSegSentenceList = bmm(Dic, windowSize, sentence, A, B, Pi )
    sentencePreLabels = viterbi(sentence, A, B, Pi)
    hSegSentenceList = seg_sentence(sentence, sentencePreLabels)

    #print("Forward: %s\nBackward: %s"%(fSegSentenceList, bSegSentenceList))

    if fSegSentenceList == bSegSentenceList:
        return fSegSentenceList
    
    fWordCnt = len(fSegSentenceList)
    bWordCnt = len(bSegSentenceList)
    hWordCnt = len(hSegSentenceList)

    fSingleCnt = bSingleCnt = 0
    fMaxSingle = bMaxSingle = 0
    fCurSingle = bCurSingle = 0
    fProbSum = bProbSum = 0

    for word in fSegSentenceList:
        if len(word) == 1:
            fCurSingle += 1
            fSingleCnt += 1
        else:
            if fCurSingle > fMaxSingle:
                fMaxSingle = fCurSingle
            fCurSingle = 0
        if word in Dic:
            fProbSum += Dic[word]

    for word in bSegSentenceList:
        if len(word) == 1:
            bCurSingle += 1
            bSingleCnt += 1
        else:
            if bCurSingle > bMaxSingle:
                bMaxSingle = bCurSingle
            bCurSingle = 0
        if word in Dic:
            bProbSum += Dic[word]
    
    weights = [10, 34, 42, 42]
    wcWeight = weights[0] /sum(weights) #分词数量的权重
    scWeight = weights[1] /sum(weights) #单字数量的权重
    msWeight = weights[2] /sum(weights) #最大连续单字数量的权重
    prWeight = weights[3] /sum(weights) #概率对数的和的权重

    normalizer = 0.1 #概率对数的和的归一化因子
    #正向匹配结果的惩罚分数
    fPenScore = wcWeight * fWordCnt + scWeight * fSingleCnt + msWeight * fMaxSingle + prWeight * (-fProbSum) * normalizer
    #逆向匹配结果的惩罚分数
    bPenScore = wcWeight * bWordCnt + scWeight * bSingleCnt + msWeight * bMaxSingle + prWeight * (-bProbSum) * normalizer
    
    #优先返回反向匹配的结果
    if fPenScore < bPenScore:
        return fSegSentenceList
    else:
        return bSegSentenceList

#单元测试
def unit_test():
    sentence = unitTestSentence[0]
    Dic, windowSize = import_models()
    print("Max key and its value: (%s: %f)"%(max(Dic, key=Dic.get), Dic[max(Dic, key=Dic.get)]))
    print("Median of Dic: %f"%(get_median(list(Dic.values()))))
    A, B, Pi = MyHMM.import_models()
    fSegSentenceList = fmm(Dic, windowSize, sentence, A, B, Pi)
    bSegSentenceList = bmm(Dic, windowSize, sentence, A, B, Pi)
    biSegSentenceList = bimm_enhance(Dic, windowSize, sentence, A, B, Pi)

    sentencePreLabels = viterbi(sentence, A, B, Pi)
    hSegSentenceList = seg_sentence(sentence, sentencePreLabels)
    print("FMM: %s\nBMM: %s\nBiMM:%s\nLabels: %s\nHMM: %s"%(fSegSentenceList, bSegSentenceList, \
                biSegSentenceList,sentencePreLabels,hSegSentenceList))
    
#对测试集进行分词
def apply_all():
    Dic, windowSize = import_models()
    A, B, Pi = MyHMM.import_models()
    testSet = open(testSetPath, encoding="utf-8")

    strTime = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(int(time.time())))
    outputFile = open("./Results/181220039-bi-" + strTime +".txt", "a", encoding="utf-8")
    sentence = testSet.readline().strip()
    while sentence:
        segSentenceList = bimm_enhance(Dic, windowSize, sentence, A, B, Pi)
        
        segSentenceStr = ''
        for idx in range(len(segSentenceList) - 1):
            segSentenceStr = segSentenceStr + segSentenceList[idx] + ' '
        segSentenceStr = segSentenceStr + segSentenceList[-1] + '\n'

        outputFile.write(segSentenceStr)

        sentence = testSet.readline().strip()
    
    testSet.close()
    outputFile.close()

if __name__ == "__main__":
    apply_all()
    #unit_test()

