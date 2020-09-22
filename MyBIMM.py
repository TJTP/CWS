import numpy as np 
import time 
import MyHMM
from MyHMM import viterbi, seg_sentence, seg_substr_f, seg_substr_b

#测试集路径
testSetPath = "./test.txt"
#模型版本号
modelVer = "2020-09-20-11_23_01"
#labels = MyHMM.labels
quanChars = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九', '十', '廿', \
                '百','千', '万', '亿']
dateChars = ['时', '日', '天', '月', '年', '后']


def import_models():
    d = np.load("./Models/" + modelVer + "-Dic.npy", allow_pickle=True).item()
    w = np.load("./Models/" + modelVer + "-WS.npy", allow_pickle=True).item()

    return d, w

def fmm(Dic, windowSize, sentence, A, B, Pi):
    sentenceLength = len(sentence)
    idx = 0
    segSentenceList = []
    sentencePreLabels = viterbi(sentence, A, B, Pi)

    while idx < sentenceLength:
        match = False
        #当出现阿拉伯数字时
        if sentence[idx].isdigit():
            #print("alb " + sentence[idx])
            j = idx + 1
            while j < sentenceLength:
                if sentence[j].isdigit() == False and sentence[j] != '.':
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
        if (sentence[idx] in quanChars) and (sentence[idx + 1] in quanChars):
            #print("zw "+ sentence[idx])
            j = idx + 2
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
            #print("yw " + sentence[idx])
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
            #以WindowSize为起始与字典滑动匹配
            subStr = sentence[idx: idx + i]
            if subStr in Dic:
                segSentenceList.append(subStr)
                match = True
                idx += i
                break
        #如果各种规则以及字典    
        if not match: #可以改进的地方1
            #segSentenceList.append(sentence[idx])
            #idx += 1
            end = seg_substr_f(sentencePreLabels, idx)
            segSentenceList.append(sentence[idx: end])
            idx = end 
    return segSentenceList

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
                if sentence[j].isdigit() == False and sentence[j] != '.':
                    break
                j -= 1
            segSentenceList.append(sentence[j + 1: idx + 1])
            match = True
            idx = j
            continue
        if sentence[idx].isdigit():
            j = idx
            while j >= 0:
                if sentence[j].isdigit() == False and sentence[j] != '.':
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

        if windowSize > idx - 0 + 1:
            windowSize = idx - 0 + 1
        for i in range(windowSize - 1, 0 - 1, -1):
            subStr = sentence[idx - i: idx + 1]
            if subStr in Dic:
                segSentenceList.append(subStr)
                match = True
                idx = (idx - 1) - i
                break
        if not match:
            #segSentenceList.append(sentence[idx - 1])
            #idx -= 1
            begin = seg_substr_b(sentencePreLabels, idx)
            segSentenceList.append(sentence[begin: idx + 1])
            idx = begin - 1
    segSentenceList.reverse()
    return segSentenceList

def bimm_enhance(Dic, windowSize, A, B, Pi, sentence):
    fSegSentenceList = fmm(Dic, windowSize, sentence, A, B, Pi)
    bSegSentenceList = bmm(Dic, windowSize, sentence, A, B, Pi )
    sentencePreLabels = viterbi(sentence, A, B, Pi)
    hSegSentenceList = seg_sentence(sentence, sentencePreLabels)

    #print("Forward: %s\nBackward: %s"%(fSegSentenceList, bSegSentenceList))

    #if fSegSentenceList == bSegSentenceList:
    #    return fSegSentenceList
    
    fWordCnt = len(fSegSentenceList)
    bWordCnt = len(bSegSentenceList)
    hWordCnt = len(hSegSentenceList)

    fSingleCnt = bSingleCnt = hSingleCnt = 0
    fNonDicCnt = bNonDicCnt = hNonDicCnt = 0


    for word in fSegSentenceList:
        if len(word) == 1:
            fSingleCnt += 1
        if word not in Dic:
            fNonDicCnt += 1

    for word in bSegSentenceList:
        if len(word) == 1:
            bSingleCnt += 1
        if word not in Dic:
            bNonDicCnt += 1
    
    for word in hSegSentenceList:
        if len(word) == 1:
            hSingleCnt += 1
        if word not in Dic:
            hNonDicCnt += 1
    
    wcWeight = 0.4
    scWeight = 0.4
    ndWeight = 0.1
    fPenScore = wcWeight * fWordCnt + scWeight * fSingleCnt + ndWeight * fNonDicCnt
    bPenScore = wcWeight * bWordCnt + scWeight * bSingleCnt + ndWeight * bNonDicCnt
    hPenScore = wcWeight * hWordCnt + scWeight + hSingleCnt + ndWeight * hNonDicCnt

    '''scoreList = [fPenScore, bPenScore, hPenScore]
    if hPenScore == min(scoreList):
        if bPenScore == hPenScore:
            return bSegSentenceList
        return hSegSentenceList
    elif fPenScore == min(scoreList):
        if bPenScore == fPenScore:
            return bSegSentenceList
        return fSegSentenceList
    elif bPenScore == min(scoreList):
        return bSegSentenceList'''
    
    if fPenScore < bPenScore:
        return fSegSentenceList
    else:
        return bSegSentenceList

def unit_test():
    sentence = "《经济日报》产经新闻部主任崔书文：负面清单管理制度的推出和修订，提高了开放度和透明度。"
    #sentence = "（4）月宫一号志愿者105天实验后出舱，氧气、水、食物实现自循环。"
    #sentence = "现在儿子电话响应速度快多了，妈妈又开始研发iPhone版…"
    Dic, windowSize = import_models()
    A, B, Pi = MyHMM.import_models()
    fSegSentenceList = fmm(Dic, windowSize, sentence, A, B, Pi)
    bSegSentenceList = bmm(Dic, windowSize, sentence, A, B, Pi)

    sentencePreLabels = viterbi(sentence, A, B, Pi)
    hSegSentenceList = seg_sentence(sentence, sentencePreLabels)
    print("Forward: %s\nBackward: %s\nLabels: %s\nHMM: %s"%(fSegSentenceList, bSegSentenceList, sentencePreLabels,hSegSentenceList))
    

def apply_all():
    Dic, windowSize = import_models()
    A, B, Pi = MyHMM.import_models()
    testSet = open(testSetPath, encoding="utf-8")
    sentence = testSet.readline().strip()
    strTime = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(int(time.time())))
    outputFile = open("./Results/181220039-bi-" + strTime +".txt", "a", encoding="utf-8")
    while sentence:
        segSentenceList = bimm_enhance(Dic, windowSize, A, B, Pi, sentence)
        segSentenceStr = ''
        for idx in range(len(segSentenceList) - 1):
            segSentenceStr = segSentenceStr + segSentenceList[idx] + ' '
        segSentenceStr = segSentenceStr + segSentenceList[-1] + '\n'
        outputFile.write(segSentenceStr)

        #break
        sentence = testSet.readline().strip()
    testSet.close()
    outputFile.close()
    #print(windowSize)

if __name__ == "__main__":
    apply_all()
    #unit_test()

