import numpy as np 
import time
from math import log
import MyUtils
from MyUtils import get_median

#测试集路径
testSetPath = "./test.txt"
#模型版本号
modelVer = "2020-09-20-11_23_01"
#一个字符可能的四种标签
labels = MyUtils.labels

#导入转移矩阵, 发射概率矩阵, 初始概率矩阵
def import_models():
    a = np.load("./Models/" + modelVer + "-A.npy", allow_pickle=True).item()
    b = np.load("./Models/" + modelVer + "-B.npy", allow_pickle=True).item()
    p = np.load("./Models/" + modelVer + "-Pi.npy", allow_pickle=True).item()
    
    return a, b, p

#Viterbi算法对句子进行标注
def viterbi(sentence, A, B, Pi):
    sentenceLength = len(sentence)
    dpTab = [{}] #动态规划表
    seqAll = {} #记录到当前字符位置为止的标签顺序, 最终只用在4个可能标签中取最佳
                #键是4种标签, 值是当前位置之前的标签顺序
    
    if sentence[0] not in B['B']: #当句子第一个字符没在训练集出现过, 那么不在B['B']中, 也就不在B中
        for state in labels:
            if state == 'S': #认为它是一个单字, 并设置发射概率
                B[state][sentence[0]] = log(0.5)
            else:
                B[state][sentence[0]] = float("-Inf")
    for label in labels:
        dpTab[0][label] = dpTab[0].get(label, 0.0) + (Pi[label] + B[label][sentence[0]])
        seqAll[label] = [label]

    for idx in range(1, sentenceLength):
        dpTab.append({}) #增加一个当前位置字符的各标签概率表
        seqNow = {}  #到当前位置的几种顺序

        for curLabel in labels: #idx位置字符可能的状态值
            choices = [] #记录前一字符到当前字符各种标签顺序的概率值
            for prevLabel in labels: #(idx - 1)位置字符的状态值
                if sentence[idx] not in B[curLabel]: #当前位置字符不在发射概率矩阵中时
                    emitProb = get_median(list(B[curLabel].values())) #以中位数作为发射概率
                    #print("Median of emitProb of %s: %f"%(curLabel, emitProb))
                    B[curLabel][sentence[idx]] = emitProb
                prob = dpTab[idx - 1][prevLabel] + A[prevLabel][curLabel] + B[curLabel][sentence[idx]]
                choices.append((prob, prevLabel))
            
            bestChoice = max(choices)
            dpTab[idx][curLabel] = bestChoice[0]
            seqNow[curLabel] = seqAll[bestChoice[1]] + [curLabel]
        seqAll = seqNow

    bestLastLabel = max(dpTab[-1], key=dpTab[-1].get)
    return seqAll[bestLastLabel]

#根据标注对句子进行切分
def seg_sentence(sentence, tags): 
    segSentence = []
    assert(len(sentence) == len(tags))
    if tags[-1] != 'S' and tags[-1] != 'E':
        if tags[-2] == 'B' or tags[-2] == 'M':
            tags[-1] = 'E'
        else:
            tags[-1] = 'S'

    sentenceLength = len(sentence)
    idx = 0
    while idx < sentenceLength:
        if tags[idx] == 'B':
            cursor = idx + 1
            while tags[cursor] != 'E' and cursor < sentenceLength:
                cursor += 1
            if cursor != sentenceLength:
                segSentence.append(sentence[idx: cursor + 1])
                idx = cursor + 1
            else:
                segSentence.append(sentence[idx: sentenceLength])
                break 
        else:
            segSentence.append(sentence[idx])
            idx += 1

    return segSentence

#给定位置和标签列表, 正向地找到最近的切分点
def seg_substr_f(tags, idx):
    if tags[idx] == 'B' or tags[idx] == 'M':
        end = idx + 1
        while end < len(tags):
            if tags[end] == 'E':
                end += 1
                return end
            end += 1
        return end
    else: 
        return (idx + 1)

#给定位置和标签列表, 逆向地找到最近的切分点
def seg_substr_b(tags, idx):
    if tags[idx] == 'E' or tags[idx] == 'M':
        begin = idx - 1
        while begin >= 0:
            if tags[begin] == 'B':
                return begin
            begin -= 1
        return (begin + 1)
    else: 
        return idx 

#对测试集进行划分
def apply_all():
    A, B, Pi = import_models()
    testSet = open(testSetPath, encoding="utf-8")
    
    strTime = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(int(time.time())))
    outputFile = open("./Results/181220039-H-" + strTime +".txt", "a", encoding="utf-8")
    sentence = testSet.readline().strip()
    while sentence:
        sentencePreLabels = viterbi(sentence, A, B, Pi)
        segSentenceList = seg_sentence(sentence, sentencePreLabels)
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