#一个字符可能的四种标签: 开头, 结尾, 中间, 单字
labels = ['B', 'E', 'S', 'M'] 

#得到一个列表的中位数
def get_median(aList):
    aList.sort()
    half = len(aList) // 2
    return (aList[half] + aList[~half]) / 2