from math import log
import operator

#dataSet: 数据集
def calcShannonEnt(dataSet): #计算香农熵
    numEntries = len(dataSet) #数据集中实例的总数
    labelCounts = {} #创建一个字典
    for featVec in dataSet:#统计最后一列所有数值出现的概率
        currentLabel = featVec[-1] #键值为当前数据集的最后一列数值
        if currentLabel not in labelCounts.keys(): #若键值不在字典中，就将当前键值加入字典
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries #求出每种键值出现的概率
        shannonEnt -=prob * log(prob,2) #根据公式计算
    return shannonEnt

def createDataSet():
    dataSet = [[1, 1, 'yes'],
            [1, 1, 'yes'],
            [1, 0, 'no'],
            [0, 1, 'no'],
            [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels
"""

dataSet: 待划分的数据集
axis: 划分数据集的特征
value: 需要返回的特征的值

"""
def spliDataSet(dataSet, axis, value):# 按照给定特征的特征值划分数据集
    retDataSet = [] #创建一个新的列表对象
    for featVec in dataSet:
        if featVec[axis] == value: #如果符合给定的特征值
            reducedFeatVec = featVec[:axis] 
            reducedFeatVec.extend(featVec[axis + 1:]) #将符合特征的数据添加到新的列表中
            retDataSet.append(reducedFeatVec) #将当前列表作为一个完整的列表添加到新的列表对象中
    return retDataSet

def chooseBestFeatureToSplit(dataSet):#选出最好的特征值划分方式
    numFeatures = len(dataSet[0]) - 1 #由数据集第0行判断出有多少种特征
    baseEntropy = calcShannonEnt(dataSet) #求出数据集的香农熵
    bestInfoGain = 0.0; bestFeature = -1 #bestInfoGain为熵值差最大的差值 bestFeature为熵最小的特征值（最好的特征值划分方式）
    for i in range(numFeatures):
        featlist = [example[i] for example  in dataSet]#遍历所有特征   
        uniqueVals = set(featlist) #获取dataSet的第i个所有特征 #集合数据类型用来存唯一的元素值 set为集合数据类型 
        newEntropy = 0.0 #按特征值划分的香农熵
        for value in uniqueVals:
            subDataSet = spliDataSet(dataSet, i, value) #按照value特征值划分数据集
            prob = len(subDataSet)/float(len(dataSet)) #根据公式计算按照满足value值的数据集的概率
            newEntropy += prob * calcShannonEnt(subDataSet) #根据公式计算按照满足value值的数据集的香农熵
        infoGain = baseEntropy - newEntropy #原始数据集的香农熵和按特征值划分的熵值的比较
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain #满足条件就更新熵的差值
            bestFeature = i #满足条件就更新为熵最小的特征值
    return bestFeature #返回最好划分方式的特征值的位置

def majorityCnt(classList):#返回出现次数最多的分类名称
    classCount = {}
    for vote in classList: 
        if vote not in classCount.keys(): classCount[vote] = 0 
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True) #数据由大到小排序
    return sortedClassCount[0][0]

def createTree(dataSet, labels): #创建决策树
    classList = [example[-1] for example in dataSet]#取分类标签
    if classList.count(classList[0]) == len(classList): #如果类别完全相同则停止继续划分
        return classList[0]
    if len(dataSet[0]) == 1:   #遍历完所有特征时返回出现次数最多的类标签
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}} #根据最优标签生成树
    del(labels[bestFeat]) #删除已经使用过的特征标签
    featValues = [example[bestFeat] for example in dataSet] #得到训练集中所有最优特征的属性值
    uniqueVals = set(featValues) #去掉重复的属性值
    for value in uniqueVals:  #遍历特征，创建决策树
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(spliDataSet(dataSet, bestFeat, value), subLabels) #递归建立决策树
    return myTree


if __name__ == '__main__':
    myDat, labels = createDataSet()
    #print(myDat)
    #print(calcShannonEnt(myDat))
    #print('\n')
    #print(spliDataSet(myDat, 0, 1)) #按第0个特征值为1的情况划分
    #print(chooseBestFeatureToSplit(myDat))
    myTree = createTree(myDat, labels)
    print(myTree)



