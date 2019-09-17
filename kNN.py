from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
from os import listdir

#def createDataSet():  #创建数据集和标签
    #group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
   # labels = ['A','A','B','B'] 
   # return group, labels

#group, labels = createDataSet()

def classify0(inX, dataSet, lables, k): #判断inX的分类类别
    dataSetSize = dataSet.shape[0]   #返回dataSet的行数
    diffMat = tile(inX, (dataSetSize,1)) - dataSet #将传进来的inX扩展成和dataSet一样大小的矩阵，然后和dataSet相减
    sqDiffMat = diffMat**2 #平方（14-16行为距离公式）
    sqDistances = sqDiffMat.sum(axis = 1)  #将矩阵的每一行分别求和
    distances = sqDistances**0.5  #求根号
    sortedDistIndicies = distances.argsort()   #返回distances中元素从小到大排序后的索引值
    classCount = {} #定一个记录类别次数的字典
    for i in range(k):#求前k个数的距离
        voteIlabel = lables[sortedDistIndicies[i]] 
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
        #dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
        #计算类别次数
    sortedClassCount = sorted(classCount.items(),#python3中用items()替换python2中的iteritems()
                              key=operator.itemgetter(1), reverse=True)
    #key=operator.itemgetter(1)根据字典的值进行排序
    #key=operator.itemgetter(0)根据字典的键进行排序
    #reverse=True降序排序字典/= False为升序排序
    #sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0] #返回次数最多的类别,即所要分类的类别

#print(classify0([0,0],group,labels,3))  #判断[0,0]点的类别
#print(classify0([1,0.8],group,labels,3))

def file2matrix(filename): #从文件中读取数据
    fr = open(filename) #打开文件
    arrayOLines = fr.readlines()  #读取文件内容
    numberOfLines = len(arrayOLines)  #获取文件行数
    returnMat = zeros((numberOfLines,3)) #返回的NumPy矩阵,解析完成的数据:numberOfLines行,3列 用0填充
    classLabelVector = [] #分类标签向量
    index = 0 #行的索引值
    for line in arrayOLines:
        line = line.strip() #s.strip(rm)，当rm空时,默认删除空白符(包括'\n','\r','\t',' ')
        listFromLine = line.split('\t') #使用s.split(str="",num=string,cout(str))将字符串根据'\t'分隔符进行切片
        returnMat[index,:] = listFromLine[0:3] #将数据前三列提取出来,存放到returnMat矩阵中,作为特征矩阵
        classLabelVector.append(int(listFromLine[-1])) #将列表的最后一列存储到向量classLabelVector中
        index += 1
    return returnMat,classLabelVector
    
def autoNorm(dataSet): #（归一化特征值）将任意取值范围的值转化为（0,1）之间的值（newValue = (oldValue-min)/(max-min)）
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0] #得到dataSet的行数
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

def datingClassTest(): #测试分类器的错误率
    horatio = 0.10
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt') #读取文件
    normMat, ranges, minVals = autoNorm(datingDataMat)#归一化
    m = normMat.shape[0] #数据集行数
    numTestVecs = int(m*horatio) #确定测试样本和训练集样本
    errorCount = 0.0 #统计错误数
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],\
            datingLabels[numTestVecs:m],3)
        print( "the classifier came back with: %d, the real answer is: %d"\
            % (classifierResult,datingLabels[i]))
        if (classifierResult != datingLabels[i]): 
            errorCount +=1.0
    print ("the total error rate is: %f" % (errorCount/float(numTestVecs)))
def classifyPerson(): #输入数据，利用kNN算法进行分类
    resultList = ['not at all','in small doses', 'in large doses']
    percentTats = float(input(\
        "percentage of time spent playing video games?"))
    ffMiles = float(input(\
        "frequent flier miles earned per year?"))
    iceCream = float(input(\
        "liters of ice cream consumed per year?"))
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals)/ranges, normMat, datingLabels,3)
    print ("You will probably like this person:",\
        resultList[classifierResult - 1])

##################################手写数字识别################################

def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():#手写数字识别
    hwLabels = []
    trainingFileList = listdir('trainingDigits') #listdir()可以列出给定目录的文件名
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')
        classNumStr = int(fileNameStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileNameStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d"\
            % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount/float(mTest)))


if __name__ == '__main__':
    #filename = "datingTestSet2.txt"# datingTestSet2列表的最后一列是数字才对应48行的int()强制转换
    #datingDataMat,datingLabels = file2matrix(filename)
    #fig = plt.figure() #创建一幅图像，使用默认参数，figure(num=None, figsize=None, dpi=None, facecolor=None, edgecolor=None, frameon=True)
    #ax = fig.add_subplot(111)  #111是单个整数编码的子绘图网格参数，“111”表示“1×1网格，第一子图”
    #ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*array(datingLabels),15.0*array(datingLabels)) #后两个参数用色彩标识了数据点的类型
    #plt.show() #显示图像
    #normMat, ranges, minVals = autoNorm(datingDataMat)
    #datingClassTest()
    #classifyPerson()
    #testVector = img2vector('testDigits/0_13.txt')
    #print(testVector[0,0:31])
    handwritingClassTest()











    
