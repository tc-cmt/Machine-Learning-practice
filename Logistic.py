####说明：Logistic回归算法为将每个特征乘上一个回归系数，其值相加，代入sigmoid函数中得到0-1的数值####

import numpy as np

'''
功能：读取数据
日期：2019/10/16
返回值:
    dataMat - 数据列表
    labelMat - 标签列表
'''
def loadDataSet():
    dataMat = []   #创建数据列表      
    labelMat = []  #创建标签列表    
    fr = open('testSet.txt')  #打开文本
    for line in fr.readlines(): #逐行读取
        lineArr = line.strip().split()  #去回车，放入列表
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])  #添加数据
        labelMat.append(int(lineArr[2])) #添加标签
    return dataMat,labelMat

'''
功能：sigmoid函数（阶跃函数）
日期：2019/10/16
'''
def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))

'''
功能：梯度上升算法（确定最佳回归系数）
日期：2019/10/16
参数：
    dataMatIn - 2维numpy数组（数据集）
    classLabels - 类别标签
返回值：
    weights - 回归系数
'''
def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn) #数组转换成numpy矩阵的数据类型
    labelMat = np.mat(classLabels).transpose() #转换成numpy矩阵的数据类型并进行转置（将行向量转为列向量）
    m, n = np.shape(dataMatrix)  #返回dataMatrix的大小
    alpha = 0.001  #想目标移动的步长
    maxCycles = 500 #迭代次数（循环次数）
    weights = np.ones((n, 1)) #ones函数可以创建任意维度和元素个数的数组，其元素值均为1
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h) #真实类别与技术类别的差值
        weights = weights + alpha * dataMatrix.transpose() * error #梯度上升算法的迭代公式
    return weights

'''
功能：画出数据集和Logistic回归最佳拟合函数
日期：2019/10/17
参数：
    weights - 回归系数
'''
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    #weights = weights.getA() #将矩阵类型转化为数组
    dataMat, labelMat = loadDataSet() #加载数据
    dataArr = np.array(dataMat) #转换成numpy的array数组
    n = np.shape(dataArr)[0] #数据个数
    xcord1 = []; ycord1 = [] #正样本    
    xcord2 = []; ycord2 = [] #负样本
    for i in range(n): #根据数据集标签进行分类
        if int(labelMat[i]) == 1: #1为正样本
            xcord1.append(dataArr[i, 1]); ycord1.append(dataArr[i,2])
        else: #0为负样本
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111) #添加subplot
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s') #绘制正样本
    ax.scatter(xcord2, ycord2, s=30, c='green') #绘制负样本
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1]*x) / weights[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2') #绘制label
    plt.show()   

'''
function: 随机梯度上升算法
参数：
    dataMatrix 
    classLabels - 类别标签
日期：
    2019/10/21
'''
def stocGradAscent0(dataMatrix, classLabels):
    m, n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

if __name__ == '__main__':
    dataArr,labelMat = loadDataSet()
  #  plotBestFit(gradAscent(dataArr, labelMat)) #(测试梯度上升算法得取消注释)weights = weights.getA() #将矩阵类型转化为数组
    plotBestFit(stocGradAscent0(np.array(dataArr),labelMat))



