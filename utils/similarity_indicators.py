#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 30/11/2021 22:19
# @Author  : Mr. Y
# @Site    : 
# @File    : similarity_indicators.py
# @Software: PyCharm

import numpy as np
np.seterr(divide='ignore',invalid='ignore')
import time
import os
from utils import Initialize
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def Calculation_AUC(MatrixAdjacency_Train, MatrixAdjacency_Test, Matrix_similarity, MaxNodeNum):
    AUC_TimeStart = time.perf_counter()
    print('    Calculation AUC......')
    AUCnum = 672400

    Matrix_similarity = np.triu(Matrix_similarity - Matrix_similarity * MatrixAdjacency_Train)
    Matrix_NoExist = np.ones(MaxNodeNum) - MatrixAdjacency_Train - MatrixAdjacency_Test - np.eye(MaxNodeNum)

    Test = np.triu(MatrixAdjacency_Test)
    NoExist = np.triu(Matrix_NoExist)

    Test_num = len(np.argwhere(Test == 1))
    NoExist_num = len(np.argwhere(NoExist == 1))

    Test_rd = [int(x) for index, x in enumerate((Test_num * np.random.rand(1, AUCnum))[0])]
    NoExist_rd = [int(x) for index, x in enumerate((NoExist_num * np.random.rand(1, AUCnum))[0])]

    TestPre = Matrix_similarity * Test
    NoExistPre = Matrix_similarity * NoExist

    TestIndex = np.argwhere(Test == 1)
    Test_Data = np.array([TestPre[x[0], x[1]] for index, x in enumerate(TestIndex)]).T
    NoExistIndex = np.argwhere(NoExist == 1)
    NoExist_Data = np.array([NoExistPre[x[0], x[1]] for index, x in enumerate(NoExistIndex)]).T

    Test_rd = np.array([Test_Data[x] for index, x in enumerate(Test_rd)])
    NoExist_rd = np.array([NoExist_Data[x] for index, x in enumerate(NoExist_rd)])

    n1, n2 = 0, 0
    for num in range(AUCnum):
        if Test_rd[num] > NoExist_rd[num]:
            n1 += 1
        elif Test_rd[num] == NoExist_rd[num]:
            n2 += 0.5
        else:
            n1 += 0
    auc = float(n1 + n2) / AUCnum
    print('    AUC指标为：%f' % auc)
    AUC_TimeEnd = time.perf_counter()
    print('    AUCTime：%f s' % (AUC_TimeEnd - AUC_TimeStart))
    return auc

def AA(MatrixAdjacency_Train):
    similarity_StartTime = time.perf_counter()

    logTrain = np.log(sum(MatrixAdjacency_Train))
    logTrain = np.nan_to_num(logTrain)
    logTrain.shape = (logTrain.shape[0], 1)
    MatrixAdjacency_Train_Log = MatrixAdjacency_Train / logTrain
    MatrixAdjacency_Train_Log = np.nan_to_num(MatrixAdjacency_Train_Log)

    Matrix_similarity = np.dot(MatrixAdjacency_Train, MatrixAdjacency_Train_Log)

    similarity_EndTime = time.perf_counter()
    print("    SimilarityTime: %f s" % (similarity_EndTime - similarity_StartTime))
    return Matrix_similarity

def Jaccavrd(MatrixAdjacency_Train):
    similarity_StartTime = time.perf_counter()

    Matrix_similarity = np.dot(MatrixAdjacency_Train, MatrixAdjacency_Train)

    deg_row = sum(MatrixAdjacency_Train)
    deg_row.shape = (deg_row.shape[0], 1)
    deg_row_T = deg_row.T
    tempdeg = deg_row + deg_row_T
    temp = tempdeg - Matrix_similarity

    Matrix_similarity = Matrix_similarity / temp

    similarity_EndTime = time.perf_counter()
    print("    SimilarityTime: %f s" % (similarity_EndTime - similarity_StartTime))
    return Matrix_similarity

def RWR(MatrixAdjacency_Train):
    similarity_StartTime = time.perf_counter()

    Parameter = 0.85

    Matrix_TransitionProbobility = MatrixAdjacency_Train / sum(MatrixAdjacency_Train)
    Matrix_EYE = np.eye(MatrixAdjacency_Train.shape[0])

    Temp = Matrix_EYE - Parameter * Matrix_TransitionProbobility.T
    INV_Temp = np.linalg.inv(Temp)
    Matrix_RWR = (1 - Parameter) * np.dot(INV_Temp, Matrix_EYE)
    Matrix_similarity = Matrix_RWR + Matrix_RWR.T

    similarity_EndTime = time.perf_counter()
    print("    SimilarityTime: %f s" % (similarity_EndTime - similarity_StartTime))
    return Matrix_similarity

def Cn(MatrixAdjacency_Train):
    similarity_StartTime = time.perf_counter()
    Matrix_similarity = np.dot(MatrixAdjacency_Train, MatrixAdjacency_Train)

    similarity_EndTime = time.perf_counter()
    print("    SimilarityTime: %f s" % (similarity_EndTime - similarity_StartTime))
    return Matrix_similarity

def Salton(MatrixAdjacency_Train):
    similarity_StartTime = time.clock()
    similarity = np.dot(MatrixAdjacency_Train, MatrixAdjacency_Train)
    deg_row = sum(MatrixAdjacency_Train)
    deg_row.shape = (deg_row.shape[0], 1)
    deg_row_T = deg_row.T
    tempdeg = np.dot(deg_row, deg_row_T)
    temp = np.sqrt(tempdeg)
    Matrix_similarity = np.nan_to_num(similarity / temp)
    similarity_EndTime = time.clock()
    print("    SimilarityTime: %f s" % (similarity_EndTime - similarity_StartTime))

    return Matrix_similarity
