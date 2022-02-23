# -*- coding: UTF-8 -*-
'''
Copyright 2013, 2014 Nedim Srndic, University of Tuebingen

This file is part of Mimicus.

Mimicus is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Mimicus is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Mimicus.  If not, see <http://www.gnu.org/licenses/>.
##############################################################################
sklearn_SVC.py

Implements the sklearn_SVC class.

Created on May 23, 2013.
'''

import pickle

from sklearn.svm import SVC

class sklearn_SVC(SVC):
    '''
    A class representing the Support Vector Machine classifier as implemented 
    by scikit-learn. 
    '''
    # 一个表示支持向量机分类器的类，由scikit-learn实现。
    # SVM模型有两个非常重要的参数C与gamma。其中
    # C是惩罚系数，即对误差的宽容度。c越高，说明越不能容忍出现误差, 容易过拟合。C越小，容易欠拟合。C过大或过小，泛化能力变差
    # gamma是选择RBF函数作为kernel后，该函数自带的一个参数。隐含地决定了数据映射到新的特征空间后的分布，gamma越大，支持向量越少，gamma值越小，支持向量越多。支持向量的个数影响训练与预测的速度。

    def __init__(self, 
                 C=10, # Found using grid search
                 kernel='rbf', #表示算法使用高斯核函数
                 degree=3, 
                 gamma=0.01, # Found using grid search
                 coef0=0.0, 
                 shrinking=True, 
                 probability=False, 
                 tol=0.001, 
                 cache_size=200, 
                 class_weight=None, 
                 verbose=False, 
                 max_iter=-1):
        '''
        Constructor
        '''
        super(sklearn_SVC, self).__init__(C=C, kernel=kernel, degree=degree, gamma=gamma, 
                       coef0=coef0, shrinking=shrinking, probability=probability, 
                       tol=tol, cache_size=cache_size, class_weight=class_weight, 
                       verbose=verbose, max_iter=max_iter)
    
    def save_model(self, modelfile):
        '''
        Saves a trained SVM model into the specified file. 
        
        modelfile - name of the file where the model should be saved.
        '''
        pickle.dump(self.__dict__, open(modelfile, 'wb+'))
    
    def load_model(self, modelfile):
        '''
        Loads a trained SVM model from the specified file. 
        
        modelfile - name of the file where the model is saved.
        '''
#         self.svc = pickle.load(open(modelfile, 'rb'))
        self.__dict__.update(pickle.load(open(modelfile, 'rb')))
