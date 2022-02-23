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
datasets.py

Created on Jun 4, 2013
'''

import csv

import numpy
from sklearn.preprocessing import StandardScaler

from mimicus.tools.featureedit import FeatureDescriptor

def csv2numpy(csv_in):
    '''
    Parses a CSV input file and returns a tuple (X, y) with 
    training vectors (numpy.array) and labels (numpy.array), respectfully. 
    
    csv_in - name of a CSV file with training data points; 
                the first column in the file is supposed to be named 
                'class' and should contain the class label for the data 
                points; the second column of this file will be ignored 
                (put data point ID here). 
    '''
    # 解析CSV输入文件，并返回一个元组（X，y），分别包含训练向量（numpy.array）和标签（numpy.array）。
    # 文件中的第一列应该命名为'class'    此文件的第二列将被忽略（将数据点ID放在此处）。

    # Parse CSV file
    # 读取每一行
    csv_rows = list(csv.reader(open(csv_in, 'rb')))
    classes = {'FALSE':0, 'TRUE':1}
    rownum = 0
    # Count exact number of data points
    # 计算数据点的确切数量  TOTAL_ROWS数据总行数
    TOTAL_ROWS = 0
    for row in csv_rows:
        if row[0] in classes:
            # Count line if it begins with a class label (boolean)
            TOTAL_ROWS += 1
    # X = vector of data points, y = label vector
    # X表示特征向量  y表示标签向量
    X = numpy.array(numpy.zeros((TOTAL_ROWS, FeatureDescriptor.get_feature_count())), dtype=numpy.float64, order='C')
    y = numpy.array(numpy.zeros(TOTAL_ROWS), dtype=numpy.float64, order='C')
    file_names = []
    for row in csv_rows:
        # Skip line if it doesn't begin with a class label (boolean)
        if row[0] not in classes:
            continue
        # Read class label from first row
        # 读取第一行数据，y[0]=0或者1
        y[rownum] = classes[row[0]]
        featnum = 0
        # 文件名存放位置file_names
        file_names.append(row[1])
        # 从三列开始是特征
        for featval in row[2:]:
            if featval in classes:
                # Convert booleans to integers
                featval = classes[featval]
            # X[0,0]表示第1行的第1个特征向量  X[0,1]...X[0,134]表示第1行的第2个特征
            X[rownum, featnum] = float(featval)
            featnum += 1
        # 开始第二行
        rownum += 1
    # 返回每行的特征的向量表示  标签  文件名列表
    return X, y, file_names

def numpy2csv(csv_out, X, y, file_names=None):
    '''
    Creates a CSV file from the given data points (X, scipy matrix) and labels 
    (y, numpy.array). The CSV file has a header. The first column is named 
    'class' and the others after PDFrate features. All features are written 
    in their respective type format (e.g., True/False for booleans). 
    
    If 'csv_out' is an open Python file, it will not be reopened. If 
    it is a string, a file will be created with that name. 
    '''
    # 从给定的数据点（X，scipy矩阵）和标签（y，numpy.array）创建CSV文件。
    # 第一列表示分类   其他表示特征  所有特征均以各自的类型格式编写
    we_opened_csvfile = type(csv_out) == str
    csvfile = open(csv_out, 'wb+') if we_opened_csvfile else csv_out
    # Write header
    csvfile.write('class')
    if file_names:
        csvfile.write(',filename')
    # 获取特征名称
    names = FeatureDescriptor.get_feature_names()
    for name in names:
        csvfile.write(',{}'.format(name))
    csvfile.write('\n')
    # 对于没一个特征都生成一个描述，表示是否可以修改。y表示可以修改  n表示无法修改  m表示无法直接修改，但是可能会通过修改其他的特征影响到该特征
    descs = FeatureDescriptor.get_feature_descriptions()
    print descs
    # Write data
    for i in range(0, X.shape[0]):
        csvfile.write('{}'.format('TRUE' if bool(y[i]) else 'FALSE'))
        if file_names:
            csvfile.write(',{}'.format(file_names[i]))
        for j in range(0, X.shape[1]):
            feat_type = descs[names[j]]['type']
            feat_val = X[i, j]
            if feat_type == bool:
                feat_val = 'TRUE' if feat_val >= 0.5 else 'FALSE'
            elif feat_type == int:
                feat_val = int(round(feat_val))
            csvfile.write(',{}'.format(feat_val))
        csvfile.write('\n')
    
    if we_opened_csvfile:
        csvfile.close()

# 标准化数据。对于每个数据点，分别减去每个特征的平均值并除以标准偏差
# 如果未提供“标准化器”（sklearn.preprocessing.StandardScaler），则将创建一个标准化器并将其安装到输入CSV文件中的数据集上。
# 返回标准化器，以便您可以将其用于其他数据集。
def standardize_csv(csv_in, csv_out, standardizer=None):
    '''
    Standardizes data (subtracts the mean and divides by the standard deviation 
    every feature independently for every data point) from a CSV file 'csv_in' 
    and writes it into 'csv_out'. If no 'standardizer' 
    (sklearn.preprocessing.StandardScaler) is provided, one will be created 
    and fit on the dataset from the input CSV file. 
    
    Returns the standardizer so you can use it for other datasets. 
    '''
    # csv转numpy   X每行特征的向量表示  y每行的标签  file_names:文件名数组
    X, y, file_names = csv2numpy(csv_in)
#     X = X.todense()
#     标准化器不存在就从sklearn.preprocessing.StandardScaler创建一个
    if standardizer is None:
        standardizer = StandardScaler(copy=False)
        standardizer.fit(X)
    standardizer.transform(X)
    # 标准化之后再将数组转成csv文件  输出到csv_out中
    numpy2csv(csv_out, X, y, file_names)
    del X
    return standardizer
