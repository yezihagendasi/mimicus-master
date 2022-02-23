'''
Copyright 2013, 2014 Nedim Srndic, Pavel Laskov, University of Tuebingen

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
mimicry.py

Implementation of the mimicry attack.

Created on July 1, 2013.
'''

import os
import random
import sys

from mimicus.tools.featureedit import FeatureEdit

def mimicry(wolf_path, targets, classifier, 
            standardizer=None, verbose=False, trials=30):
    # 对于每个恶意文件，模拟随机的良性文件“trials”，并使用“分类器”对结果进行分类，以找到最佳模拟样本。
    '''
    For every malicious file, mimic random benign files 'trials' times and 
    classify the result using 'classifier' to find the best mimicry 
    sample. 
    '''
    # wolf_path:恶意文件位置  targets:良性文件压缩后的文件名和对应的特征向量 classifier：分类器   standardizer：标准化器
    # 提取恶意样本的特征
    wolf = FeatureEdit(wolf_path)
    # 参照性最好的良性样本路径
    best_ben_path = ''
    mimic_paths = set()
    # best_mimic_score：最好的模拟样本分数    best_mimic_path：最好的模拟样本的路径
    best_mimic_score, best_mimic_path = 1.1, ''
    # retrieve_feature_vector_numpy：将特征值转换成numpy数组
    wolf_feats = wolf.retrieve_feature_vector_numpy()
    # 如果有标准化器，将向量进行标准化
    if standardizer:
        standardizer.transform(wolf_feats)
    # decision_function有符号，大于0表示正样本的可信度大于负样本，否则可信度小于负样本
    wolf_score = classifier.decision_function(wolf_feats)[0, 0]
    if verbose:
        sys.stdout.write('  Modifying {path} [{score}]:\n'
                         .format(path=wolf_path, score=wolf_score))
    # trials：表示试验次数，此时设为30
    for rand_i in random.sample(range(len(targets)), trials):
        target_path, target = targets[rand_i]
        # 修改文件
        mimic = wolf.modify_file(target.copy())
        mimic_feats = mimic['feats']
        if standardizer:
            standardizer.transform(mimic_feats)
        mimic_score = classifier.decision_function(mimic_feats)[0, 0]
        if verbose:
            sys.stdout.write('    ..trying {path}: [{score}]\n'
                             .format(path=target_path, score=mimic_score))
        # best_mimic_score：1.1
        if mimic_score < best_mimic_score:
            best_mimic_score = mimic_score
            best_ben_path = target_path
            best_mimic_path = mimic['path']
        mimic_paths.add(mimic['path'])
    if verbose:
        sys.stdout.write('  BEST: {path} [{score}]\n'
                         .format(path=best_ben_path, score=best_mimic_score))
        sys.stdout.write('  WRITING best to: {}\n\n'.format(best_mimic_path))
    # Remove all but the best mimic file
    for mimic_path in mimic_paths:
        if mimic_path != best_mimic_path:
            os.remove(mimic_path)
    return best_ben_path, best_mimic_path, best_mimic_score, wolf_score

